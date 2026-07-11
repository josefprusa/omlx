# SPDX-License-Identifier: Apache-2.0
"""int8 MLA-KV cache: group-quantized latent, dense rope keys, start-threshold.

MLA models (GLM-5.2 glm_moe_dsa, DeepSeek-V3.2/V4 family, Kimi) store per layer
a ``CacheList(latent_cache[, indexer_cache])``. The latent cache holds the
low-rank compressed latent (``kv_lora_rank`` wide, in the *keys* slot) plus the
rope key ``k_pe`` (in the *values* slot). This cache int8 (affine, group along
the feature axis) quantizes ONLY the latent and keeps ``k_pe`` dense.

The core is model-agnostic: the class takes ``(latent_dim, start, bits,
group_size)`` and bakes in no model constants. Model wiring (which cache slot
is the latent, group size choice) lives in the model patch files.

Design rules (post-mortem of the first attempt — see GLM int8 MLA-KV notes):

- **start-threshold**: below ``start`` tokens the cache stores plain fp16/bf16
  buffers, bit-identical to ``mlx_lm.models.cache.KVCache``. When
  ``offset + incoming >= start`` the whole latent history is quantized once
  (fork ``maybe_quantize_kv_cache`` semantics) and new tokens quantize on
  write. ``start=0`` quantizes from the first token.
- **quantized reads return the raw triple**: ``update_and_fetch`` hands back
  ``(packed, scales, biases)`` for the latent once quantized (fork
  ``QuantizedKVCache`` behavior). The attention layer dequantizes right after
  the fetch (or feeds the triple to an int8-native kernel); this module never
  materializes a dense full-context latent on its own.
- **``.state`` is raw storage** (dense 2-tuple or quantized 4-tuple): the
  scheduler evaluates ``c.state`` every prefill chunk, so ``.state`` must
  never dequantize (a per-chunk full-latent dequant is the exact fixed
  transient that death-spiraled prefill chunking in the first attempt).
- **hot/cold blocks persist NATIVE int8** (``native_kv_state()``): once
  quantized, blocks store the raw ``(packed, scales, biases, k_pe)`` 4-tuple
  under class name ``Int8MLAKVCache`` so restore is read-and-go (zero
  requantization — a legacy fp16 256k restore spent ~80s requantizing and
  read 2x the bytes). Below ``start`` (dense) the export stays the plain
  fp16 ``KVCache`` 2-tuple, byte-identical to legacy blocks. Format
  detection on restore is the persisted sub-cache class name + element
  count: legacy fp16 blocks (class ``KVCache``, 2 elements) still restore
  through the old path (scheduler-hook requantization); native blocks
  restored into an int8-OFF session are dequantized on load by the same
  scheduler hook (cross-mode compatibility both directions).
  ``fp16_kv_state()`` remains as the legacy/doctrine fallback export.
- Sequence-axis (axis 2) slicing of the quantized triple is exact because
  quantization groups run along the FEATURE axis: ``dequantize(x[..., a:b, :])
  == dequantize(x)[..., a:b, :]`` bit-for-bit (covered by tests).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import mlx.core as mx

# NOTE: cache.py's create_attention_mask is the builder that takes ``offset``;
# base.py's same-named function delegates to make_mask (importing that one
# causes infinite delegation / a TypeError on ``offset``).
from mlx_lm.models.cache import (
    _BaseCache,
    create_attention_mask,
    create_causal_mask,
    dynamic_roll,
)

from omlx.cache.type_handlers import (
    CacheStateAxisInfo,
    CacheType,
    KVCacheHandler,
)

logger = logging.getLogger(__name__)


def _validate_latent_dim(dim: int, group_size: int) -> None:
    if dim % group_size != 0:
        raise ValueError(
            f"int8 MLA-KV: latent dim {dim} not divisible by group_size "
            f"{group_size}; cannot group-quantize along the feature axis"
        )


class Int8MLAKVCache(_BaseCache):
    """Latent-quantizing MLA KV cache with a start-threshold.

    keys slot = latent (quantized past ``start``); values slot = k_pe (dense).
    Below ``start`` this is storage-identical to ``KVCache``.
    """

    step = 256

    # omlx CacheListHandler hot/cold export hook: blocks persist as plain
    # fp16 KVCache state so SSD dirs stay mode-independent (doctrine).
    fp16_kv_class_name = "KVCache"

    def __init__(
        self,
        group_size: int = 64,
        bits: int = 8,
        start: int = 0,
        latent_dim: Optional[int] = None,
    ):
        self.keys = None  # dense latent array, or [packed, scales, biases]
        self.values = None  # k_pe — always dense
        self.offset = 0
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.start = max(0, int(start))
        self.quantized = False
        if latent_dim is not None:
            _validate_latent_dim(int(latent_dim), self.group_size)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_kv(cls, kv: Any, group_size: int = 64, bits: int = 8, start: int = 0):
        """Adopt a plain KVCache (e.g. an SSD-restored fp16 latent cache).

        Quantizes the history when ``offset >= start``; otherwise stays dense.
        """
        cache = cls(group_size=group_size, bits=bits, start=start)
        cache.keys = kv.keys
        cache.values = kv.values
        cache.offset = int(kv.offset)
        if cache.keys is not None and cache.offset >= cache.start:
            cache._convert_to_quantized()
        return cache

    def _convert_to_quantized(self) -> None:
        """One-shot dense -> quantized storage conversion.

        Quantizes the full buffer (padding rows included): groups run along
        the feature axis, so each token row quantizes independently and the
        valid rows are bit-identical to quantizing the trimmed slice.
        """
        if self.quantized:
            return
        if self.keys is not None:
            _validate_latent_dim(self.keys.shape[-1], self.group_size)
            self.keys = list(
                mx.quantize(self.keys, group_size=self.group_size, bits=self.bits)
            )
        self.quantized = True

    # ------------------------------------------------------------------
    # core update
    # ------------------------------------------------------------------

    def update_and_fetch(self, keys, values):
        # keys = latent [B, 1, S, latent_dim]; values = k_pe [B, 1, S, rope_dim]
        prev = self.offset
        num_steps = keys.shape[2]

        if not self.quantized and (prev + num_steps) >= self.start:
            _validate_latent_dim(keys.shape[-1], self.group_size)
            self._convert_to_quantized()

        if not self.quantized:
            # Dense mode: bit-identical to mlx-lm KVCache.
            if self.keys is None or (prev + num_steps) > self.values.shape[2]:
                B, H, _, k_dim = keys.shape
                v_dim = values.shape[3]
                n_steps = (self.step + num_steps - 1) // self.step
                new_k = mx.zeros((B, H, n_steps * self.step, k_dim), keys.dtype)
                new_v = mx.zeros((B, H, n_steps * self.step, v_dim), values.dtype)
                if self.keys is not None:
                    if prev % self.step != 0:
                        self.keys = self.keys[..., :prev, :]
                        self.values = self.values[..., :prev, :]
                    self.keys = mx.concatenate([self.keys, new_k], axis=2)
                    self.values = mx.concatenate([self.values, new_v], axis=2)
                else:
                    self.keys, self.values = new_k, new_v
            self.offset += num_steps
            self.keys[..., prev : self.offset, :] = keys
            self.values[..., prev : self.offset, :] = values
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

        # Quantized mode.
        if self.keys is None or (prev + num_steps) > self.values.shape[2]:
            B, H, _, k_dim = keys.shape
            v_dim = values.shape[3]
            el_per_int = 8 * mx.uint32.size // self.bits
            n_steps = (self.step + num_steps - 1) // self.step
            shape = (B, H, n_steps * self.step)
            new_k = [
                mx.zeros((*shape, k_dim // el_per_int), dtype=mx.uint32),
                mx.zeros((*shape, k_dim // self.group_size), dtype=keys.dtype),
                mx.zeros((*shape, k_dim // self.group_size), dtype=keys.dtype),
            ]
            new_v = mx.zeros((*shape, v_dim), dtype=values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = [x[..., :prev, :] for x in self.keys]
                    self.values = self.values[..., :prev, :]
                self.keys = [
                    mx.concatenate([o, n], axis=2) for o, n in zip(self.keys, new_k)
                ]
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += num_steps
        q = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = q[i]
        self.values[..., prev : self.offset, :] = values

        # Raw (packed, scales, biases) triple — the caller dequantizes (or
        # feeds an int8-native kernel). Never materialize dense here.
        cur = tuple(x[..., : self.offset, :] for x in self.keys)
        return cur, self.values[..., : self.offset, :]

    # ------------------------------------------------------------------
    # serialization / bookkeeping
    # ------------------------------------------------------------------

    def size(self):
        return self.offset

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        if self.quantized:
            return sum(x.nbytes for x in self.keys) + self.values.nbytes
        return self.keys.nbytes + self.values.nbytes

    @property
    def state(self):
        """Raw storage (cheap to eval): dense 2-tuple or quantized 4-tuple.

        Never dequantizes — the scheduler evaluates ``c.state`` every prefill
        chunk. Serialization for hot/SSD blocks goes through
        ``fp16_kv_state()`` instead.
        """
        if self.keys is None:
            return (None, None) if not self.quantized else (None, None, None, None)
        if self.offset == self.values.shape[2]:
            k, v = self.keys, self.values
        else:
            v = self.values[..., : self.offset, :]
            if self.quantized:
                k = [x[..., : self.offset, :] for x in self.keys]
            else:
                k = self.keys[..., : self.offset, :]
        if self.quantized:
            return (*k, v)
        return (k, v)

    @state.setter
    def state(self, v):
        if len(v) == 2:
            self.keys, self.values = v
            self.quantized = False
        elif len(v) == 4:
            packed, scales, biases, kpe = v
            self.keys = None if packed is None else [packed, scales, biases]
            self.values = kpe
            self.quantized = True
        else:
            raise ValueError(f"Int8MLAKVCache.state: expected 2 or 4 elements, got {len(v)}")
        if self.values is not None:
            self.offset = self.values.shape[2]

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (self.offset, self.group_size, self.bits, self.start, int(self.quantized)),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.group_size, self.bits, self.start, quantized = map(int, v)
        self.quantized = bool(quantized)

    def fp16_kv_state(self):
        """Dense ``(latent, k_pe)`` export for hot/SSD blocks (fp16 doctrine).

        Blocks persisted from this cache are byte-compatible with a plain
        fp16 ``KVCache`` session sharing the cache dir.
        """
        if self.keys is None:
            return (None, None)
        v = self.values[..., : self.offset, :]
        if self.quantized:
            k = mx.dequantize(
                *(x[..., : self.offset, :] for x in self.keys),
                group_size=self.group_size,
                bits=self.bits,
            )
        else:
            k = self.keys[..., : self.offset, :]
        return (k, v)

    def native_kv_state(self):
        """Block-persist export: ``(state, class_name, meta_state)``.

        Quantized: the raw ``(packed, scales, biases, k_pe)`` 4-tuple under
        this class name — restore constructs the cache directly with zero
        requantization. Dense (below ``start``): the plain fp16 2-tuple
        under ``KVCache``, byte-identical to legacy blocks. The persisted
        class name + element count IS the block format flag.
        """
        if self.keys is None or not self.quantized:
            return self.fp16_kv_state(), "KVCache", ""
        return self.state, "Int8MLAKVCache", self.meta_state

    def to_kv(self):
        """Dequantized plain ``KVCache`` (cross-mode restore, int8 OFF)."""
        from mlx_lm.models.cache import KVCache

        kv = KVCache()
        k, v = self.fp16_kv_state()
        kv.keys = k
        kv.values = v
        kv.offset = 0 if v is None else int(v.shape[2])
        return kv

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def merge(cls, caches):
        # Continuous batching: merge per-row single caches into a batched one.
        return BatchInt8MLAKVCache.merge(caches)


class BatchInt8MLAKVCache(_BaseCache):
    """Batched (continuous-batching) variant of :class:`Int8MLAKVCache`.

    Mirrors mlx-lm ``BatchKVCache`` (left-padded, per-row offsets,
    merge/filter/extract) with the latent quantized past the start threshold.
    Padded positions dequantize to ~0 and are masked by ``left_padding``.
    """

    step = 256

    fp16_kv_class_name = "BatchKVCache"

    def __init__(
        self,
        left_padding,
        group_size: int = 64,
        bits: int = 8,
        start: int = 0,
    ):
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.start = max(0, int(start))
        self.quantized = False

    def _convert_to_quantized(self) -> None:
        if self.quantized:
            return
        if self.keys is not None:
            _validate_latent_dim(self.keys.shape[-1], self.group_size)
            self.keys = list(
                mx.quantize(self.keys, group_size=self.group_size, bits=self.bits)
            )
        self.quantized = True

    def _grow(self, B, H, k_dim, v_dim, num_steps, prev, dtype):
        n_steps = (self.step + num_steps - 1) // self.step
        shape = (B, H, n_steps * self.step)
        if self.quantized:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_k = [
                mx.zeros((*shape, k_dim // el_per_int), dtype=mx.uint32),
                mx.zeros((*shape, k_dim // self.group_size), dtype=dtype),
                mx.zeros((*shape, k_dim // self.group_size), dtype=dtype),
            ]
        else:
            new_k = mx.zeros((*shape, k_dim), dtype=dtype)
        new_v = mx.zeros((*shape, v_dim), dtype=dtype)
        if self.keys is not None:
            if prev % self.step != 0:
                if self.quantized:
                    self.keys = [x[..., :prev, :] for x in self.keys]
                else:
                    self.keys = self.keys[..., :prev, :]
                self.values = self.values[..., :prev, :]
            if self.quantized:
                self.keys = [
                    mx.concatenate([o, n], axis=2) for o, n in zip(self.keys, new_k)
                ]
            else:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
            self.values = mx.concatenate([self.values, new_v], axis=2)
        else:
            self.keys, self.values = new_k, new_v

    def update_and_fetch(self, keys, values):
        prev = self._idx
        num_steps = keys.shape[2]

        if not self.quantized and (prev + num_steps) >= self.start:
            _validate_latent_dim(keys.shape[-1], self.group_size)
            self._convert_to_quantized()

        if self.keys is None or (prev + num_steps) > self.values.shape[2]:
            B, H, _, k_dim = keys.shape
            self._grow(B, H, k_dim, values.shape[3], num_steps, prev, keys.dtype)

        self.offset += num_steps
        self._idx += num_steps

        if self.quantized:
            q = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
            for i in range(len(self.keys)):
                self.keys[i][..., prev : self._idx, :] = q[i]
            self.values[..., prev : self._idx, :] = values
            cur = tuple(x[..., : self._idx, :] for x in self.keys)
            return cur, self.values[..., : self._idx, :]

        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        return (
            self.keys[..., : self._idx, :],
            self.values[..., : self._idx, :],
        )

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError("Left padding only on an empty cache")
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            pad = self._right_padding
            if self.quantized:
                self.keys = [
                    dynamic_roll(x, pad[:, None], axis=2) for x in self.keys
                ]
            else:
                self.keys = dynamic_roll(self.keys, pad[:, None], axis=2)
            self.values = dynamic_roll(self.values, pad[:, None], axis=2)
            self.offset -= pad
            self.left_padding += pad
            self._right_padding = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if k is not None and self._idx < self.values.shape[2]:
            v = v[..., : self._idx, :]
            if self.quantized:
                k = [x[..., : self._idx, :] for x in k]
            else:
                k = k[..., : self._idx, :]
        if self.quantized:
            if k is None:
                return (None, None, None, v, self.offset, self.left_padding)
            return (*k, v, self.offset, self.left_padding)
        return (k, v, self.offset, self.left_padding)

    @state.setter
    def state(self, s):
        if len(s) == 4:
            k, v, off, lp = s
            self.keys = k
            self.quantized = False
        elif len(s) == 6:
            packed, scales, biases, v, off, lp = s
            self.keys = None if packed is None else [packed, scales, biases]
            self.quantized = True
        else:
            raise ValueError(
                f"BatchInt8MLAKVCache.state: expected 4 or 6 elements, got {len(s)}"
            )
        self.values = v
        self.offset = off
        self.left_padding = lp
        self._idx = 0 if v is None else v.shape[2]

    @property
    def meta_state(self):
        return tuple(
            map(str, (self.group_size, self.bits, self.start, int(self.quantized)))
        )

    @meta_state.setter
    def meta_state(self, v):
        self.group_size, self.bits, self.start, quantized = map(int, v)
        self.quantized = bool(quantized)

    def fp16_kv_state(self):
        """Dense BatchKVCache-shaped export (fp16 doctrine for hot/SSD)."""
        k, v = self.keys, self.values
        if k is None:
            return (None, None, self.offset, self.left_padding)
        v = v[..., : self._idx, :]
        if self.quantized:
            k = mx.dequantize(
                *(x[..., : self._idx, :] for x in k),
                group_size=self.group_size,
                bits=self.bits,
            )
        else:
            k = k[..., : self._idx, :]
        return (k, v, self.offset, self.left_padding)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        if self.keys is not None:
            if self.quantized:
                self.keys = [x[batch_indices] for x in self.keys]
            else:
                self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                if self.quantized:
                    self.keys = [x[..., min_left_pad:, :] for x in self.keys]
                else:
                    self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        """In-place extend with another batch cache (BatchGenerator reshape).

        Modes are aligned first: if either side is quantized, both convert
        (full-buffer quantize is exact per token row — groups run along the
        feature axis).
        """
        if self.quantized or getattr(other, "quantized", False):
            self._convert_to_quantized()
            if hasattr(other, "_convert_to_quantized"):
                other._convert_to_quantized()

        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate(
                [self.left_padding, other.left_padding]
            )
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        def _elems(c):
            if c.keys is None:
                return None
            ks = list(c.keys) if self.quantized else [c.keys]
            return ks + [c.values]

        ref = _elems(self if self.keys is not None else other)
        H = ref[0].shape[1]
        dims = [e.shape[3] for e in ref]
        dtypes = [e.dtype for e in ref]
        max_idx = max(self._idx, other._idx)
        max_size = max(
            self.values.shape[2] if self.keys is not None else 0,
            other.values.shape[2] if other.keys is not None else 0,
        )

        def _pad(c):
            es = _elems(c)
            if es is None:
                Bc = c.offset.shape[0]
                es = [
                    mx.zeros((Bc, H, 0, d), dtype=t)
                    for d, t in zip(dims, dtypes)
                ]
            left = max_idx - c._idx
            right = max_size - es[-1].shape[2] - left
            if right < 0:
                es = [e[..., :right, :] for e in es]
                right = 0
            if left != 0 or right != 0:
                spec = [(0, 0), (0, 0), (left, right), (0, 0)]
                es = [mx.pad(e, spec) for e in es]
            return es, c.offset, c.left_padding + left

        a_elems, a_off, a_lp = _pad(self)
        b_elems, b_off, b_lp = _pad(other)
        merged = [mx.concatenate([x, y]) for x, y in zip(a_elems, b_elems)]
        if self.quantized:
            self.keys = merged[:3]
            self.values = merged[3]
        else:
            self.keys, self.values = merged
        self.offset = mx.concatenate([a_off, b_off])
        self.left_padding = mx.concatenate([a_lp, b_lp])
        self._idx = max_idx

    def extract(self, idx):
        cache = Int8MLAKVCache(
            group_size=self.group_size, bits=self.bits, start=self.start
        )
        padding = self.left_padding[idx].item()
        if self.quantized:
            cache.keys = [
                mx.contiguous(x[idx : idx + 1, :, padding : self._idx])
                for x in self.keys
            ]
            cache.quantized = True
        else:
            cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        cache.offset = cache.values.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        # Reference params come from the first threshold-aware cache; plain
        # KVCache rows (e.g. a restored-below-start row) are tolerated as
        # dense sources.
        ref_params = next(
            (c for c in caches if isinstance(c, Int8MLAKVCache)), caches[0]
        )
        gs = getattr(ref_params, "group_size", 64)
        bits = getattr(ref_params, "bits", 8)
        start = getattr(ref_params, "start", 0)
        if max_length == 0:
            return cls([0] * len(caches), group_size=gs, bits=bits, start=start)

        quantized = any(getattr(c, "quantized", False) for c in caches)
        padding = [max_length - l for l in lengths]
        B = len(caches)
        ref = next(c for c in caches if c.keys is not None)
        H = (ref.keys[0] if getattr(ref, "quantized", False) else ref.keys).shape[1]
        Dv = ref.values.shape[3]
        dt = ref.values.dtype
        if getattr(ref, "quantized", False):
            k_dim = ref.keys[0].shape[3] * (8 * mx.uint32.size // bits)
        else:
            k_dim = ref.keys.shape[3]

        cache = cls(padding, group_size=gs, bits=bits, start=start)
        if quantized:
            cache.quantized = True
            el_per_int = 8 * mx.uint32.size // bits
            keys = [
                mx.zeros((B, H, max_length, k_dim // el_per_int), dtype=mx.uint32),
                mx.zeros((B, H, max_length, k_dim // gs), dtype=dt),
                mx.zeros((B, H, max_length, k_dim // gs), dtype=dt),
            ]
        else:
            keys = mx.zeros((B, H, max_length, k_dim), dtype=dt)
        kpe = mx.zeros((B, H, max_length, Dv), dtype=dt)

        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            o = int(c.offset)
            if quantized:
                if getattr(c, "quantized", False):
                    triple = c.keys
                else:
                    triple = mx.quantize(
                        c.keys[..., :o, :], group_size=gs, bits=bits
                    )
                for j in range(3):
                    keys[j][i : i + 1, :, p : p + o] = triple[j][..., :o, :]
            else:
                keys[i : i + 1, :, p : p + o] = c.keys[..., :o, :]
            kpe[i : i + 1, :, p : p + o] = c.values[..., :o, :]

        cache.keys = keys
        cache.values = kpe
        cache.offset += max_length
        cache._idx = max_length
        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        if self.quantized:
            return sum(x.nbytes for x in self.keys) + self.values.nbytes
        return self.keys.nbytes + self.values.nbytes


def _parse_int8_meta(meta_state: Any) -> tuple[int, int, int]:
    """``(group_size, bits, start)`` from a persisted 5-tuple meta_state.

    Meta layout: (offset, group_size, bits, start, quantized) as strings.
    Model-wiring defaults (64, 8, 0) when meta is missing or malformed.
    """
    if isinstance(meta_state, (list, tuple)) and len(meta_state) >= 4:
        try:
            _, group_size, bits, start = map(int, meta_state[:4])
            return group_size, bits, start
        except (TypeError, ValueError):
            pass
    return 64, 8, 0


class Int8MLAKVCacheHandler(KVCacheHandler):
    """omlx hot/SSD handler for Int8MLAKVCache.

    Native block state is the quantized 4-tuple ``(packed, scales, biases,
    values)``; all four elements carry the sequence on axis 2 (quant groups
    run along the FEATURE axis), so per-block slicing/concatenation stay
    exact and ``supports_block_slicing`` is True — this cache must never
    fall into the boundary-snapshot tier (the prior attempt's 256-token
    prefill clamp). Legacy fp16 2-tuple block state deserializes to a plain
    ``KVCache``; the scheduler restore hook is the single requantization
    site for that path.
    """

    @property
    def cache_type(self) -> CacheType:
        return CacheType.INT8_MLA_KV

    @property
    def supports_block_slicing(self) -> bool:
        return True

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        return (
            CacheStateAxisInfo(name="packed", sequence_axis=2, sliceable=True),
            CacheStateAxisInfo(name="scales", sequence_axis=2, sliceable=True),
            CacheStateAxisInfo(name="biases", sequence_axis=2, sliceable=True),
            CacheStateAxisInfo(name="values", sequence_axis=2, sliceable=True),
        )

    def serialize_state(self, cache_obj: Any) -> tuple[Any, ...]:
        native = getattr(cache_obj, "native_kv_state", None)
        if callable(native):
            return tuple(native()[0])
        fp16 = getattr(cache_obj, "fp16_kv_state", None)
        if callable(fp16):
            return tuple(fp16())
        return super().serialize_state(cache_obj)

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        fp16 = getattr(cache_obj, "fp16_kv_state", None)
        if callable(fp16):
            keys, values = fp16()
        else:
            keys, values = super().serialize_state(cache_obj)[:2]
        return {
            "keys": keys,
            "values": values,
            "offset": getattr(
                cache_obj, "offset", keys.shape[2] if keys is not None else 0
            ),
            "cache_type": self.cache_type.value,
        }

    def deserialize_state(
        self,
        elements: tuple[Any, ...],
        meta_state: Any | None = None,
    ) -> Any:
        if len(elements) == 4:
            # Native int8 block state — construct directly, zero requant.
            packed, scales, biases, values = elements
            if packed is None or values is None:
                return None
            group_size, bits, start = _parse_int8_meta(meta_state)
            cache = Int8MLAKVCache(group_size=group_size, bits=bits, start=start)
            cache.state = (packed, scales, biases, values)
            return cache
        if len(elements) == 2:
            # Legacy fp16 block state — plain KVCache; an int8 session
            # re-quantizes via the scheduler restore hook (streamed).
            from mlx_lm.models.cache import KVCache

            keys, values = elements
            if keys is None or values is None:
                return None
            kv = KVCache()
            kv.keys = keys
            kv.values = values
            kv.offset = int(keys.shape[2])
            return kv
        return None

    def concatenate_states(
        self,
        states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not states:
            return {}
        if all(len(s.get("states", ())) == 4 for s in states):
            cat = tuple(
                mx.concatenate([s["states"][k] for s in states], axis=2)
                for k in range(4)
            )
            return {"states": cat, "cache_type": self.cache_type.value}
        # Legacy 2-element states: elements are (keys, values) regardless of
        # this handler's native axis names.
        legacy = []
        for s in states:
            elems = s.get("states")
            if isinstance(elems, (list, tuple)) and len(elems) >= 2:
                legacy.append({"keys": elems[0], "values": elems[1]})
            else:
                legacy.append({"keys": s.get("keys"), "values": s.get("values")})
        return super().concatenate_states(legacy)

    def reconstruct_cache(
        self,
        state: dict[str, Any],
        meta_state: tuple | None = None,
    ) -> Any:
        elements = state.get("states")
        if isinstance(elements, (list, tuple)) and len(elements) == 4:
            return self.deserialize_state(tuple(elements), meta_state)
        keys = state.get("keys")
        values = state.get("values")
        if keys is None or values is None:
            return None
        return self.deserialize_state((keys, values), meta_state)

    def _get_meta_state_keys(self) -> tuple[str, ...]:
        return ("offset", "group_size", "bits", "start", "quantized")
