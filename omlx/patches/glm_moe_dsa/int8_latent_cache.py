# SPDX-License-Identifier: Apache-2.0
"""Int8 group-quantized cache for the GLM-5.2 MLA latent.

GLM ``glm_moe_dsa`` MLA stores, per layer, a CacheList(latent_cache[, indexer_cache]).
The latent_cache holds the low-rank latent (kv_lora_rank, in the *keys* slot) plus the
rope key ``k_pe`` (in the *values* slot). This cache int8-quantizes ONLY the latent and
keeps ``k_pe`` dense, then **dequantizes on read** in ``update_and_fetch`` so every
downstream consumer (native sparse-MLA / exact-block kernels, SDPA, decode top-k gather,
the embed_q/unembed_out projections) sees a dense fp16/bf16 latent unchanged. The win is
KV-cache *memory* (the 512-d latent at int8 is ~2x smaller than fp16); accuracy is bounded
by int8 group-quant of the latent only.

Mirrors mlx-lm QuantizedKVCache (keys) + KVCache (values). Works with omlx hot (RAM,
boundary-snapshot) and SSD (cold) caching via Int8MLALatentCacheHandler, registered by the
GLM patch. supports_block_slicing=False routes it through the fully-supported snapshot tier.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
# NOTE: cache.py's create_attention_mask is the builder that takes ``offset``;
# base.py's same-named function is the entry point that delegates to make_mask
# (importing that one causes infinite delegation / a TypeError on ``offset``).
from mlx_lm.models.cache import (
    _BaseCache,
    create_attention_mask,
    create_causal_mask,
    dynamic_roll,
)

from omlx.cache.type_handlers import CacheStateAxisInfo, CacheType, CacheTypeHandler

logger = logging.getLogger(__name__)


class Int8MLALatentCache(_BaseCache):
    """Quantized-latent + dense-k_pe cache for GLM MLA. Dequant-on-read."""

    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8):
        self.keys = None  # (packed_uint32, scales, biases) — quantized latent
        self.values = None  # k_pe — dense
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys, values):
        # keys = latent [B, 1, S, kv_lora_rank]; values = k_pe [B, 1, S, rope_dim]
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.values.shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                pad = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, pad], axis=-2)

            new_v = mx.zeros((*shape, v_head_dim), dtype=values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = tuple(x[..., :prev, :] for x in self.keys)
                    self.values = self.values[..., :prev, :]
                self.keys = tuple(expand_quant(x) for x in self.keys)
                self.values = mx.concatenate([self.values, new_v], axis=-2)
            else:
                self.keys = init_quant(k_head_dim)
                self.values = new_v

        self.offset += num_steps

        q = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = q[i]
        self.values[..., prev : self.offset, :] = values

        # Dequant-on-read: hand back a dense latent so every consumer is unchanged.
        cur = tuple(x[..., : self.offset, :] for x in self.keys)
        latent = mx.dequantize(*cur, group_size=self.group_size, bits=self.bits)
        return latent, self.values[..., : self.offset, :]

    def size(self):
        return self.offset

    @property
    def state(self):
        if self.keys is None:
            return (None, None, None, None)
        if self.offset == self.values.shape[-2]:
            return (*self.keys, self.values)
        return (
            *(x[..., : self.offset, :] for x in self.keys),
            self.values[..., : self.offset, :],
        )

    @state.setter
    def state(self, v):
        *k, val = v
        if k[0] is None:
            self.keys, self.values = None, None
        else:
            self.keys, self.values = tuple(k), val

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.group_size, self.bits = map(int, v)

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
        return BatchInt8MLALatentCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return sum(x.nbytes for x in self.keys) + self.values.nbytes


class BatchInt8MLALatentCache(_BaseCache):
    """Batched (continuous-batching) variant of Int8MLALatentCache.

    Mirrors mlx-lm BatchKVCache (left-padded, per-row offsets, merge/filter/extend/
    extract) but stores the latent (keys slot) int8 group-quantized and k_pe (values
    slot) dense, dequantizing the latent on read. Padded positions dequantize to 0 and
    are masked by left_padding, so attention is unchanged.
    """

    step = 256

    def __init__(self, left_padding, group_size: int = 64, bits: int = 8):
        self.keys = None  # (packed, scales, biases) — quantized latent
        self.values = None  # k_pe dense
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None
        self.group_size = group_size
        self.bits = bits

    def _grow(self, B, H, k_head_dim, v_head_dim, num_steps, prev, dtype):
        el_per_int = 8 * mx.uint32.size // self.bits
        n_steps = (self.step + num_steps - 1) // self.step
        new_len = n_steps * self.step
        shape = (B, H, new_len)
        new_k = (
            mx.zeros((*shape, k_head_dim // el_per_int), dtype=mx.uint32),
            mx.zeros((*shape, k_head_dim // self.group_size), dtype=dtype),
            mx.zeros((*shape, k_head_dim // self.group_size), dtype=dtype),
        )
        new_v = mx.zeros((*shape, v_head_dim), dtype=dtype)
        if self.keys is not None:
            if prev % self.step != 0:
                self.keys = tuple(x[..., :prev, :] for x in self.keys)
                self.values = self.values[..., :prev, :]
            self.keys = tuple(
                mx.concatenate([o, n], axis=2) for o, n in zip(self.keys, new_k)
            )
            self.values = mx.concatenate([self.values, new_v], axis=2)
        else:
            self.keys, self.values = new_k, new_v

    def update_and_fetch(self, keys, values):
        prev = self._idx
        if self.keys is None or (prev + keys.shape[2]) > self.values.shape[2]:
            B, H, _, k_head_dim = keys.shape
            self._grow(B, H, k_head_dim, values.shape[3], keys.shape[2], prev, keys.dtype)

        self.offset += keys.shape[2]
        self._idx += keys.shape[2]
        q = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self._idx, :] = q[i]
        self.values[..., prev : self._idx, :] = values

        cur = tuple(x[..., : self._idx, :] for x in self.keys)
        latent = mx.dequantize(*cur, group_size=self.group_size, bits=self.bits)
        return latent, self.values[..., : self._idx, :]

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
            self.keys = tuple(dynamic_roll(x, pad[:, None], axis=2) for x in self.keys)
            self.values = dynamic_roll(self.values, pad[:, None], axis=2)
            self.offset -= pad
            self.left_padding += pad
            self._right_padding = None

    @property
    def state(self):
        k = self.keys
        v = self.values
        if k is not None and self._idx < self.values.shape[2]:
            k = tuple(x[..., : self._idx, :] for x in k)
            v = v[..., : self._idx, :]
        if k is None:
            return (None, None, None, v, self.offset, self.left_padding)
        return (*k, v, self.offset, self.left_padding)

    @state.setter
    def state(self, s):
        packed, scales, biases, v, off, lp = s
        self.keys = None if packed is None else (packed, scales, biases)
        self.values = v
        self.offset = off
        self.left_padding = lp
        self._idx = 0 if v is None else v.shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.group_size, self.bits = map(int, v)

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
            self.keys = tuple(x[batch_indices] for x in self.keys)
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = tuple(x[..., min_left_pad:, :] for x in self.keys)
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extract(self, idx):
        cache = Int8MLALatentCache(group_size=self.group_size, bits=self.bits)
        padding = self.left_padding[idx].item()
        cache.keys = tuple(
            mx.contiguous(x[idx : idx + 1, :, padding : self._idx]) for x in self.keys
        )
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        cache.offset = cache.values.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        gs = caches[0].group_size
        bits = caches[0].bits
        if max_length == 0:
            return cls([0] * len(caches), group_size=gs, bits=bits)

        padding = [max_length - l for l in lengths]
        B = len(caches)
        ref = next(c for c in caches if c.keys is not None)
        H = ref.keys[0].shape[1]
        packed_dim = ref.keys[0].shape[3]
        groups = ref.keys[1].shape[3]
        Dv = ref.values.shape[3]
        dt = ref.keys[1].dtype

        packed = mx.zeros((B, H, max_length, packed_dim), dtype=mx.uint32)
        scales = mx.zeros((B, H, max_length, groups), dtype=dt)
        biases = mx.zeros((B, H, max_length, groups), dtype=dt)
        kpe = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            o = c.offset
            packed[i : i + 1, :, p : p + o] = c.keys[0][..., :o, :]
            scales[i : i + 1, :, p : p + o] = c.keys[1][..., :o, :]
            biases[i : i + 1, :, p : p + o] = c.keys[2][..., :o, :]
            kpe[i : i + 1, :, p : p + o] = c.values[..., :o, :]

        cache = cls(padding, group_size=gs, bits=bits)
        cache.keys = (packed, scales, biases)
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
        return sum(x.nbytes for x in self.keys) + self.values.nbytes


class Int8MLALatentCacheHandler(CacheTypeHandler):
    """omlx hot/SSD handler for Int8MLALatentCache (mirrors PoolingCacheHandler).

    The 4-array state (latent packed/scales/biases + dense k_pe) is round-tripped as an
    opaque full-state snapshot; supports_block_slicing=False routes it through the
    boundary-snapshot path used by the SSD cold tier and the RAM hot tier.
    """

    _NAMES = ("lat_packed", "lat_scales", "lat_biases", "k_pe")

    @property
    def cache_type(self) -> CacheType:
        return CacheType.INT8_MLA_LATENT

    @property
    def supports_block_slicing(self) -> bool:
        return False

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        st = cache_obj.state
        out = {name: st[i] for i, name in enumerate(self._NAMES)}
        out["cache_type"] = self.cache_type.value
        return out

    def get_seq_len(self, state: dict[str, Any]) -> int:
        kpe = state.get("k_pe")
        if kpe is not None and hasattr(kpe, "shape") and len(kpe.shape) >= 3:
            return int(kpe.shape[2])
        return 0

    def slice_state(self, state, start_idx, end_idx):
        # Not block-sliceable; opaque full-state snapshot.
        return {**state, "is_full_state": True}

    def concatenate_states(self, states):
        return states[-1] if states else {}

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        return tuple(
            CacheStateAxisInfo(name=n, sequence_axis=2, sliceable=False)
            for n in self._NAMES
        )

    def reconstruct_cache(self, state, meta_state=None):
        cache = Int8MLALatentCache()
        if isinstance(meta_state, (list, tuple)) and len(meta_state) == 3:
            cache.meta_state = meta_state
        packed = state.get("lat_packed")
        cache.state = (
            packed,
            state.get("lat_scales"),
            state.get("lat_biases"),
            state.get("k_pe"),
        )
        return cache

    def deserialize_state(self, elements, meta_state=None):
        if not isinstance(elements, (list, tuple)) or len(elements) != 4:
            logger.error(
                "Int8MLALatentCache deserialize: expected 4 elements, got %s",
                len(elements) if isinstance(elements, (list, tuple)) else type(elements),
            )
            return None
        state = {name: elements[i] for i, name in enumerate(self._NAMES)}
        return self.reconstruct_cache(state, meta_state)

    def _get_state_keys(self) -> tuple[str, ...]:
        return self._NAMES

    def _get_meta_state_keys(self) -> tuple[str, ...]:
        return ("offset", "group_size", "bits")
