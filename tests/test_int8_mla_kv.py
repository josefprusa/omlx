# SPDX-License-Identifier: Apache-2.0
"""Tests for the int8 MLA-KV cache (Int8MLAKVCache + hot/cold SSD doctrine).

Covers: quantization exactness, start-threshold semantics, seq-axis slicing
bit-exactness of the quantized triple, trim (MTP chained-draft pattern),
batched merge/extract/filter/extend, NATIVE int8 hot/SSD block persistence
(read-and-go restore with zero requantization, bit-exact vs stored),
cross-mode restore in both directions (int8-stored -> fp16 session via
dequant-on-load, legacy fp16-stored -> int8 session via the scheduler
requant hook), mixed-era block chains, GLM make_cache wiring, and
OFF = perfect no-op.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import CacheList, KVCache  # noqa: E402

from omlx.cache.type_handlers import CacheType, CacheListHandler  # noqa: E402
from omlx.cache.type_registry import CacheTypeRegistry  # noqa: E402
from omlx.patches.glm_moe_dsa.int8_mla_kv import (  # noqa: E402
    BatchInt8MLAKVCache,
    Int8MLAKVCache,
    Int8MLAKVCacheHandler,
)

LR = 512  # latent width (kv_lora_rank in GLM-5.2)
RD = 64  # rope key width
GS = 64


def _quant_roundtrip(x, bits=8, group_size=GS):
    return mx.dequantize(
        *mx.quantize(x, group_size=group_size, bits=bits),
        group_size=group_size,
        bits=bits,
    )


def _rand(n, dim=LR, seed=0, B=1):
    mx.random.seed(seed)
    return mx.random.normal((B, 1, n, dim), dtype=mx.bfloat16)


def _dequant_cache(c):
    triple = c.keys if isinstance(c.keys, (list, tuple)) else None
    assert triple is not None, "cache not quantized"
    return mx.dequantize(
        *(x[..., : c.offset, :] for x in triple),
        group_size=c.group_size,
        bits=c.bits,
    )


def _max_abs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


# ---------------------------------------------------------------------------
# single-cache core
# ---------------------------------------------------------------------------


def test_quantized_fetch_returns_triple_and_matches_roundtrip():
    x, kpe = _rand(40, seed=0), _rand(40, RD, seed=1)
    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    lat, kp = c.update_and_fetch(x, kpe)
    assert isinstance(lat, tuple) and len(lat) == 3
    assert lat[0].dtype == mx.uint32
    deq = mx.dequantize(*lat, group_size=GS, bits=8)
    mx.eval(deq, kp)
    assert c.quantized and c.offset == 40
    assert _max_abs(deq, _quant_roundtrip(x)) == 0.0
    assert _max_abs(kp, kpe) == 0.0  # k_pe stays dense and exact


def test_start_threshold_dense_below_then_converts():
    c = Int8MLAKVCache(group_size=GS, bits=8, start=32)
    ref = KVCache()
    x1, p1 = _rand(20, seed=2), _rand(20, RD, seed=3)
    lat, kp = c.update_and_fetch(x1, p1)
    rlat, rkp = ref.update_and_fetch(x1, p1)
    # Below start: dense mode, bit-identical to plain KVCache.
    assert not isinstance(lat, tuple) and not c.quantized
    mx.eval(lat, rlat, kp, rkp)
    assert _max_abs(lat, rlat) == 0.0 and _max_abs(kp, rkp) == 0.0

    x2, p2 = _rand(20, seed=4), _rand(20, RD, seed=5)
    lat, kp = c.update_and_fetch(x2, p2)
    # 20 + 20 >= 32: whole history quantized once (fork semantics).
    assert isinstance(lat, tuple) and c.quantized and c.offset == 40
    deq = mx.dequantize(*lat, group_size=GS, bits=8)
    full = mx.concatenate([x1, x2], axis=2)
    mx.eval(deq)
    assert _max_abs(deq, _quant_roundtrip(full)) == 0.0
    assert _max_abs(kp, mx.concatenate([p1, p2], axis=2)) == 0.0


def test_below_threshold_stays_dense_forever():
    c = Int8MLAKVCache(group_size=GS, bits=8, start=10**9)
    for seed in (6, 7, 8):
        lat, _ = c.update_and_fetch(_rand(30, seed=seed), _rand(30, RD, seed=seed + 50))
    assert not c.quantized and not isinstance(lat, tuple)
    assert c.offset == 90


def test_growth_across_step_boundary_matches_reference():
    chunks = [300, 1, 1, 1]
    xs = [_rand(n, seed=10 + i) for i, n in enumerate(chunks)]
    ps = [_rand(n, RD, seed=20 + i) for i, n in enumerate(chunks)]
    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    for x, p in zip(xs, ps):
        lat, kp = c.update_and_fetch(x, p)
    deq = mx.dequantize(*lat, group_size=GS, bits=8)
    mx.eval(deq, kp)
    total = sum(chunks)
    assert c.offset == total and deq.shape[2] == total
    ref = mx.concatenate([_quant_roundtrip(x) for x in xs], axis=2)
    assert _max_abs(deq, ref) == 0.0
    assert _max_abs(kp, mx.concatenate(ps, axis=2)) == 0.0


def test_seq_axis_block_slice_of_quantized_triple_is_bit_exact():
    """Quant groups run along the FEATURE axis, so 64-token (block-size)
    slices along the sequence axis dequantize bit-identically to slicing
    the dequantized full tensor. This is the property that makes
    supports_block_slicing=True legal."""
    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    c.update_and_fetch(_rand(256, seed=30), _rand(256, RD, seed=31))
    packed, scales, biases, kpe = c.state
    full = mx.dequantize(packed, scales, biases, group_size=GS, bits=8)
    for start, end in ((0, 64), (64, 128), (128, 192), (192, 256), (32, 96)):
        sl = mx.dequantize(
            packed[..., start:end, :],
            scales[..., start:end, :],
            biases[..., start:end, :],
            group_size=GS,
            bits=8,
        )
        mx.eval(sl)
        assert bool(mx.array_equal(sl, full[..., start:end, :]))
        assert bool(mx.array_equal(kpe[..., start:end, :], kpe[..., start:end, :]))


def test_trim_mtp_chained_draft_pattern_is_exact():
    """MTP verify cycles append K draft rows then trim the rejected ones
    (_trim_mtp_spec). Trim on the quantized cache must be metadata-exact:
    A = base + [t1,t2,t3], trim(2), + [t4]  ==  B = base + [t1] + [t4]."""
    base_x, base_p = _rand(40, seed=40), _rand(40, RD, seed=41)
    ts = [(_rand(1, seed=50 + i), _rand(1, RD, seed=60 + i)) for i in range(4)]

    a = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    a.update_and_fetch(base_x, base_p)
    a.update_and_fetch(
        mx.concatenate([ts[0][0], ts[1][0], ts[2][0]], axis=2),
        mx.concatenate([ts[0][1], ts[1][1], ts[2][1]], axis=2),
    )
    assert a.trim(2) == 2 and a.offset == 41
    lat_a, kp_a = a.update_and_fetch(*ts[3])

    b = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    b.update_and_fetch(base_x, base_p)
    b.update_and_fetch(*ts[0])
    lat_b, kp_b = b.update_and_fetch(*ts[3])

    da = mx.dequantize(*lat_a, group_size=GS, bits=8)
    db = mx.dequantize(*lat_b, group_size=GS, bits=8)
    mx.eval(da, db, kp_a, kp_b)
    assert a.offset == b.offset == 42
    assert _max_abs(da, db) == 0.0
    assert _max_abs(kp_a, kp_b) == 0.0


def test_trim_dense_mode():
    c = Int8MLAKVCache(group_size=GS, bits=8, start=10**6)
    c.update_and_fetch(_rand(10, seed=70), _rand(10, RD, seed=71))
    assert c.is_trimmable() and c.trim(3) == 3 and c.offset == 7


def test_state_raw_never_dequantizes():
    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    c.update_and_fetch(_rand(16, seed=72), _rand(16, RD, seed=73))
    st = c.state
    assert len(st) == 4 and st[0].dtype == mx.uint32  # raw quantized storage
    # meta round-trip
    c2 = Int8MLAKVCache()
    c2.state = st
    c2.meta_state = c.meta_state
    assert c2.quantized and c2.offset == 16 and c2.bits == 8 and c2.group_size == GS
    assert _max_abs(_dequant_cache(c2), _dequant_cache(c)) == 0.0


def test_fp16_kv_state_doctrine_export():
    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    x, p = _rand(24, seed=74), _rand(24, RD, seed=75)
    c.update_and_fetch(x, p)
    k, v = c.fp16_kv_state()
    assert k.dtype == mx.bfloat16 and k.shape == (1, 1, 24, LR)
    mx.eval(k, v)
    assert _max_abs(k, _quant_roundtrip(x)) == 0.0
    assert _max_abs(v, p) == 0.0
    assert c.fp16_kv_class_name == "KVCache"
    # dense mode exports the exact fp16 history
    d = Int8MLAKVCache(group_size=GS, bits=8, start=10**6)
    d.update_and_fetch(x, p)
    k2, v2 = d.fp16_kv_state()
    assert _max_abs(k2, x) == 0.0 and _max_abs(v2, p) == 0.0


def test_from_kv_requantizes_past_threshold_only():
    kv = KVCache()
    x, p = _rand(48, seed=76), _rand(48, RD, seed=77)
    kv.update_and_fetch(x, p)
    past = Int8MLAKVCache.from_kv(kv, group_size=GS, bits=8, start=32)
    assert past.quantized and past.offset == 48
    assert _max_abs(_dequant_cache(past), _quant_roundtrip(x)) == 0.0

    kv2 = KVCache()
    kv2.update_and_fetch(x[..., :16, :], p[..., :16, :])
    below = Int8MLAKVCache.from_kv(kv2, group_size=GS, bits=8, start=32)
    assert not below.quantized and below.offset == 16
    k, _ = below.fp16_kv_state()
    assert _max_abs(k, x[..., :16, :]) == 0.0


def test_latent_dim_must_be_groupable():
    with pytest.raises(ValueError):
        Int8MLAKVCache(group_size=GS, bits=8, start=0, latent_dim=80)
    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    with pytest.raises(ValueError):
        c.update_and_fetch(_rand(4, dim=80, seed=78), _rand(4, RD, seed=79))


def test_memory_materially_smaller_than_fp16():
    n = 512
    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    c.update_and_fetch(_rand(n, seed=80), _rand(n, RD, seed=81))
    fp16_latent = n * LR * 2
    int8_latent = sum(k.nbytes for k in c.keys)
    assert int8_latent < fp16_latent * 0.7


def test_make_mask_does_not_raise():
    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    c.update_and_fetch(_rand(8, seed=82), _rand(8, RD, seed=83))
    m = c.make_mask(2, return_array=True, window_size=None)
    assert m is None or hasattr(m, "shape")


# ---------------------------------------------------------------------------
# batched variant
# ---------------------------------------------------------------------------


def _single(n, seed, start=0):
    x = _rand(n, seed=seed)
    p = _rand(n, RD, seed=seed + 500)
    c = Int8MLAKVCache(group_size=GS, bits=8, start=start)
    lat, kp = c.update_and_fetch(x, p)
    return c, x, p


def test_batch_merge_and_extract_roundtrip():
    c0, x0, p0 = _single(30, 100)
    c1, x1, p1 = _single(50, 101)
    b = Int8MLAKVCache.merge([c0, c1])
    assert isinstance(b, BatchInt8MLAKVCache) and b.quantized
    assert b.size() == 50
    for i, (x, p, off) in enumerate([(x0, p0, 30), (x1, p1, 50)]):
        e = b.extract(i)
        assert isinstance(e, Int8MLAKVCache) and e.offset == off and e.quantized
        assert _max_abs(_dequant_cache(e), _quant_roundtrip(x)) == 0.0
        assert _max_abs(e.values[..., :off, :], p) == 0.0


def test_batch_merge_mixed_modes_quantizes_dense_rows():
    dense, xd, pd = _single(20, 102, start=10**6)  # below threshold: dense
    quant, xq, pq = _single(50, 103, start=0)
    assert not dense.quantized and quant.quantized
    b = Int8MLAKVCache.merge([dense, quant])
    assert b.quantized
    e0 = b.extract(0)
    assert _max_abs(_dequant_cache(e0), _quant_roundtrip(xd)) == 0.0
    e1 = b.extract(1)
    assert _max_abs(_dequant_cache(e1), _quant_roundtrip(xq)) == 0.0


def test_batch_merge_all_dense_stays_dense_then_converts_on_threshold():
    c0, x0, p0 = _single(20, 104, start=30)
    c1, x1, p1 = _single(24, 105, start=30)
    assert not c0.quantized and not c1.quantized
    b = Int8MLAKVCache.merge([c0, c1])
    assert not b.quantized and b.size() == 24
    # decode steps push _idx across the start threshold -> one-shot convert
    for i in range(6):
        lat, _ = b.update_and_fetch(
            _rand(1, seed=200 + i, B=2), _rand(1, RD, seed=300 + i, B=2)
        )
    assert b.quantized and isinstance(lat, tuple)
    assert b.size() == 30


def test_batch_update_after_merge():
    c0, x0, p0 = _single(40, 106)
    b = Int8MLAKVCache.merge([c0])
    nx, npe = _rand(1, seed=107), _rand(1, RD, seed=108)
    lat, kp = b.update_and_fetch(nx, npe)
    assert b.size() == 41 and isinstance(lat, tuple)
    deq = mx.dequantize(*lat, group_size=GS, bits=8)
    mx.eval(deq, kp)
    assert _max_abs(deq[..., 40:, :], _quant_roundtrip(nx)) == 0.0
    assert _max_abs(kp[..., 40:, :], npe) == 0.0


def test_batch_filter():
    c0, x0, p0 = _single(30, 109)
    c1, _, _ = _single(50, 110)
    b = Int8MLAKVCache.merge([c0, c1])
    b.filter(mx.array([0]))
    assert b.size() == 30
    e = b.extract(0)
    assert e.offset == 30
    assert _max_abs(_dequant_cache(e), _quant_roundtrip(x0)) == 0.0


def test_batch_extend_quantized_and_dense():
    c0, x0, _ = _single(30, 111)
    c1, x1, _ = _single(20, 112, start=10**6)  # dense
    b0 = Int8MLAKVCache.merge([c0])
    b1 = Int8MLAKVCache.merge([c1])
    assert b0.quantized and not b1.quantized
    b0.extend(b1)  # mode-aligns: b1 quantizes
    assert b0.quantized and b0.offset.shape[0] == 2
    e0, e1 = b0.extract(0), b0.extract(1)
    assert e0.offset == 30 and e1.offset == 20
    assert _max_abs(_dequant_cache(e0), _quant_roundtrip(x0)) == 0.0
    assert _max_abs(_dequant_cache(e1), _quant_roundtrip(x1)) == 0.0


def test_batch_trim_mtp_pattern():
    c0, _, _ = _single(40, 113)
    b = Int8MLAKVCache.merge([c0])
    drafts_x = mx.concatenate([_rand(1, seed=400 + i) for i in range(3)], axis=2)
    drafts_p = mx.concatenate([_rand(1, RD, seed=410 + i) for i in range(3)], axis=2)
    b.update_and_fetch(drafts_x, drafts_p)
    assert b.trim(2) == 2 and b.size() == 41
    lat, _ = b.update_and_fetch(_rand(1, seed=420), _rand(1, RD, seed=421))
    assert b.size() == 42
    deq = mx.dequantize(*lat, group_size=GS, bits=8)
    mx.eval(deq)
    assert _max_abs(deq[..., 40:41, :], _quant_roundtrip(drafts_x[..., :1, :])) == 0.0
    assert _max_abs(deq[..., 41:42, :], _quant_roundtrip(_rand(1, seed=420))) == 0.0


# ---------------------------------------------------------------------------
# handler / registry / doctrine
# ---------------------------------------------------------------------------


def test_registry_recognizes_cache_and_slicing_enabled():
    CacheTypeRegistry.register(Int8MLAKVCacheHandler())
    h = CacheTypeRegistry.get_handler_by_class_name("Int8MLAKVCache")
    assert isinstance(h, Int8MLAKVCacheHandler)
    assert h.cache_type == CacheType.INT8_MLA_KV
    # Non-negotiable: True, or prefill falls into the boundary-snapshot
    # tier and clamps to block_size chunks (the prior attempt's clamp #3).
    assert h.supports_block_slicing is True
    axis_info = h.get_state_axis_info()
    assert all(i.sliceable and i.sequence_axis == 2 for i in axis_info)
    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    c.update_and_fetch(_rand(4, seed=120), _rand(4, RD, seed=121))
    assert CacheTypeRegistry.detect_cache_type(c) == CacheType.INT8_MLA_KV


def test_scheduler_sliceable_allowlist_covers_int8_cache():
    from omlx.scheduler import _prompt_cache_needs_snapshots

    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    c.update_and_fetch(_rand(4, seed=122), _rand(4, RD, seed=123))
    glm_like = [CacheList(c, KVCache()), CacheList(Int8MLAKVCache())]
    assert _prompt_cache_needs_snapshots(glm_like) is False


def test_cachelist_handler_exports_native_int8_blocks():
    """Quantized latents persist NATIVELY: class name Int8MLAKVCache, raw
    (packed, scales, biases, k_pe) 4-tuple, bit-exact vs live storage."""
    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    x, p = _rand(24, seed=124), _rand(24, RD, seed=125)
    c.update_and_fetch(x, p)
    layer = CacheList(c, KVCache())
    layer.caches[1].update_and_fetch(
        mx.zeros((1, 1, 24, 16), dtype=mx.bfloat16),
        mx.zeros((1, 1, 24, 0), dtype=mx.bfloat16),
    )

    state = CacheListHandler().extract_state(layer)
    assert state["sub_class_names"] == ["Int8MLAKVCache", "KVCache"]
    assert tuple(state["sub_meta_states"][0]) == c.meta_state
    lat_state = state["sub_states"][0]
    assert len(lat_state) == 4 and lat_state[0].dtype == mx.uint32
    mx.eval(*lat_state)
    ref = mx.quantize(x, group_size=GS, bits=8)
    for got, want in zip(lat_state[:3], ref):
        assert mx.array_equal(got, want).item()
    assert _max_abs(lat_state[3], p) == 0.0  # k_pe stays dense fp16


def test_cachelist_handler_exports_dense_below_start_as_legacy_kvcache():
    """Below the start threshold the export stays the legacy fp16 KVCache
    2-tuple — byte-identical to blocks from an fp16 session."""
    c = Int8MLAKVCache(group_size=GS, bits=8, start=1000)
    x, p = _rand(24, seed=224), _rand(24, RD, seed=225)
    c.update_and_fetch(x, p)
    state = CacheListHandler().extract_state(CacheList(c))
    assert state["sub_class_names"] == ["KVCache"]
    assert state["sub_meta_states"][0] == ""
    lat_state = state["sub_states"][0]
    assert len(lat_state) == 2 and lat_state[0].dtype == mx.bfloat16
    assert _max_abs(lat_state[0], x) == 0.0


def test_store_int8_session_restore_fp16_session():
    """Native int8 blocks restored into an int8-OFF session dequantize to a
    plain KVCache on load (cross-mode direction 1, dequant-on-restore)."""
    from omlx.scheduler import Scheduler

    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    x, p = _rand(24, seed=126), _rand(24, RD, seed=127)
    c.update_and_fetch(x, p)
    layer = CacheList(c)

    handler = CacheListHandler()
    state = handler.extract_state(layer)
    rebuilt = handler.reconstruct_cache(
        {"sub_states": state["sub_states"]},
        (state["sub_class_names"], state["sub_meta_states"]),
    )
    assert rebuilt is not None
    sub = rebuilt.caches[0]
    # Restore itself is native (zero requant)...
    assert isinstance(sub, Int8MLAKVCache) and sub.quantized
    assert sub.offset == 24

    # ...and the disarmed scheduler hook converts it for the fp16 session.
    restored = [rebuilt]
    stub_off = SimpleNamespace(_int8_mla_kv_bits=None, _int8_mla_kv_start=0)
    Scheduler._apply_int8_mla_kv_restore(stub_off, restored)
    kv = restored[0].caches[0]
    assert type(kv).__name__ == "KVCache"
    mx.eval(kv.keys, kv.values)
    assert kv.offset == 24
    assert _max_abs(kv.keys, _quant_roundtrip(x)) == 0.0
    assert _max_abs(kv.values, p) == 0.0


def test_store_fp16_session_restore_int8_session_via_scheduler_hook():
    """Direction 2: fp16 blocks restore as plain KVCache; the scheduler hook
    re-quantizes the latent slot past the start threshold."""
    from omlx.scheduler import Scheduler

    kv = KVCache()
    x, p = _rand(72, seed=128), _rand(72, RD, seed=129)
    kv.update_and_fetch(x, p)
    idx = KVCache()  # indexer sub-cache must stay untouched
    restored = [CacheList(kv, idx)]

    stub = SimpleNamespace(_int8_mla_kv_bits=8, _int8_mla_kv_start=32)
    Scheduler._apply_int8_mla_kv_restore(stub, restored)

    sub = restored[0].caches[0]
    assert isinstance(sub, Int8MLAKVCache) and sub.quantized
    assert sub.offset == 72 and sub.start == 32
    assert _max_abs(_dequant_cache(sub), _quant_roundtrip(x)) == 0.0
    assert restored[0].caches[1] is idx  # slot 1 untouched

    # Below the threshold the class still swaps (batch-merge uniformity)
    # but storage stays dense fp16.
    kv2 = KVCache()
    kv2.update_and_fetch(x[..., :16, :], p[..., :16, :])
    restored2 = [CacheList(kv2)]
    Scheduler._apply_int8_mla_kv_restore(stub, restored2)
    sub2 = restored2[0].caches[0]
    assert isinstance(sub2, Int8MLAKVCache) and not sub2.quantized
    k2, _ = sub2.fp16_kv_state()
    assert _max_abs(k2, x[..., :16, :]) == 0.0

    # Disarmed scheduler: perfect no-op.
    stub_off = SimpleNamespace(_int8_mla_kv_bits=None, _int8_mla_kv_start=0)
    restored3 = [CacheList(KVCache())]
    before = restored3[0].caches[0]
    Scheduler._apply_int8_mla_kv_restore(stub_off, restored3)
    assert restored3[0].caches[0] is before


def _extract_states_via_scheduler(cache):
    from unittest.mock import MagicMock

    from omlx.scheduler import Scheduler

    scheduler = MagicMock(spec=Scheduler)
    scheduler.model_name = "glm-int8-test"
    scheduler._normalize_rotating_snapshot_state = (
        Scheduler._normalize_rotating_snapshot_state.__get__(scheduler, Scheduler)
    )
    scheduler._extract_cache_states = Scheduler._extract_cache_states.__get__(
        scheduler, Scheduler
    )
    return scheduler._extract_cache_states(cache)


class _MockModel:
    def __init__(self, num_layers=1):
        self.layers = [SimpleNamespace() for _ in range(num_layers)]


def _make_prefix_cache_with_ssd(ssd_manager):
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(
        block_size=4,
        max_blocks=16,
        model_name="glm-int8-test",
        initial_blocks=16,
    )
    return BlockAwarePrefixCache(
        model=_MockModel(num_layers=1),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd_manager,
    )


def _block_table_for(prefix_cache, hashes, tokens_per_block=4):
    from omlx.cache.paged_cache import BlockTable

    block_ids = []
    for h in hashes:
        block = prefix_cache.paged_cache.allocate_block()
        block.block_hash = h
        block.token_count = tokens_per_block
        block_ids.append(block.block_id)
    return BlockTable(
        request_id="req-int8",
        block_ids=block_ids,
        num_tokens=tokens_per_block * len(hashes),
    )


def test_hot_and_cold_ssd_block_round_trip_native_int8(tmp_path):
    """Full PagedSSDCacheManager round trip: blocks written from an int8
    session persist the NATIVE quantized triple (uint32 packed on disk),
    and load back bit-exact — hot buffer and cold .safetensors tiers."""
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    x, p = _rand(8, seed=130), _rand(8, RD, seed=131)
    c.update_and_fetch(x, p)
    cache = [CacheList(c)]
    mx.eval([lc.state for lc in cache])
    ref = mx.quantize(x, group_size=GS, bits=8)

    extracted, model_cache_config = _extract_states_via_scheduler(cache)
    assert model_cache_config is not None
    # The extracted sub-state is the native int8 export.
    assert extracted[0]["meta_state"][0] == ["Int8MLAKVCache"]

    prefix_cache = _make_prefix_cache_with_ssd(None)
    block_data = prefix_cache._extract_block_tensor_slice(
        extracted,
        0,
        4,
        model_cache_config=model_cache_config,
        is_last_block=False,
    )
    assert block_data is not None
    assert block_data[0][0] == "__cache_list__"
    sub = block_data[0][1][0]  # latent sub-state: __nstate__ native marker
    assert isinstance(sub, tuple) and sub[0] == "__nstate__"
    assert sub[1] == "Int8MLAKVCache"
    elems = sub[2]
    assert len(elems) == 4 and elems[0].dtype == mx.uint32
    mx.eval(*elems)
    for got, want in zip(elems[:3], ref):
        assert mx.array_equal(got, want[..., :4, :]).item()
    assert _max_abs(elems[3], p[..., :4, :]) == 0.0

    block_hash = b"glm_int8_mla_kv_block"
    manager = PagedSSDCacheManager(
        cache_dir=tmp_path / "int8_cache",
        max_size_bytes=64 * 1024**2,
        hot_cache_max_bytes=16 * 1024**2,
    )
    try:
        assert manager.save_block(
            block_hash,
            block_data,
            token_count=4,
            model_name="glm-int8-test",
            layer_cache_types=model_cache_config.get_type_names(),
            layer_meta_states=model_cache_config.get_meta_states(cache),
        )
        # Hot-tier read-back: native marker, bit-exact.
        loaded = manager.load_block(block_hash)
        assert loaded is not None
        lsub = loaded[0][0]
        assert lsub[0] == "__nstate__" and lsub[1] == "Int8MLAKVCache"
        lelems = lsub[2]
        assert lelems[0].dtype == mx.uint32
        mx.eval(*lelems)
        for got, want in zip(lelems[:3], ref):
            assert mx.array_equal(got, want[..., :4, :]).item()
    finally:
        manager.close()

    # Cold tier: the .safetensors on disk carries the packed uint32 payload
    # and the format flag (sub-cache class name) in its metadata.
    files = list((tmp_path / "int8_cache").glob("*/*.safetensors"))
    assert len(files) == 1
    arrays, metadata = mx.load(str(files[0]), return_metadata=True)
    assert metadata["layer_0_sub_0_state_class_name"] == "Int8MLAKVCache"
    assert metadata["layer_0_sub_0_state_count"] == "4"
    assert arrays["layer_0_sub_0_state_0"].dtype == mx.uint32
    assert mx.array_equal(arrays["layer_0_sub_0_state_0"], ref[0][..., :4, :]).item()

    # Fresh manager = server restart: scan indexes the block, cold load works.
    manager2 = PagedSSDCacheManager(
        cache_dir=tmp_path / "int8_cache",
        max_size_bytes=64 * 1024**2,
        hot_cache_max_bytes=16 * 1024**2,
    )
    try:
        loaded2 = manager2.load_block(block_hash)
        assert loaded2 is not None
        assert loaded2[0][0][1] == "Int8MLAKVCache"
        mx.eval(*loaded2[0][0][2])
        assert mx.array_equal(loaded2[0][0][2][0], ref[0][..., :4, :]).item()
    finally:
        manager2.close()


def test_multi_block_native_restore_end_to_end_no_requant(tmp_path):
    """Store an int8 session's cache at a NON-block-aligned length (10 tokens,
    block_size 4 -> two full blocks persist), restore via the real
    BlockAwarePrefixCache.reconstruct_cache, and prove the fast path:
    restored packed bits identical to stored (zero quantize round trip) and
    the armed scheduler hook leaves the object untouched (fast return)."""
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.scheduler import Scheduler

    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    x, p = _rand(10, seed=140), _rand(10, RD, seed=141)
    c.update_and_fetch(x, p)
    cache = [CacheList(c)]
    mx.eval([lc.state for lc in cache])
    stored_triple = [t[..., :10, :] for t in c.keys]

    extracted, model_cache_config = _extract_states_via_scheduler(cache)
    manager = PagedSSDCacheManager(
        cache_dir=tmp_path / "int8_cache",
        max_size_bytes=64 * 1024**2,
        hot_cache_max_bytes=16 * 1024**2,
    )
    try:
        hashes = [b"int8_native_block_0", b"int8_native_block_1"]
        for i, h in enumerate(hashes):
            prefix_cache_tmp = _make_prefix_cache_with_ssd(manager)
            block_data = prefix_cache_tmp._extract_block_tensor_slice(
                extracted,
                i * 4,
                (i + 1) * 4,
                model_cache_config=model_cache_config,
                is_last_block=(i == 1),
            )
            assert block_data is not None
            assert manager.save_block(
                h,
                block_data,
                token_count=4,
                model_name="glm-int8-test",
                layer_cache_types=model_cache_config.get_type_names(),
                layer_meta_states=model_cache_config.get_meta_states(cache),
            )

        # Restore both blocks (8 of the 10 tokens — the partial block is
        # never persisted).
        prefix_cache = _make_prefix_cache_with_ssd(manager)
        table = _block_table_for(prefix_cache, hashes)
        restored = prefix_cache.reconstruct_cache(table)
        assert restored is not None and len(restored) == 1
        sub = restored[0].caches[0]
        assert isinstance(sub, Int8MLAKVCache) and sub.quantized
        assert sub.offset == 8
        assert sub.group_size == GS and sub.bits == 8
        mx.eval(sub.state)
        # Bit-exact vs STORED triple — proves no quantize round trip.
        for got, want in zip(sub.keys, stored_triple):
            assert mx.array_equal(got, want[..., :8, :]).item()
        assert _max_abs(sub.values, p[..., :8, :]) == 0.0

        # Armed scheduler hook: fast return, object identity preserved.
        stub = SimpleNamespace(_int8_mla_kv_bits=8, _int8_mla_kv_start=0)
        Scheduler._apply_int8_mla_kv_restore(stub, restored)
        assert restored[0].caches[0] is sub
        for got, want in zip(sub.keys, stored_triple):
            assert mx.array_equal(got, want[..., :8, :]).item()

        # Walk-back/trim shape: restoring a shorter prefix (block 0 only)
        # yields a consistent 4-token native cache.
        prefix_cache2 = _make_prefix_cache_with_ssd(manager)
        table2 = _block_table_for(prefix_cache2, hashes[:1])
        restored2 = prefix_cache2.reconstruct_cache(table2)
        assert restored2 is not None
        sub2 = restored2[0].caches[0]
        assert isinstance(sub2, Int8MLAKVCache) and sub2.offset == 4
        mx.eval(sub2.state)
        for got, want in zip(sub2.keys, stored_triple):
            assert mx.array_equal(got, want[..., :4, :]).item()
    finally:
        manager.close()


def test_mixed_era_chain_normalizes_to_fp16(tmp_path):
    """A prefix chain mixing legacy fp16 blocks (pre-upgrade) and native int8
    blocks restores as a plain dense KVCache (native blocks dequantize), so
    old on-disk data never breaks a restore."""
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

    c = Int8MLAKVCache(group_size=GS, bits=8, start=0)
    x, p = _rand(8, seed=150), _rand(8, RD, seed=151)
    c.update_and_fetch(x, p)
    cache = [CacheList(c)]
    mx.eval([lc.state for lc in cache])
    x_rt = _quant_roundtrip(x)

    extracted, model_cache_config = _extract_states_via_scheduler(cache)

    manager = PagedSSDCacheManager(
        cache_dir=tmp_path / "int8_cache",
        max_size_bytes=64 * 1024**2,
        hot_cache_max_bytes=16 * 1024**2,
    )
    try:
        # Block 0: legacy-era fp16 payload (hand-built the way the old
        # fp16_kv_state doctrine persisted it).
        legacy_block = [
            (
                "__cache_list__",
                [(x_rt[..., :4, :], p[..., :4, :])],
            )
        ]
        assert manager.save_block(
            b"int8_mixed_block_0",
            legacy_block,
            token_count=4,
            model_name="glm-int8-test",
            layer_cache_types=["CacheList"],
            layer_meta_states=[(["KVCache"], [""])],
        )

        # Block 1: native int8 payload from the current extract path.
        prefix_cache_tmp = _make_prefix_cache_with_ssd(manager)
        native_block = prefix_cache_tmp._extract_block_tensor_slice(
            extracted,
            4,
            8,
            model_cache_config=model_cache_config,
            is_last_block=True,
        )
        assert native_block is not None
        assert manager.save_block(
            b"int8_mixed_block_1",
            native_block,
            token_count=4,
            model_name="glm-int8-test",
            layer_cache_types=model_cache_config.get_type_names(),
            layer_meta_states=model_cache_config.get_meta_states(cache),
        )

        prefix_cache = _make_prefix_cache_with_ssd(manager)
        table = _block_table_for(
            prefix_cache, [b"int8_mixed_block_0", b"int8_mixed_block_1"]
        )
        restored = prefix_cache.reconstruct_cache(table)
        assert restored is not None and len(restored) == 1
        sub = restored[0].caches[0]
        assert type(sub).__name__ == "KVCache"
        mx.eval(sub.keys, sub.values)
        assert sub.keys.shape[2] == 8
        assert _max_abs(sub.keys, x_rt) == 0.0
        assert _max_abs(sub.values, p) == 0.0
    finally:
        manager.close()


# ---------------------------------------------------------------------------
# GLM model wiring + OFF no-op
# ---------------------------------------------------------------------------


def _load_patched_glm_module():
    from omlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch

    apply_glm_moe_dsa_patch()
    from mlx_lm.models import glm_moe_dsa

    return glm_moe_dsa


def _small_glm_args(glm_moe_dsa, kv_lora_rank=64):
    return glm_moe_dsa.ModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=1024,
        hidden_size=128,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=4,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=24,
        qk_rope_head_dim=16,
        v_head_dim=32,
        qk_nope_head_dim=16,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0},
        attention_bias=False,
        index_topk_pattern="FSFSFS",
    )


def test_make_cache_off_is_plain_kvcache():
    glm_moe_dsa = _load_patched_glm_module()
    model = glm_moe_dsa.Model(_small_glm_args(glm_moe_dsa))
    cache = model.make_cache()
    for layer_cache in cache:
        assert type(layer_cache.caches[0]) is KVCache


def test_make_cache_on_builds_int8_latent_slot_only():
    glm_moe_dsa = _load_patched_glm_module()
    model = glm_moe_dsa.Model(_small_glm_args(glm_moe_dsa))
    model._int8_mla_kv_bits = 8
    model._int8_mla_kv_start = 128
    cache = model.make_cache()
    assert [len(c.caches) for c in cache] == [2, 1, 2, 1, 2, 1]
    for layer_cache in cache:
        latent = layer_cache.caches[0]
        assert isinstance(latent, Int8MLAKVCache)
        assert latent.bits == 8 and latent.group_size == 64 and latent.start == 128
        if len(layer_cache.caches) == 2:
            assert type(layer_cache.caches[1]) is KVCache  # indexer stays fp16


def test_forward_below_start_is_token_identical_to_off():
    """OFF = perfect no-op: with the threshold never crossed, the int8 cache
    is storage-identical to KVCache and logits match bit-for-bit."""
    glm_moe_dsa = _load_patched_glm_module()
    args = _small_glm_args(glm_moe_dsa)
    mx.random.seed(0)
    model = glm_moe_dsa.Model(args)
    prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])

    cache_off = model.make_cache()
    logits_off = model(prompt, cache=cache_off)

    model._int8_mla_kv_bits = 8
    model._int8_mla_kv_start = 10**9
    cache_on = model.make_cache()
    logits_on = model(prompt, cache=cache_on)
    mx.eval(logits_off, logits_on)
    assert bool(mx.array_equal(logits_off, logits_on))

    nxt = mx.argmax(logits_off[0, -1:, :], keepdims=True)
    step_off = model(nxt, cache=cache_off)
    step_on = model(nxt, cache=cache_on)
    mx.eval(step_off, step_on)
    assert bool(mx.array_equal(step_off, step_on))


def test_forward_quantized_dequant_on_read_end_to_end(caplog):
    """start=0: the attention layer sees the quantized triple, dequantizes
    before any kernel branch, and produces finite, close-to-fp16 logits.
    The [INT8KV] engagement counter fires."""
    import logging

    glm_moe_dsa = _load_patched_glm_module()
    args = _small_glm_args(glm_moe_dsa)
    mx.random.seed(0)
    model = glm_moe_dsa.Model(args)
    prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])

    baseline = model(prompt, cache=model.make_cache())

    model._int8_mla_kv_bits = 8
    model._int8_mla_kv_start = 0
    cache = model.make_cache()
    glm_moe_dsa._INT8_LOGGED.clear()
    with caplog.at_level(logging.WARNING, logger="omlx.glm_dko"):
        logits = model(prompt, cache=cache)
        mx.eval(logits)
        nxt = mx.argmax(logits[0, -1:, :], keepdims=True)
        step = model(nxt, cache=cache)
        mx.eval(step)

    assert bool(mx.all(mx.isfinite(logits)).item())
    assert bool(mx.all(mx.isfinite(step)).item())
    mx.eval(baseline)
    # int8 latent quantization error is small on a fresh random tiny model.
    assert _max_abs(logits, baseline) < 1.0
    assert cache[0].caches[0].quantized and cache[0].caches[0].offset == 9
    assert any("[INT8KV] ENGAGED" in r.message for r in caplog.records)
    # cache state stays cheap/raw and evaluable (per-chunk scheduler eval)
    mx.eval([c.state for c in cache])


# ---------------------------------------------------------------------------
# settings plumbing
# ---------------------------------------------------------------------------


def test_settings_defaults_and_mutual_exclusion():
    from omlx.model_settings import ModelSettings

    s = ModelSettings()
    assert s.int8_mla_kv_enabled is False
    assert s.int8_mla_kv_bits == 8
    assert s.int8_mla_kv_start == 0
    with pytest.raises(ValueError):
        ModelSettings(int8_mla_kv_enabled=True, turboquant_kv_enabled=True)


def test_profile_whitelist_includes_int8_keys():
    from omlx.model_profiles import MODEL_SPECIFIC_PROFILE_FIELDS

    for key in ("int8_mla_kv_enabled", "int8_mla_kv_bits", "int8_mla_kv_start"):
        assert key in MODEL_SPECIFIC_PROFILE_FIELDS


def test_post_load_transform_sets_model_attrs():
    from omlx.utils.model_loading import apply_post_load_transforms

    class _M:
        def make_cache(self):  # pragma: no cover - presence only
            return []

    settings = SimpleNamespace(
        int8_mla_kv_enabled=True,
        int8_mla_kv_bits=8,
        int8_mla_kv_start=4096,
        index_cache_freq=None,
    )
    model = apply_post_load_transforms(_M(), settings)
    assert model._int8_mla_kv_bits == 8
    assert model._int8_mla_kv_start == 4096

    off = apply_post_load_transforms(_M(), SimpleNamespace(index_cache_freq=None))
    assert not hasattr(off, "_int8_mla_kv_bits")
