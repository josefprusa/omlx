"""Tests for the GLM-5.2 int8 MLA latent cache (hot + SSD caching)."""

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.cache.type_handlers import CacheType  # noqa: E402
from omlx.cache.type_registry import CacheTypeRegistry  # noqa: E402
from omlx.patches.glm_moe_dsa.int8_latent_cache import (  # noqa: E402
    BatchInt8MLALatentCache,
    Int8MLALatentCache,
    Int8MLALatentCacheHandler,
    dequantize_latent,
)
from omlx.patches.glm_moe_dsa.kernels import fast as glm_fast  # noqa: E402

LR = 512  # kv_lora_rank
RD = 64  # qk_rope_head_dim
GS = 64


def _quant_roundtrip(x, bits=8, group_size=GS):
    return mx.dequantize(
        *mx.quantize(x, group_size=group_size, bits=bits),
        group_size=group_size,
        bits=bits,
    )


def test_update_and_fetch_matches_quant_roundtrip():
    mx.random.seed(0)
    x = mx.random.normal((1, 1, 40, LR), dtype=mx.bfloat16)
    kpe = mx.random.normal((1, 1, 40, RD), dtype=mx.bfloat16)
    c = Int8MLALatentCache(bits=8, group_size=GS)
    lat, kp = c.update_and_fetch(x, kpe)
    mx.eval(lat, kp)
    # Latent is exactly quantize->dequantize of the input; k_pe is untouched.
    assert lat.shape == (1, 1, 40, LR) and lat.dtype == mx.bfloat16
    assert float(mx.max(mx.abs(lat - _quant_roundtrip(x))).item()) == 0.0
    assert float(mx.max(mx.abs(kp - kpe)).item()) == 0.0
    assert c.offset == 40


def test_multi_step_growth_matches_reference():
    mx.random.seed(1)
    # prefill chunk then several decode steps spanning the 256 step boundary
    chunks = [300, 1, 1, 1]
    xs = [mx.random.normal((1, 1, n, LR), dtype=mx.bfloat16) for n in chunks]
    ps = [mx.random.normal((1, 1, n, RD), dtype=mx.bfloat16) for n in chunks]
    c = Int8MLALatentCache(bits=8, group_size=GS)
    lat = kp = None
    for x, p in zip(xs, ps):
        lat, kp = c.update_and_fetch(x, p)
    mx.eval(lat, kp)
    total = sum(chunks)
    assert c.offset == total and lat.shape[2] == total and kp.shape[2] == total
    # k_pe is preserved exactly across growth/realloc
    assert float(mx.max(mx.abs(kp - mx.concatenate(ps, axis=2))).item()) == 0.0
    # each token's latent equals its own quant roundtrip (stored once, never re-quantized)
    ref = mx.concatenate([_quant_roundtrip(x) for x in xs], axis=2)
    assert float(mx.max(mx.abs(lat - ref)).item()) == 0.0


def test_int8_is_accurate_enough():
    """int8 group-quant of the latent stays close to fp16 (sanity, not exactness)."""
    mx.random.seed(2)
    x = mx.random.normal((1, 1, 128, LR), dtype=mx.bfloat16)
    err = mx.max(mx.abs(_quant_roundtrip(x).astype(mx.float32) - x.astype(mx.float32)))
    # int8 affine over unit-normal data: max abs error well under 0.1
    assert float(err.item()) < 0.1


def test_state_meta_roundtrip_via_handler():
    mx.random.seed(3)
    x = mx.random.normal((1, 1, 77, LR), dtype=mx.bfloat16)
    kpe = mx.random.normal((1, 1, 77, RD), dtype=mx.bfloat16)
    c = Int8MLALatentCache(bits=8, group_size=GS)
    lat, kp = c.update_and_fetch(x, kpe)
    mx.eval(lat, kp)

    state = c.state  # flat 4-tuple (packed, scales, biases, k_pe)
    meta = c.meta_state
    assert len(state) == 4
    assert state[0].dtype == mx.uint32  # packed latent
    assert meta == ("77", str(GS), "8")

    handler = Int8MLALatentCacheHandler()
    c2 = handler.deserialize_state(state, [int(m) for m in meta])
    assert isinstance(c2, Int8MLALatentCache)
    assert c2.offset == 77 and c2.group_size == GS and c2.bits == 8
    # restored cache dequantizes to the same latent + k_pe
    relat = mx.dequantize(*c2.keys, group_size=GS, bits=8)
    mx.eval(relat)
    assert float(mx.max(mx.abs(relat - lat)).item()) == 0.0
    assert float(mx.max(mx.abs(c2.values - kp)).item()) == 0.0


def test_registry_recognizes_cache():
    CacheTypeRegistry.register(Int8MLALatentCacheHandler())
    h = CacheTypeRegistry.get_handler_by_class_name("Int8MLALatentCache")
    assert isinstance(h, Int8MLALatentCacheHandler)
    assert h.cache_type == CacheType.INT8_MLA_LATENT
    assert h.supports_block_slicing is False
    c = Int8MLALatentCache()
    c.update_and_fetch(
        mx.zeros((1, 1, 4, LR), dtype=mx.bfloat16),
        mx.zeros((1, 1, 4, RD), dtype=mx.bfloat16),
    )
    assert CacheTypeRegistry.detect_cache_type(c) == CacheType.INT8_MLA_LATENT


def test_memory_smaller_than_fp16():
    mx.random.seed(4)
    n = 512
    x = mx.random.normal((1, 1, n, LR), dtype=mx.bfloat16)
    kpe = mx.random.normal((1, 1, n, RD), dtype=mx.bfloat16)
    c = Int8MLALatentCache(bits=8, group_size=GS)
    c.update_and_fetch(x, kpe)
    fp16_latent_bytes = n * LR * 2  # bf16 latent if stored dense
    int8_latent_bytes = sum(k.nbytes for k in c.keys)
    # int8 latent (+scales/biases) is materially smaller than dense bf16 latent
    assert int8_latent_bytes < fp16_latent_bytes * 0.7


def test_trim_and_empty():
    c = Int8MLALatentCache()
    assert c.empty() and c.nbytes == 0
    c.update_and_fetch(
        mx.zeros((1, 1, 10, LR), dtype=mx.bfloat16),
        mx.zeros((1, 1, 10, RD), dtype=mx.bfloat16),
    )
    assert not c.empty() and c.is_trimmable()
    assert c.trim(3) == 3 and c.offset == 7


def test_make_mask_does_not_raise():
    # regression: make_mask must use cache.py create_attention_mask (takes offset)
    c = Int8MLALatentCache()
    c.update_and_fetch(
        mx.random.normal((1, 1, 8, LR), dtype=mx.bfloat16),
        mx.random.normal((1, 1, 8, RD), dtype=mx.bfloat16),
    )
    # base.create_attention_mask always passes window_size when delegating to make_mask
    m = c.make_mask(2, return_array=True, window_size=None)
    assert m is None or hasattr(m, "shape")


def _single(n, seed):
    mx.random.seed(seed)
    x = mx.random.normal((1, 1, n, LR), dtype=mx.bfloat16)
    p = mx.random.normal((1, 1, n, RD), dtype=mx.bfloat16)
    c = Int8MLALatentCache(bits=8, group_size=GS)
    lat, kp = c.update_and_fetch(x, p)
    mx.eval(lat, kp)
    return c, lat, p


def test_batch_merge_and_extract_roundtrip():
    # continuous batching: merge per-row caches of DIFFERENT lengths, extract back
    c0, lat0, p0 = _single(30, 10)
    c1, lat1, p1 = _single(50, 11)
    b = Int8MLALatentCache.merge([c0, c1])
    assert isinstance(b, BatchInt8MLALatentCache)
    assert b.size() == 50  # max length, left-padded
    for i, (lat, p, off) in enumerate([(lat0, p0, 30), (lat1, p1, 50)]):
        e = b.extract(i)
        assert isinstance(e, Int8MLALatentCache) and e.offset == off
        relat = mx.dequantize(*e.keys, group_size=GS, bits=8)
        mx.eval(relat)
        assert float(mx.max(mx.abs(relat - lat)).item()) == 0.0
        assert float(mx.max(mx.abs(e.values - p)).item()) == 0.0


def test_batch_update_after_merge():
    c0, _, _ = _single(40, 12)
    b = Int8MLALatentCache.merge([c0])  # B=1, idx=40
    nx = mx.random.normal((1, 1, 1, LR), dtype=mx.bfloat16)
    npe = mx.random.normal((1, 1, 1, RD), dtype=mx.bfloat16)
    lat, kp = b.update_and_fetch(nx, npe)
    mx.eval(lat, kp)
    assert b.size() == 41 and lat.shape == (1, 1, 41, LR)
    assert float(mx.max(mx.abs(lat[:, :, 40:, :] - _quant_roundtrip(nx))).item()) == 0.0
    assert float(mx.max(mx.abs(kp[:, :, 40:, :] - npe)).item()) == 0.0


def test_return_quantized_single_no_dequant():
    mx.random.seed(20)
    x = mx.random.normal((1, 1, 50, LR), dtype=mx.bfloat16)
    kpe = mx.random.normal((1, 1, 50, RD), dtype=mx.bfloat16)
    c = Int8MLALatentCache(bits=8, group_size=GS)
    quant, kp = c.update_and_fetch(x, kpe, return_quantized=True)
    # returns the (packed, scales, biases) tuple, NOT a dense latent
    assert isinstance(quant, tuple) and len(quant) == 3
    assert quant[0].dtype == mx.uint32 and quant[0].shape == (1, 1, 50, LR // 4)
    mx.eval(*quant, kp)
    # dequantizing it reproduces the dense read path exactly
    deq = dequantize_latent(quant, GS, 8)
    ref = _quant_roundtrip(x)
    assert float(mx.max(mx.abs(deq - ref)).item()) == 0.0
    assert float(mx.max(mx.abs(kp - kpe)).item()) == 0.0


def test_return_quantized_batch_no_dequant():
    c0, _, _ = _single(40, 21)
    b = Int8MLALatentCache.merge([c0])
    nx = mx.random.normal((1, 1, 1, LR), dtype=mx.bfloat16)
    npe = mx.random.normal((1, 1, 1, RD), dtype=mx.bfloat16)
    quant, kp = b.update_and_fetch(nx, npe, return_quantized=True)
    assert isinstance(quant, tuple) and len(quant) == 3
    assert quant[0].dtype == mx.uint32
    deq = dequantize_latent(quant, b.group_size, b.bits)
    mx.eval(deq, kp)
    assert deq.shape == (1, 1, 41, LR)
    assert float(mx.max(mx.abs(deq[:, :, 40:, :] - _quant_roundtrip(nx))).item()) == 0.0


@pytest.mark.skipif(
    not glm_fast.has("glm_dsa_sparse_mla_attention_q8")
    or not glm_fast.has("glm_dsa_sparse_mla_attention"),
    reason="native GLM kernels not built",
)
def test_q8_kernel_matches_dense():
    # int8-native sparse-MLA must equal dequantize + dense sparse-MLA.
    H, TOPK = 64, 2048
    for dtype in (mx.float16, mx.bfloat16):
        for K, L, flags in (
            (4096, 4096, {}),
            (4096, 512, {}),
            (2048, 2048, dict(topk_valid_prefix=True, causal_prefix_indices=True)),
        ):
            mx.random.seed(7)
            ql = (mx.random.normal((1, H, L, LR)) * 0.1).astype(dtype)
            qp = (mx.random.normal((1, H, L, RD)) * 0.1).astype(dtype)
            kv = (mx.random.normal((1, 1, K, LR)) * 0.1).astype(dtype)
            kpe = (mx.random.normal((1, 1, K, RD)) * 0.1).astype(dtype)
            packed, scales, biases = mx.quantize(kv, group_size=GS, bits=8)
            kv_deq = mx.dequantize(packed, scales, biases, group_size=GS, bits=8)
            topk = mx.random.randint(0, K, (1, 1, L, TOPK)).astype(mx.uint32)
            topk = mx.concatenate(
                [mx.zeros((1, 1, L, 1), dtype=mx.uint32), topk[..., 1:]], axis=-1
            )
            scale = 1.0 / (LR**0.5)
            o_q8 = glm_fast.glm_dsa_sparse_mla_attention_q8(
                ql, qp, packed, scales, biases, kpe, topk, scale,
                group_size=GS, bits=8, **flags,
            )
            o_dn = glm_fast.glm_dsa_sparse_mla_attention(
                ql, qp, kv_deq, kpe, topk, scale, **flags
            )
            mx.eval(o_q8, o_dn)
            assert not bool(mx.any(mx.isnan(o_q8)).item())
            md = float(mx.max(mx.abs(o_q8.astype(mx.float32) - o_dn.astype(mx.float32))).item())
            assert md < 5e-2, f"{dtype} K={K} L={L} flags={flags} max_abs={md}"


def test_batch_filter():
    c0, lat0, _ = _single(30, 13)
    c1, _, _ = _single(50, 14)
    b = Int8MLALatentCache.merge([c0, c1])
    b.filter(mx.array([0]))  # keep only row 0
    assert b.size() == 30  # left-pad shifted away (was 50, row0 had 20 pad)
    e = b.extract(0)
    relat = mx.dequantize(*e.keys, group_size=GS, bits=8)
    mx.eval(relat)
    assert e.offset == 30
    assert float(mx.max(mx.abs(relat - lat0)).item()) == 0.0
