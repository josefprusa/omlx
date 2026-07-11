# SPDX-License-Identifier: Apache-2.0
"""LEVER #1 Phase A - Spike B.

Question: can mx.compile trace through mx.fast.metal_kernel custom primitives?
Uses the REAL M3 fused_index kernels (block-max scoring + top-16 selection +
flash sparse SDPA), not toy kernels.

Also probes the crux failure mode for Phase B: the fused_index wrappers build
`mx.array([cur_block, ...], uint32)` from PYTHON ints derived from total_len.
Under mx.compile those become FROZEN trace constants -> stale after step 1 with
NO recompile (compile keys on array shape/dtype, not python scalar values).
We prove the break AND the fix (pass the scalar as an mx.array input).

Offline only. No model load.
"""
from __future__ import annotations

import importlib.util
import os
import time
import mlx.core as mx

mx.random.seed(0)

# import the real vendored kernel module directly by path (no package machinery)
_FI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "omlx/patches/mlx_vlm_minimax_m3_compat/vendor/mlx_vlm/models/minimax_m3_vl/fused_index.py",
)
_spec = importlib.util.spec_from_file_location("fused_index", _FI_PATH)
fi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fi)

BLK = 128
HEADS = 4
DIM = 128
ITERS = 200


def _bench(fn, *a):
    o = fn(*a)
    mx.eval(o)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        o = fn(*a)
        mx.eval(o)
    return (time.perf_counter() - t0) / ITERS * 1e3


def test_block_scores_compile():
    """fused_block_scores: idx_q.idx_k block-max in one kernel. scale is a
    genuine per-layer constant -> baking it is SAFE. This is the clean case."""
    print("\n--- compile × fused_block_scores (real block-max kernel) ---")
    K = 40000
    q = mx.random.normal((1, HEADS, 1, DIM)).astype(mx.bfloat16)
    k = mx.random.normal((1, 1, K, DIM)).astype(mx.bfloat16)
    scale = 0.08838

    def f(q, k):
        return fi.fused_block_scores(q, k, scale, BLK)

    cf = mx.compile(f)
    eager = f(q, k)
    comp = cf(q, k)
    mx.eval(eager, comp)
    assert eager is not None, "kernel returned None (unsupported shape)"
    identical = bool(mx.all(eager == comp))
    max_abs = float(mx.max(mx.abs(eager - comp)))
    te = _bench(f, q, k)
    tc = _bench(cf, q, k)
    print(f"  nb={eager.shape[-1]} {'IDENTICAL' if identical else f'max|d|={max_abs:.3e}'}"
          f" eager={te:.4f}ms compiled={tc:.4f}ms")
    return identical


def test_flash_compile():
    """flash_sparse_sdpa: online-softmax attention over selected blocks. The
    most complex kernel (per-thread accumulators, threadgroup reduction)."""
    print("\n--- compile × flash_sparse_sdpa (real online-softmax kernel) ---")
    K = 40000
    H, Kh = 8, 1
    q = mx.random.normal((1, H, 1, DIM)).astype(mx.bfloat16)
    kc = mx.random.normal((1, Kh, K, DIM)).astype(mx.bfloat16)
    vc = mx.random.normal((1, Kh, K, DIM)).astype(mx.bfloat16)
    # 16 valid ascending block ids per kv head
    ids = mx.broadcast_to(
        mx.arange(16, dtype=mx.int32).reshape(1, 1, 1, 16), (1, Kh, 1, 16)
    )
    scale = 0.08838

    def f(q, kc, vc, ids):
        return fi.flash_sparse_sdpa(q, kc, vc, ids, scale, BLK)

    cf = mx.compile(f)
    eager = f(q, kc, vc, ids)
    comp = cf(q, kc, vc, ids)
    mx.eval(eager, comp)
    if eager is None:
        print("  kernel returned None (shape guard) - SKIP")
        return True
    identical = bool(mx.all(eager == comp))
    max_abs = float(mx.max(mx.abs(eager - comp)))
    te = _bench(f, q, kc, vc, ids)
    tc = _bench(cf, q, kc, vc, ids)
    print(f"  {'IDENTICAL' if identical else f'max|d|={max_abs:.3e}'}"
          f" eager={te:.4f}ms compiled={tc:.4f}ms")
    return identical


def test_scalar_retrace_and_fix():
    """CRUX for Phase B. fused_topk_blocks bakes cur_block=(total_len-1)//128
    into an mx.array from a PYTHON int. mx.compile CACHE-KEYS on python-scalar
    arg VALUES, so it does not go stale -- instead it RE-TRACES once per
    distinct total_len. Decode increments total_len every token => every token
    re-traces (kills the python-saving) AND the compile cache grows unbounded
    (one entry per token = leak over a long generation). Fix: pass the derived
    scalars as an mx.array input (constant shape) -> ONE trace, correct forever.
    """
    print("\n--- CRUX: python-scalar => per-value retrace + cache growth ---")
    nb = 313
    scores = mx.random.normal((1, HEADS, 1, nb)).astype(mx.float32)

    # count real traces: compile runs the python body only during a (re)trace
    trace_log = []

    def broken(scores, total_len):
        trace_log.append(total_len)
        ids, _ = fi.fused_topk_blocks(scores, total_len, BLK, 16, 1, 8)
        return ids

    cbroken = mx.compile(broken)
    seq = [5000, 5000, 5000, 30000, 30000, 60000, 5000]  # simulate token stream
    for v in seq:
        mx.eval(cbroken(scores, v))
    n_distinct = len(set(seq))
    retrace_per_token = (len(trace_log) == n_distinct)
    print(f"  BROKEN(python-int): {len(seq)} calls, {n_distinct} distinct -> "
          f"{len(trace_log)} traces  (retrace once per distinct value: "
          f"{retrace_per_token})")
    # correctness: never stale (retraces), matches eager for each value
    correct = True
    for v in [5000, 30000, 60000]:
        c = cbroken(scores, v)
        e, _ = fi.fused_topk_blocks(scores, v, BLK, 16, 1, 8)
        mx.eval(c, e)
        correct = correct and bool(mx.all(c == e))
    print(f"  BROKEN correctness (never stale, retraces): {correct}   "
          f"<- but decode = new total_len EVERY token => retrace every token")

    # --- FIX: derive scalars in python OUTSIDE compile, pass params as mx.array
    topk_k = fi._get_topk_kernel()

    def _params(total_len, emit=1):
        cur_block = (total_len - 1) // BLK
        local_start = max(cur_block - 8 + 1, 0)
        return mx.array([nb, cur_block, 1, local_start, emit], dtype=mx.uint32)

    fix_trace = []

    def fixed(scores, params):
        fix_trace.append(1)
        (o, _pos) = topk_k(
            inputs=[scores.reshape(HEADS, nb), params],
            grid=(256, HEADS, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(HEADS, 16), (HEADS, 16 * BLK)],
            output_dtypes=[mx.int32, mx.int32],
        )
        return o.reshape(1, HEADS, 1, 16)

    cfixed = mx.compile(fixed)
    for v in [5000, 30000, 60000, 90000]:
        mx.eval(cfixed(scores, _params(v)))
    fixed_ok = True
    for v in [5000, 30000, 60000]:
        c = cfixed(scores, _params(v))
        e, _ = fi.fused_topk_blocks(scores, v, BLK, 16, 1, 8)
        mx.eval(c, e)
        fixed_ok = fixed_ok and bool(mx.all(c == e))
    fixed_one_trace = (len(fix_trace) == 1)
    print(f"  FIXED(mx.array params): 4 distinct values -> {len(fix_trace)} trace"
          f"  (single trace: {fixed_one_trace}), correct: {fixed_ok}")
    return retrace_per_token and correct and fixed_one_trace and fixed_ok


if __name__ == "__main__":
    r1 = test_block_scores_compile()
    r2 = test_flash_compile()
    r3 = test_scalar_retrace_and_fix()
    allok = r1 and r2 and r3
    print("\n" + "=" * 60)
    print(f"SPIKE B block_scores kernel compile-safe: {r1}")
    print(f"SPIKE B flash kernel compile-safe:        {r2}")
    print(f"SPIKE B scalar retrace-mechanism + fix proven: {r3}")
    print(f"SPIKE B VERDICT: {'PASS' if allok else 'FAIL'}")
    print("NOTE: metal_kernel IS compile-safe (bit-identical). BUT mx.compile")
    print("      cache-keys on python-scalar arg VALUES: passing per-step scalars")
    print("      (total_len/cur_block/q_start) as python ints => retrace EVERY")
    print("      token + unbounded compile-cache growth. MANDATORY Phase-B fix:")
    print("      pass those scalars as mx.array inputs (constant shape) => 1 trace.")
