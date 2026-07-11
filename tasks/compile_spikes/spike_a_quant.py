# SPDX-License-Identifier: Apache-2.0
"""LEVER #1 Phase A - Spike A.

Question: can mx.compile wrap quantized_matmul + gather_qmm across the three
quant modes M3/GLM actually use (affine gs64, nvfp4, mxfp8) and produce
bit-identical results vs the eager path?  If yes, the MoE gate/up/down + dense
projections inside the decode step are compile-safe.

Offline only. No model load. Decode-shaped (batch-1 token) calls.
"""
from __future__ import annotations

import time
import mlx.core as mx

mx.random.seed(0)

# ---- decode-ish dims (kept small so the spike runs in <1s) ----
HID = 2048       # divisible by 64/32/16 -> valid for all modes
OUT = 1408       # MoE intermediate-ish
E = 32           # experts
K = 8            # top-k routed experts (decode: 1 token -> K experts)
ITERS = 200

MODES = [
    ("affine", 64, 4),
    ("affine", 64, 8),
    ("mxfp8", 32, 8),
    ("nvfp4", 16, 4),
]


def _q(w, mode, gs, bits):
    packed = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
    qw, sc, *bi = packed
    return qw, sc, (bi[0] if bi else None)


def _bench(fn, *args):
    o = fn(*args)
    mx.eval(o)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        o = fn(*args)
        mx.eval(o)
    return (time.perf_counter() - t0) / ITERS * 1e3  # ms/call


def test_quantized_matmul():
    print("\n--- quantized_matmul (dense proj, x=[1,HID]) ---")
    x = mx.random.normal((1, HID)).astype(mx.bfloat16)
    w = mx.random.normal((OUT, HID)).astype(mx.bfloat16)
    ok = True
    for mode, gs, bits in MODES:
        qw, sc, bi = _q(w, mode, gs, bits)

        def f(x, qw, sc, bi):
            return mx.quantized_matmul(
                x, qw, sc, bi, transpose=True,
                group_size=gs, bits=bits, mode=mode,
            )

        cf = mx.compile(f)
        eager = f(x, qw, sc, bi)
        comp = cf(x, qw, sc, bi)
        mx.eval(eager, comp)
        max_abs = float(mx.max(mx.abs(eager - comp)))
        identical = bool(mx.all(eager == comp))
        te = _bench(f, x, qw, sc, bi)
        tc = _bench(cf, x, qw, sc, bi)
        status = "IDENTICAL" if identical else f"max|d|={max_abs:.3e}"
        print(f"  {mode:7} gs={gs} b={bits}: {status:20} "
              f"eager={te:.4f}ms compiled={tc:.4f}ms")
        ok = ok and (identical or max_abs < 1e-2)
    return ok


def test_gather_qmm():
    # Mirror QuantizedSwitchLinear.__call__ exactly:
    #   x [T,1,1,HID], w [E,OUT,HID] packed, rhs_indices [T,K] -> [T,K,1,OUT]
    print("\n--- gather_qmm (MoE expert path, mirrors switch_layers.py) ---")
    T = 1
    x = mx.random.normal((T, 1, 1, HID)).astype(mx.bfloat16)
    w = mx.random.normal((E, OUT, HID)).astype(mx.bfloat16)
    idx = mx.array([[i for i in range(K)]], dtype=mx.uint32)  # [T,K]
    ok = True
    for mode, gs, bits in MODES:
        qw, sc, bi = _q(w, mode, gs, bits)

        def f(x, qw, sc, bi, idx):
            return mx.gather_qmm(
                x, qw, sc, bi, rhs_indices=idx, transpose=True,
                group_size=gs, bits=bits, mode=mode, sorted_indices=False,
            )

        cf = mx.compile(f)
        eager = f(x, qw, sc, bi, idx)
        comp = cf(x, qw, sc, bi, idx)
        mx.eval(eager, comp)
        max_abs = float(mx.max(mx.abs(eager - comp)))
        identical = bool(mx.all(eager == comp))
        te = _bench(f, x, qw, sc, bi, idx)
        tc = _bench(cf, x, qw, sc, bi, idx)
        status = "IDENTICAL" if identical else f"max|d|={max_abs:.3e}"
        print(f"  {mode:7} gs={gs} b={bits}: {status:20} "
              f"out={tuple(eager.shape)} eager={te:.4f}ms compiled={tc:.4f}ms")
        ok = ok and (identical or max_abs < 1e-2)
    return ok


def test_fused_chain():
    # Realistic: compile a whole gate_up -> swiglu -> down chain (the shape a
    # single MoE layer's expert compute takes), so ops around the qmm fuse.
    print("\n--- compiled gate_up->silu*x->down chain (affine gs64 b4) ---")
    gs, bits, mode = 64, 4, "affine"
    x = mx.random.normal((1, HID)).astype(mx.bfloat16)
    wg = mx.random.normal((OUT, HID)).astype(mx.bfloat16)
    wu = mx.random.normal((OUT, HID)).astype(mx.bfloat16)
    wd = mx.random.normal((HID, OUT)).astype(mx.bfloat16)
    qg = _q(wg, mode, gs, bits)
    qu = _q(wu, mode, gs, bits)
    qd = _q(wd, mode, gs, bits)

    def chain(x, qg, qu, qd):
        g = mx.quantized_matmul(x, *qg, transpose=True, group_size=gs, bits=bits, mode=mode)
        u = mx.quantized_matmul(x, *qu, transpose=True, group_size=gs, bits=bits, mode=mode)
        h = (g * mx.sigmoid(g)) * u  # silu(g)*u
        return mx.quantized_matmul(h, *qd, transpose=True, group_size=gs, bits=bits, mode=mode)

    cf = mx.compile(chain)
    eager = chain(x, qg, qu, qd)
    comp = cf(x, qg, qu, qd)
    mx.eval(eager, comp)
    max_abs = float(mx.max(mx.abs(eager - comp)))
    te = _bench(chain, x, qg, qu, qd)
    tc = _bench(cf, x, qg, qu, qd)
    print(f"  chain: max|d|={max_abs:.3e} eager={te:.4f}ms compiled={tc:.4f}ms "
          f"speedup={te/tc:.2f}x")
    return max_abs < 1e-2


if __name__ == "__main__":
    r1 = test_quantized_matmul()
    r2 = test_gather_qmm()
    r3 = test_fused_chain()
    allok = r1 and r2 and r3
    print("\n" + "=" * 60)
    print(f"SPIKE A quantized_matmul compile-safe: {r1}")
    print(f"SPIKE A gather_qmm compile-safe:       {r2}")
    print(f"SPIKE A fused chain compile-safe:      {r3}")
    print(f"SPIKE A VERDICT: {'PASS' if allok else 'FAIL'}")
