# SPDX-License-Identifier: Apache-2.0
"""LEVER #1 Phase A - Spike C.

Question: can a compiled decode step carry a MUTATING KV cache as compile
inputs=/outputs= state, write new K/V with slice_update at a DYNAMIC offset
(offset carried as an mx.array -> zero python-int branches inside the step),
stay correct over 300 steps, survive a 256-boundary buffer growth (which forces
a recompile), and how expensive is that retrace?

Design mirrors decode: per step we get k_new/v_new/q [1,H,1,D], slice_update
into the cache at position `offset` (mx.array), then do a masked attention
readout over the valid cache (mask = arange(CAP) < offset, fully dynamic).
Cache grows in 256-slot chunks like mlx_lm KVCache; we rebucket the compiled
fn at the boundary (matches Phase-B "bucket recompile at cache-growth").

Offline only. No model load.
"""
from __future__ import annotations

import time
import mlx.core as mx

mx.random.seed(0)

H, D = 8, 128
STEP = 256          # cache growth chunk (mlx_lm KVCache default)
N_STEPS = 300       # crosses exactly one 256 boundary
SCALE = 1.0 / (D ** 0.5)


def make_step(cache_state):
    """Build a compiled decode step bound to `cache_state` (dict of arrays).
    offset is read/written as an mx.array -> no python-int branch inside."""

    def step(q, k_new, v_new):
        off = cache_state["offset"]                      # mx.array scalar
        ck = mx.slice_update(cache_state["k"], k_new,
                             start_indices=off.reshape(1), axes=[2])
        cv = mx.slice_update(cache_state["v"], v_new,
                             start_indices=off.reshape(1), axes=[2])
        cache_state["k"] = ck
        cache_state["v"] = cv
        cap = ck.shape[2]                                # python int: STATIC per trace
        # masked attention readout over valid positions (mask is DYNAMIC via off)
        scores = (q @ ck.swapaxes(-1, -2)) * SCALE       # [1,H,1,cap]
        valid = (mx.arange(cap) <= off)                  # <= off: includes just-written
        scores = mx.where(valid, scores, -1e9)
        w = mx.softmax(scores, axis=-1)
        out = w @ cv                                     # [1,H,1,D]
        cache_state["offset"] = off + 1
        return out

    return mx.compile(step, inputs=cache_state, outputs=cache_state)


def eager_reference(qs, ks, vs, n):
    """Ground-truth: plain growing python-list cache, uncompiled."""
    kbuf, vbuf, outs = [], [], []
    for i in range(n):
        kbuf.append(ks[i]); vbuf.append(vs[i])
        K = mx.concatenate(kbuf, axis=2)                 # [1,H,i+1,D]
        V = mx.concatenate(vbuf, axis=2)
        s = (qs[i] @ K.swapaxes(-1, -2)) * SCALE
        w = mx.softmax(s, axis=-1)
        outs.append(w @ V)
    return outs


def new_cache(cap):
    return {
        "k": mx.zeros((1, H, cap, D)),
        "v": mx.zeros((1, H, cap, D)),
        "offset": mx.array(0, dtype=mx.int32),
    }


if __name__ == "__main__":
    # fixed per-step inputs so eager and compiled see identical data
    qs = [mx.random.normal((1, H, 1, D)) for _ in range(N_STEPS)]
    ks = [mx.random.normal((1, H, 1, D)) for _ in range(N_STEPS)]
    vs = [mx.random.normal((1, H, 1, D)) for _ in range(N_STEPS)]
    mx.eval(qs, ks, vs)

    ref = eager_reference(qs, ks, vs, N_STEPS)
    mx.eval(ref)

    # compiled run with growth + rebucket at the 256 boundary
    cap = STEP
    state = new_cache(cap)
    trace_count = [0]
    _orig = mx.slice_update

    step = make_step(state)

    step_ms = []
    retrace_ms = None
    max_err = 0.0
    n_rebuckets = 0

    for i in range(N_STEPS):
        # grow BEFORE writing if we'd exceed capacity (offset == cap)
        if i == cap:  # offset has reached capacity -> grow + rebucket
            newcap = cap + STEP
            gk = mx.zeros((1, H, newcap, D))
            gv = mx.zeros((1, H, newcap, D))
            gk = mx.slice_update(gk, state["k"], start_indices=mx.array([0]), axes=[2])
            gv = mx.slice_update(gv, state["v"], start_indices=mx.array([0]), axes=[2])
            mx.eval(gk, gv)
            state["k"], state["v"] = gk, gv
            cap = newcap
            step = make_step(state)     # rebucket: fresh compiled fn for new cap
            n_rebuckets += 1
            t0 = time.perf_counter()
            out = step(qs[i], ks[i], vs[i])   # this call pays the retrace
            mx.eval(out, state["k"], state["v"], state["offset"])
            retrace_ms = (time.perf_counter() - t0) * 1e3
        else:
            t0 = time.perf_counter()
            out = step(qs[i], ks[i], vs[i])
            mx.eval(out, state["offset"])
            step_ms.append((time.perf_counter() - t0) * 1e3)
        err = float(mx.max(mx.abs(out - ref[i])))
        max_err = max(max_err, err)

    # steady-state compiled vs eager single-step (same shapes, cap=256, off~128)
    st = new_cache(STEP)
    st["offset"] = mx.array(128, dtype=mx.int32)
    cstep = make_step(st)
    q1, k1, v1 = qs[10], ks[10], vs[10]
    mx.eval(cstep(q1, k1, v1))                    # warm
    ITER = 300
    t0 = time.perf_counter()
    for _ in range(ITER):
        st["offset"] = mx.array(128, dtype=mx.int32)  # pin offset so shape stable
        mx.eval(cstep(q1, k1, v1))
    comp_ss = (time.perf_counter() - t0) / ITER * 1e3

    # eager single-step equivalent (offset 128, cap 256, masked)
    def eager_step(q, k_new, v_new, ck, cv, off):
        ck = mx.slice_update(ck, k_new, start_indices=off.reshape(1), axes=[2])
        cv = mx.slice_update(cv, v_new, start_indices=off.reshape(1), axes=[2])
        s = (q @ ck.swapaxes(-1, -2)) * SCALE
        valid = (mx.arange(ck.shape[2]) <= off)
        s = mx.where(valid, s, -1e9)
        return mx.softmax(s, axis=-1) @ cv
    ek = mx.zeros((1, H, STEP, D)); ev = mx.zeros((1, H, STEP, D))
    eoff = mx.array(128, dtype=mx.int32)
    mx.eval(eager_step(q1, k1, v1, ek, ev, eoff))
    t0 = time.perf_counter()
    for _ in range(ITER):
        mx.eval(eager_step(q1, k1, v1, ek, ev, eoff))
    eager_ss = (time.perf_counter() - t0) / ITER * 1e3

    steady_avg = sum(step_ms[10:]) / len(step_ms[10:])
    passed = (max_err < 1e-3) and (n_rebuckets == 1)
    print(f"steps run:              {N_STEPS} (crossed 1x 256-boundary)")
    print(f"max |compiled-eager|:   {max_err:.3e}  (parity over full run)")
    print(f"cache growths/rebuckets:{n_rebuckets}")
    print(f"retrace step latency:   {retrace_ms:.3f} ms  (single post-growth call)")
    print(f"steady compiled step:   {comp_ss:.4f} ms")
    print(f"steady eager step:      {eager_ss:.4f} ms")
    print(f"compile speedup:        {eager_ss/comp_ss:.2f}x")
    print(f"retrace overhead:       {retrace_ms - comp_ss:.3f} ms amortized over "
          f"{STEP} steps = {(retrace_ms-comp_ss)/STEP*1e3:.2f} us/token")
    print("=" * 60)
    print(f"SPIKE C VERDICT: {'PASS' if passed else 'FAIL'}  "
          f"(dynamic mx.array offset + slice_update state, 300 steps, "
          f"1 growth, parity {max_err:.1e})")
