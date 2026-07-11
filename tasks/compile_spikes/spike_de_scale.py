# SPDX-License-Identifier: Apache-2.0
"""LEVER #1 Phase A - Spikes D + E on a shared realistic 60-layer decode graph.

D: retrace (recompile) cost at 60-layer scale. Phase B rebuckets the compiled
   step at each cache-growth boundary, so each new bucket pays ONE retrace. We
   measure that recurring cost (metal kernels already globally cached).

E: op-count / buffer reduction from mx.compile. Fewer graph ops => fewer Metal
   kernel dispatches => fewer command-buffer commits => less pressure on Lever
   #2's MAX_ACTIVE_TASKS / MAX_OPS_PER_BUFFER backpressure. Quantified by
   counting primitives in the traced graph (mx.export_to_dot) eager vs compiled.

Dims are small (fast) but the PER-LAYER OP STRUCTURE is faithful to an M3 MoE
decode layer (RMSNorm, quantized q/k/v/o, slice_update KV, masked SDPA, router,
gather_qmm gate_up + swiglu + gather_qmm down, residuals). Op count — which is
what drives graph-build cost AND #2 pressure — is dim-independent.

Offline only. No model load.
"""
from __future__ import annotations

import io
import re
import time
from collections import Counter

import mlx.core as mx

mx.random.seed(0)

L = 60            # layers
HD = 512          # hidden
NH = 8            # attn heads
HDIM = 64         # head dim (NH*HDIM = 512)
KVH = 1           # kv heads (GQA/MLA-ish)
NE = 8            # experts
TOPK = 2          # routed experts
FFN = 512         # expert intermediate
GS, BITS, MODE = 64, 4, "affine"
STEP = 256
SCALE = 1.0 / (HDIM ** 0.5)


def q(w):
    qw, sc, bi = mx.quantize(w, group_size=GS, bits=BITS, mode=MODE)
    return qw, sc, bi


# ---- shared quantized weights (op count is identical whether shared or not) ----
Wq = q(mx.random.normal((NH * HDIM, HD)) * 0.02)
Wk = q(mx.random.normal((KVH * HDIM, HD)) * 0.02)
Wv = q(mx.random.normal((KVH * HDIM, HD)) * 0.02)
Wo = q(mx.random.normal((HD, NH * HDIM)) * 0.02)
Wr = mx.random.normal((NE, HD)) * 0.02                      # router (dense)
Wgu = q(mx.random.normal((NE, 2 * FFN, HD)) * 0.02)         # gate_up experts
Wd = q(mx.random.normal((NE, HD, FFN)) * 0.02)              # down experts
norm_w = mx.ones((HD,))
mx.eval(*Wq, *Wk, *Wv, *Wo, Wr, *Wgu, *Wd, norm_w)


def rmsnorm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6) * norm_w


def layer(x, cache_k, cache_v, off):
    # --- attention block ---
    h = rmsnorm(x)                                              # [1,HD]
    qh = mx.quantized_matmul(h, *Wq, transpose=True, group_size=GS, bits=BITS, mode=MODE)
    kh = mx.quantized_matmul(h, *Wk, transpose=True, group_size=GS, bits=BITS, mode=MODE)
    vh = mx.quantized_matmul(h, *Wv, transpose=True, group_size=GS, bits=BITS, mode=MODE)
    qh = qh.reshape(1, NH, 1, HDIM)
    kh = kh.reshape(1, KVH, 1, HDIM)
    vh = vh.reshape(1, KVH, 1, HDIM)
    ck = mx.slice_update(cache_k, kh, start_indices=off.reshape(1), axes=[2])
    cv = mx.slice_update(cache_v, vh, start_indices=off.reshape(1), axes=[2])
    cap = ck.shape[2]
    # GQA broadcast: repeat kv heads to NH
    ckb = mx.repeat(ck, NH // KVH, axis=1)
    cvb = mx.repeat(cv, NH // KVH, axis=1)
    scores = (qh @ ckb.swapaxes(-1, -2)) * SCALE               # [1,NH,1,cap]
    scores = mx.where(mx.arange(cap) <= off, scores, -1e9)
    attn = mx.softmax(scores, axis=-1) @ cvb                   # [1,NH,1,HDIM]
    attn = attn.reshape(1, NH * HDIM)
    o = mx.quantized_matmul(attn, *Wo, transpose=True, group_size=GS, bits=BITS, mode=MODE)
    x = x + o
    # --- MoE block ---
    h2 = rmsnorm(x)
    logits = h2 @ Wr.swapaxes(-1, -2)                          # [1,NE]
    probs = mx.softmax(logits, axis=-1)
    idx = mx.argpartition(logits, -TOPK, axis=-1)[..., -TOPK:].astype(mx.uint32)  # [1,TOPK]
    gw = mx.take_along_axis(probs, idx, axis=-1)               # [1,TOPK]
    hx = h2.reshape(1, 1, 1, HD)
    gu = mx.gather_qmm(hx, *Wgu, rhs_indices=idx, transpose=True,
                       group_size=GS, bits=BITS, mode=MODE)     # [1,TOPK,1,2FFN]
    g, u = mx.split(gu, 2, axis=-1)
    act = (g * mx.sigmoid(g)) * u                              # silu(g)*u
    dn = mx.gather_qmm(act, *Wd, rhs_indices=idx, transpose=True,
                       group_size=GS, bits=BITS, mode=MODE)     # [1,TOPK,1,HD]
    dn = dn.reshape(1, TOPK, HD)
    moe = (dn * gw.reshape(1, TOPK, 1)).sum(axis=1)            # [1,HD]
    x = x + moe
    return x, ck, cv


def make_forward(cache_state):
    def forward(x):
        off = cache_state["offset"]
        for i in range(L):
            x, ck, cv = layer(x, cache_state["k"][i], cache_state["v"][i], off)
            cache_state["k"][i] = ck
            cache_state["v"][i] = cv
        cache_state["offset"] = off + 1
        return rmsnorm(x)
    return forward


def new_state(cap):
    return {
        "k": [mx.zeros((1, KVH, cap, HDIM)) for _ in range(L)],
        "v": [mx.zeros((1, KVH, cap, HDIM)) for _ in range(L)],
        "offset": mx.array(0, dtype=mx.int32),
    }


def op_stats(*outs):
    buf = io.StringIO()
    mx.export_to_dot(buf, *outs)
    labels = re.findall(r'label ="([^"]+)", shape=rectangle', buf.getvalue())
    return len(labels), Counter(labels)


if __name__ == "__main__":
    x0 = mx.random.normal((1, HD)) * 0.1
    mx.eval(x0)

    # ================= SPIKE E: op-count reduction =================
    print("=== SPIKE E: op count eager vs compiled (60-layer decode graph) ===")
    st_e = new_state(STEP)
    fwd = make_forward(st_e)
    eager_out = fwd(x0)                       # build graph (un-eval'd)
    n_eager, hist_e = op_stats(eager_out)

    st_e2 = new_state(STEP)
    cfwd = mx.compile(make_forward(st_e2))
    comp_out = cfwd(x0)                        # traced+fused graph
    n_comp, hist_c = op_stats(comp_out)
    n_fused = sum(v for k, v in hist_c.items() if k.startswith("Compiled"))

    print(f"  eager    total ops: {n_eager}   ({n_eager/L:.1f}/layer)")
    print(f"  compiled total ops: {n_comp}    ({n_comp/L:.1f}/layer)")
    print(f"  fused 'Compiled*' kernels in compiled graph: {n_fused}")
    print(f"  op reduction: {n_eager - n_comp} ops ({(1-n_comp/n_eager)*100:.1f}%)")
    top_e = ", ".join(f"{k}:{v}" for k, v in hist_e.most_common(6))
    print(f"  eager top ops: {top_e}")

    # ================= SPIKE D: retrace cost at scale =================
    print("\n=== SPIKE D: recompile (retrace) cost at 60-layer scale ===")
    # warm a compiled fwd fully (pays one-time GLOBAL metal kernel compilation)
    st = new_state(STEP)
    f256 = mx.compile(make_forward(st))
    t0 = time.perf_counter(); mx.eval(f256(x0)); cold = (time.perf_counter() - t0) * 1e3
    # steady state at cap=256
    for _ in range(3):
        st["offset"] = mx.array(10, dtype=mx.int32); mx.eval(f256(x0))
    ITER = 100
    t0 = time.perf_counter()
    for _ in range(ITER):
        st["offset"] = mx.array(10, dtype=mx.int32); mx.eval(f256(x0))
    steady = (time.perf_counter() - t0) / ITER * 1e3

    # REBUCKET cost: new compiled fn for a bigger cap. Metal kernels already
    # globally cached, so this isolates the graph re-trace + MLX compile pass —
    # the recurring per-growth cost in Phase B.
    rebucket_ms = []
    cap = STEP
    for gcap in (512, 1024, 2048):
        st_b = new_state(gcap)
        fb = mx.compile(make_forward(st_b))
        t0 = time.perf_counter(); mx.eval(fb(x0)); rb = (time.perf_counter() - t0) * 1e3
        rebucket_ms.append((gcap, rb))

    print(f"  cold first call (incl. global metal compile): {cold:.1f} ms  (once/process)")
    print(f"  steady compiled step (cap=256):               {steady:.3f} ms")
    for gcap, rb in rebucket_ms:
        print(f"  rebucket->cap={gcap}: first-call {rb:.1f} ms  "
              f"(retrace overhead ≈ {rb - steady:.1f} ms)")
    avg_rt = sum(rb for _, rb in rebucket_ms) / len(rebucket_ms) - steady
    # growth boundaries over a 64k-token generation with mlx_lm 256-chunk growth
    n_growths_64k = 64000 // STEP
    amort_us = avg_rt / STEP * 1e3
    print(f"  avg retrace overhead/bucket: {avg_rt:.1f} ms")
    print(f"  amortized over a 256-token bucket: {amort_us:.1f} us/token")
    print(f"  (a 64k-token gen = {n_growths_64k} growths = {avg_rt*n_growths_64k/1000:.2f}s "
          f"total retrace, i.e. {avg_rt*n_growths_64k/64000*1e3:.1f} us/token amortized)")

    print("\n" + "=" * 60)
    e_pass = n_comp < n_eager and n_fused >= 1
    d_pass = avg_rt < steady * 100      # retrace must be bounded/reasonable
    print(f"SPIKE E VERDICT: {'PASS' if e_pass else 'FAIL'} "
          f"({n_eager}->{n_comp} ops, {(1-n_comp/n_eager)*100:.0f}% fewer)")
    print(f"SPIKE D VERDICT: {'PASS' if d_pass else 'FAIL'} "
          f"(rebucket retrace ~{avg_rt:.0f}ms, {amort_us:.0f}us/token amortized)")
