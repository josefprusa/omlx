# LEVER #1 (mx.compile) — Phase A results — compile-builder — 2026-07-04

Offline de-risking spikes for wrapping the MiniMax-M3 single-token decode
forward in `mx.compile`. MLX 0.31.2. No server, no model load. All spike
scripts live beside this file and print their own PASS/FAIL + microbench line.

## Verdicts

| Spike | What | Verdict | Key number |
|---|---|---|---|
| A | compile × quantized_matmul + gather_qmm (affine gs64 b4/b8, mxfp8, nvfp4) | **PASS** | bit-IDENTICAL all 4 modes; fused chain 1.20x @tiny |
| B | compile × real fused_index metal kernels (block-max + flash SDPA) | **PASS** | bit-IDENTICAL; scalar-as-array refactor mandatory |
| C | stateful KV cache via inputs=/outputs=, dynamic mx.array offset, 300 steps + 256-growth | **PASS** | parity 4.8e-7; 1 trace/bucket; retrace scales in D |
| D | retrace cost at 60-layer scale | **PASS** | ~11ms/bucket, FLAT vs cap; 43us/token amortized |
| E | op-count / buffer reduction | **PASS** | 4931→3007 ops (39% fewer), compiled ≡ eager (Δ=0) |

**NO spike FAILED.** No blocking error encountered.

## The three findings that shape Phase B

1. **mx.compile is safe-by-recompile, NOT stale.** It cache-keys on python
   *scalar arg values*. Passing a per-step scalar (total_len/cur_block/q_start)
   as a python int does not go stale — it RE-TRACES once per distinct value.
   Decode increments total_len every token ⇒ **retrace every token** (kills the
   host-cost win) **+ unbounded compile-cache growth** (leak).
   **Fix (proven):** pass those scalars as `mx.array` inputs (constant shape)
   ⇒ exactly ONE trace per shape bucket, correct across all values.

2. **Dynamic offset + slice_update on captured state works.**
   `mx.slice_update(buf, upd, start_indices=off.reshape(1), axes=[2])` with
   `off` an `mx.array` writes at the dynamic position with zero python-int
   branches. 256 steps inside one 256-cap bucket = 1 trace (counter-proven).
   Cache growth = rebucket (fresh `mx.compile` for the new cap); ~11ms, negligible.

3. **Python side-effects inside a compiled fn run once PER TRACE, not per call**
   (fired 1x over 50 calls). Census/telemetry counters MUST stay on the
   uncompiled fallback path or they undercount 50→1 (validates plan line 52).

## Op-reduction detail (Spike E), 60-layer decode graph
- eager 4931 ops (82/layer) → compiled 3007 ops (50/layer), **39% fewer**.
- 301 fused `Compiled*` kernels. Fusion eats the glue: Broadcast 1084,
  Reshape 780, Multiply 724, Add 241 collapse; QuantizedMatmul 240 unchanged
  (primitives don't fuse; their surrounding elementwise does).
- Compiled forward **bit-identical** to eager (max|Δ| = 0.0).
- → 39% fewer kernel dispatches ⇒ fewer command-buffer commits ⇒ **less Lever-#2
  backpressure (MAX_ACTIVE_TASKS=10 / MAX_OPS_PER_BUFFER)**. Levers compound.

## Integration-feasibility verdict: FEASIBLE (all mechanisms de-risked)
Every primitive the M3 decode step uses — affine/mxfp8/nvfp4 quantized ops,
gather_qmm MoE, custom metal kernels, a mutating KV cache — is compile-traceable
and bit-identical under compile. The only landmines are known and have proven
fixes (scalars-as-arrays; census off the compiled path; rebucket on growth).

Caveat this Phase cannot answer offline: the *magnitude* of the batch-1 tok/s
win. The mechanism (skip ~4931 python op-constructions/token on cache hits, the
~29% host cost) is sound and Spike E quantifies the op cut, but the real wall
delta needs Phase B on the server. Per plan line 49, if Lever #2 already drives
wall≈GPU at batch-1, #1's batch-1 win shrinks toward 0 → rescope to CPU-burn /
batch≥2 / host-overlap. **Phase B is gated on the lead relaying #2's result.**

## Recommended Phase-B approach (when go)
1. Refactor every fused_index/kernel wrapper on the decode path to accept
   per-step scalars (total_len, cur_block, local_start, q_start) as `mx.array`
   inputs — NOT python ints baked via `mx.array([...])`. Bounded, well-scoped.
2. Carry KV cache + offset as `mx.compile(inputs=state, outputs=state)`; offset
   an `mx.array`; writes via `slice_update` (Spike C pattern).
3. Bucket the compiled step by cache capacity; rebucket (fresh `mx.compile`) at
   each 256-growth boundary (~11ms, 43us/token amortized).
4. Census/counters stay on the uncompiled fallback path (side-effect finding).
5. Gate behind `OMLX_M3_COMPILE=1` (default OFF).
6. Gates: argmax-agreement 200 tokens temp0 vs uncompiled + probe benches +
   NIAH 12/12 @32k.
