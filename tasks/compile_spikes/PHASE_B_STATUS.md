# LEVER #1 Phase B — integration status — compile-builder — 2026-07-04

## Milestone 1: SEAM REFACTOR DONE + offline parity proxy PASS

### What shipped (all additive, gated OMLX_M3_COMPILE=1, DEFAULT OFF)
- **NEW** `.../minimax_m3_vl/compiled_decode.py` — the compiled L=1 decode machinery:
  - `CompiledDecoder`: host orchestrator (regime gate, lazy build, 256-growth rebucket,
    cache-sync writeback, telemetry counters `compiled_calls`/`rebuilds`).
  - compiled step over the whole 60-layer stack; **offset carried as an `mx.array` arg**
    (1 trace/bucket), **KV+index cache buffers as `inputs=/outputs=` state** written with
    `mx.slice_update` at the dynamic offset.
  - reuses the REAL submodules (norms/packed proj/rope/`block_sparse_moe`/`mlp`) so math ==
    eager; fused_index kernels driven directly with dynamic `mx.array` params (fused_index.py
    UNTOUCHED). Sparse attn = compact-gather + masked SDPA (mirrors eager, not flash, for parity).
  - dense layers (0–2) = full SDPA over full-cap buffer + dynamic length mask.
- **EDIT** `language.py`: +1 import, +1 gated hook in `MiniMaxM3Model.__call__` (engages only
  L=1 / B=1 / no hidden-capture / mask None|str / all-sparse-above-crossover). Eager path and
  census/route-trace untouched; hook is a no-op when the env is off.

### Offline parity proxy (tasks/compile_spikes/spike_phaseb_smoke.py) — PASS
Tiny real `MiniMaxM3Model` (5 layers: 3 dense + 2 sparse-MoE, head_dim=128, kv=idx heads=4),
prefill 2100 (> the 2048 sparse crossover), teacher-forced decode **260 steps across one 256
cache-growth boundary**, EAGER vs COMPILED from identical warm state. TWO configs:

| config | compiled steps | rebuilds | argmax | max|Δlogit| |
|---|---|---|---|---|
| unquantized (separate proj) | 260/260 | 2 (init+growth) | 260/260 | **0.0** |
| affine gs64 b4 (packed "full" proj + gather_qmm MoE) | 260/260 | 2 | 260/260 | **0.0** |

Bit-identical, both configs. Sparse selection is a TRUE approximation here (16 of 17 blocks),
so identity confirms the fused topk + compact-gather + slice_update-state path matches eager
exactly — not a dense-fallback coincidence. Compiled path served every step (no silent fallback).

### Verified mechanisms (composing Spikes A–E on the real classes)
offset-as-mx.array rope (Δ=0 vs int, 1 trace) · nested compile (block_sparse_moe→_minimax_moe_select
is @mx.compile) · 60-block module-capture + multi-array state · packed quantized_matmul + gather_qmm
· fused metal kernels with dynamic K/params · slice_update cache state · 256-growth rebucket.

## Milestone 2: REAL fs5 PARITY GATE — PASS (2026-07-04, coordinated window)
Standalone (server stopped), `phaseb_real_parity.py`, real `MiniMax-M3-oQNVFP4-fs5`,
WARM_LEN=5008 (compiled engages step 1), 220 teacher-forced decode steps:
- **STEP-1 max|Δlogit| = 0.0** → nvfp4 `weight_scale_2` ts-carry survives the compiled
  trace exactly (per-expert tensor gather, not a baked scalar — the key risk, cleared).
- **all-step max|Δlogit| = 0.0**, argmax **220/220**, no divergence — BIT-IDENTICAL on the
  real nvfp4+ts model. compiled served 220/220 (no fallback), 2 rebuilds (init + growth).
- Server relaunched at production env, `/health` 200 in 2s (loaded_count=0, lazy reload).

### Step-time preview + the strategic gap
- Standalone (single-stream, NO MTP, DEFAULT mlx env): eager 48.3 → compiled 44.3 ms/tok
  = **1.09x, Δ4ms**. Rough lower bound; NOT comparable to production 28.4 tok/s.
- **MTP gap**: production decode runs MTP with L=2 verify; the L=1 compiled hook does NOT
  engage there. As-built the win doesn't reach the production MTP path. Also, the compiled
  region is only the 60-layer stack — per-token python wrappers (embed, lm_head@200k vocab,
  cache writeback, sampling sync) stay eager, capping the standalone delta.
- Decision pending with lead: (a) extend compile to the L=2..8 MTP verify bucket (_multi
  kernels exist); (b) server A/B with MTP off (golden env) to size the true L=1 ceiling
  first; (c) rescope. Parity foundation is solid regardless.

## Milestone 3: server A/B bug — offset-as-mx.array in _build (FIXED)
Bench owner hit a crash at first long-ctx engage on the live server (not offline):
`_build` seeded compiled buffers via a python slice `keys[:, :, :offset, :]`, but server
batch/quantized caches (mlx_vlm.models.cache) carry `offset` as an **mx.array** → "Slice
indices must be integers". Offline smoke missed it (standalone cache uses int offsets) —
a live-vs-offline gap, exactly what the engagement-log house rule guards against.
FIX: `_seed_to_cap()` seeds by BUFFER SHAPE (`mx.pad` tail) with zero offset reads →
robust to int or mx.array offsets; correctness-neutral (rows past valid length are never
read: sparse bound by K=off+1, dense by arange<=off mask). VERIFIED: smoke still
bit-identical (regression) + reproduced the server condition (sparse offsets forced to
mx.array, dense int) → no crash, engaged, Δlogit=0.0. Audited: no other python-slice-with-
array remains; only offset into the compiled step is `off=mx.array(cache[0].offset)` (dense
int, gate-checked). Note: fs5 attention is affine 8b/5b MIXED → packed "full" proj returns
None → compiled path uses the separate-proj branch (already covered by the passing parity gate).

### Milestone 3b: batch-cache robustness (live MiniMaxM3BatchKVCache)
Root cause of the live crash: the server uses MiniMaxM3BatchKVCache (offset = mx.array,
NOT int). Three landmines, all fixed at the HOST boundary (never in the per-step trace):
1. `_build` python slice `keys[:,:,:offset,:]` with array offset → FIXED via `_seed_to_cap`
   (shape-based, no offset read).
2. writeback `c.offset = new` — batch cache `.offset` is a READ-ONLY property → would crash
   → FIXED via `_advance_offset` (writes underlying kv_cache.offset, preserves int vs
   mx.array type, sets index_offset).
3. regime gate `isinstance(off, int)` would REJECT array offsets (never engage) → FIXED via
   `_as_int` (int | scalar mx.array → int; multi-elem → None); + guard rejecting PADDED
   batches (`_omlx_all_zero_padding is False`) so only unpadded singletons (the live
   fast-path) engage.
VERIFIED offline: smoke bit-identical (regression); `_advance_offset` unit-tested against a
mock read-only `.offset` property (array preserved, no crash); full array-offset decode
Δlogit=0.0; `_as_int` coercion table. CAVEAT: batch-path NUMERICAL correctness is unverified
offline (can't instantiate the batch cache) — live temp0 TOKEN-IDENTITY (compiled vs MTP-off
eager) is the correctness gate for the A/B.

### Milestone 3c: batch-cache landmine #3 + OFFLINE batch verification
Live re-run (post 3b) engaged ([M3COMPILE] fired, bucket_cap=10240) but produced zero output:
`MiniMaxM3BatchKVCache.offset` is a READ-ONLY property → the writeback's offset-set raised
AttributeError → scheduler classified it as unrecoverable cache corruption → clear+re-prefill
loop. FIX: `_advance_offset` unwraps `getattr(kv, "kv_cache", kv)` to the inner BatchKVCache's
settable offset (preserving array type) + try/except so it can NEVER raise.

Then CLOSED the batch-path gap OFFLINE (was wrong that I couldn't — `MiniMaxM3KVCache.to_batch([0])`
builds the exact live `MiniMaxM3BatchKVCache`). `spike_phaseb_batch.py` reproduces the precise
live condition (sparse = batch wrapper, offset = array, `.offset` read-only; dense `cache[0]` = int)
→ compiled engaged 40/40, **argmax 40/40 TOKEN-IDENTICAL** to eager on the real batch cache.
So the full compiled decode is now verified token-identical on BOTH the non-batch AND the real
batch cache offline. The 3 batch landmines (offset slice / read-only offset / gate int-check) are
all fixed at the host boundary; per-step trace stays retrace-free (off = mx.array).

### Milestone 3d: bug CLASS killed offline via the SCHEDULER-built batch cache
Per lead directive (stop one-landmine-per-bench-cycle). Built the repro the way the scheduler
does — `type(c).merge([c])` per layer (dense→BatchKVCache, sparse→MiniMaxM3BatchKVCache, array
offsets, read-only wrapper `.offset`) — NOT `to_batch`. This caught a 4th issue `to_batch` missed:
the batch path's `create_attention_mask` returns an ALL-TRUE bool mask `[1,1,1,K]` at L=1
(non-batch returns None) → the gate rejected array masks → wouldn't engage. FIX: accept an
all-True bool mask (no-op; compiled re-derives causality from offset); bail on any mask with a
False entry.
API audit (every attr the compiled path touches, wrapper vs inner): wrapper `.offset` = READ-ONLY
property (unwrap to inner `.kv_cache.offset`, settable); wrapper `index_keys`/`index_offset`
settable; inner `keys`/`values`/`offset` settable — never write a read-only attr.
`spike_phaseb_batch.py` (merge-built, +256-growth, +Δlogit): served 260/260, rebuilds 2,
argmax 260/260, **max|Δlogit|=0.0**, offsets valid post-run. Non-batch smoke still bit-identical.
compiled_decode.py sha256 after all fixes = **a1a037ce…9901** (failing run was 3687905…).
Residual: repro covers cache+model, not the full engine (scheduler/batch-gen/finalize) — gated by
the live temp0 token-identity check.

### Milestone 3e: cross-request KV contamination (caught by live token-identity gate) — FIXED
Live re-run (post 3d): mechanically clean (0 corruption, [M3COMPILE] engaged, NIAH@32k 3/3,
retrace=3) BUT the temp0 token-identity gate caught a CORRECTNESS bug — the compiled parity
request's output leaked a PRIOR request's nonce. ROOT CAUSE: `self.state` seeded once per
cap-bucket, reseeded only on GROWTH; a shorter NEW request reusing an existing bucket (no growth
→ no rebuild) decoded on the PRIOR request's KV.
FIX: reseed per request. `_continuation(cache)` — after each step's writeback `cache[i].keys IS`
my state buffer AND `cache[0].offset == self._last_offset`; a new request re-prefills (fresh key
arrays + reset offset), tripping both → `forward()` rebuilds/reseeds from that request's warm
cache. Host-side, per-request; per-token trace stays retrace-free.
VERIFIED `spike_phaseb_reuse.py` (persistent decoder, A len 2100 → B len 2050 same bucket, diff
content): B reseeded (rebuilds 1→2), B argmax 24/24 == B eager. Full offline suite green
(non-batch bit-identical, merge-batch bit-identical +growth, cross-request PASS).
compiled_decode.py sha256 = **9959bdc0…4039** (a1a037ce now stale).
Lesson: single-request offline PASS ≠ multi-request-live correct — cross-request test now permanent.

## (superseded) Remaining before server A/B (needs the real fs5 weights — coordinate window)
1. **REAL parity gate**: ≥200-token temp0 argmax-agreement compiled-vs-eager on fs5 (234GB,
   real 60-layer, real nvfp4). `mx.set_wired_limit(506GB)`; NOT during a big server bench.
2. **nvfp4 + weight_scale_2 (ts) MoE**: could NOT be synthesized offline (needs a real nvfp4
   ckpt's second-level scales). The compiled path calls the same `block_sparse_moe`/PackedSwitchGLU
   incl. the fused `swiglu_oai_ts` kernel (Spike B proved metal-kernel compile-safe), but this is
   the one production op path unexercised offline. First thing to watch in the fs5 gate.
3. **First standalone bench**: tok/s compiled vs eager @ long ctx (the 35→~22-25ms prize), and
   re-check no regression at the golden env (OPS=4000/MB=4000) — op-count cut also cuts cbufs/token.
4. Then request the server A/B window.

## Notes / design record (for codex-reviewer)
- python-int offset tracked in lockstep with the mx.array (cache.offset writeback) — NO `.item()`
  sync on the hot path.
- regime gate keeps the compiled path to the ONE steady sparse-decode bucket; short ctx / padding /
  capture / verify(L>1) all fall to eager. MTP/EAGLE stays eager (per the box).
- kill switch: unset OMLX_M3_COMPILE (default) → zero code-path change.
