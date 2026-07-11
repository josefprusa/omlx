# Nemotron-3-Ultra decode speed campaign — plan v2.2 (2026-07-05)

Target model: `~/.omlx/models/unigilby/Nemotron-3-Ultra-oQNVFP4` (550B-A55B, 333GB on disk,
334GB resident). Baseline: **7.68 tok/s sustained decode (130.2 ms/token), batch-1**, golden env
`MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000`, MLX 0.31.2. Quality proven
(gsm8k 91.7 / mmlu 85.3 / arc 96.7) — every lever below must preserve it or be gated off.

## v2.2 changelog (M0 + K1 measured; P0-K parked; sort lever added — 2026-07-05 night)

- **M0 MEASURED, P0-1 GO (aggregate gate governs per RC5):** weighted model-level saving
  **47.2ms/token ≥ 25 bar** → GO. Mode locked **affine8-gs64** (beat mxfp8 at these
  shapes). Per-class: mamba 1.44-1.54× (misses the 1.6 gloss; aggregate governs — the
  RC5 sentence exists precisely so staging proceeds on per-class underperformance);
  **attn k/v 0.87-0.91× = LOSS → k_proj/v_proj EXCLUDED**; stage C redefined q/o-only.
  Live expectation after the lessons.md in-stream discount (~50-60% survives):
  **~24-28ms/token → DQ8-alone ≈ 9.4-9.8 tok/s** (below the earlier 10.5-11 projection;
  P0-2/P0-3 stack on top).
- **K1 MEASURED under golden env** (12 reps, in-stream ×48 chained): unsorted-production
  **593.4µs/layer = 28.48ms/token = 1.87× ideal** (spread 21.7%); sorted 520.4µs =
  1.64× (spread 5.6%); **dense-equal-bytes control 543.3µs = 1.71×** (spread 0.8%).
  The first run's 13.35× was confirmed harness artifact (missing golden env).
  Attributed EXPERT-SPECIFIC excess = unsorted − dense control = 50.1µs/layer =
  **2.41ms/token**. RULING (planner ratified): decision rule read 1.87× as formally
  inconclusive, but the dense control is the sharper discriminator — gather_qmm is
  ~as efficient as a dense read at equal bytes; a fused expert-MLP kernel (K3) can
  capture only ~2-4ms/token, nowhere near DQ8. **P0-K PARKED to P2.** The 1.71×-of-ideal
  dense control IS the diffuse in-stream overhead (same animal as GLM foray §10b's
  ~32ms) — assigned to the future cross-model mega-kernel/fused-decoder-layer campaign,
  not this one. The conditional 14-16 tok/s leg is DEAD; base case 12.5-13.5 stands.
- **NEW P0-2d — pre-sorted expert indices** (from K1's own data): sorted vs unsorted =
  73µs/layer = **3.50ms/token gross, measured in-stream**; cost ≈ 3 tiny dispatches/layer
  (~1ms/token) → **net ~2.5ms/token**. Ordering-correctness ruled (see 2d spec):
  weighted-sum epilogue is permutation-invariant when (inds, scores) permute JOINTLY;
  ts-fold gathers by inds so joint permutation is exact; output is allclose-but-not-
  bit-identical (fp sum reorder) — parity bar set accordingly.
- Day-0 status: SF-1/SF-2 implemented + reviewed (SF-1 APPROVE-WITH-NOTES, race verified
  safe via reconstruct_cache truncate-on-miss + save_block pending-write staging; SF-2
  APPROVE). Required before SF live legs: racing-reader test T1 + retract-while-shared-ref
  test T2. Captain ruled KEEP (verified) on the Day-0 MTP creep — governance recorded in
  the review thread; it is NOT campaign work (doctrine (d) unchanged).

## v2.1 changelog (M1 results folded, 2026-07-05 evening)

Lead ran M1 (`~/.claude/jobs/62f9cfe9/tmp/m1_kernel_bench.py`, isolated-eval harness, 200
iters, M=1, Ultra's exact expert shapes). **Headline: nvfp4-gs16, affine4-gs64 and
mxfp4-gs32 are within ±4% of each other — the Super 0.58× gs16 tax does NOT exist at
Ultra's fat shapes at batch-1.** Applied:
- **P1-1 expert requant DELETED** — was last-resort by doctrine (b), now dead on measured
  speed grounds too. A one-line rejection note remains in §4/P1.
- mxfp4 / 5-bit container ideas **closed as moot for Ultra** (measured equal; non-goals).
- **P0-K pivoted** from "nvfp4 vs affine mode gap" to ABSOLUTE gather_qmm efficiency:
  isolated harness shows 824.7µs/layer (fc1+fc2) vs 316.9µs bandwidth-ideal ≈ **38% of
  peak** — but eval-per-op harness inflation is a known trap (lessons.md), so **new K1 =
  in-stream expert-path attribution BEFORE any kernel is written**. If ≥2× headroom
  survives in-stream, K3 = fused expert-MLP kernel (fc1+relu²+fc2, one dispatch/layer,
  10-20ms/token potential — rivals DQ8). If the gap is harness artifact, P0-K downgrades
  to P2 and DQ8 carries the campaign alone.
- M1 batch legs (M=2/4) errored on `rhs_indices` broadcasting — corrected spec: indices
  shape **(M, 1, TOP)**, not (M, TOP). Only matters for the batch≥2 era.

## v2 changelog

Amended per codex review (`tasks/ultra_speed_review_codex.md`, verdict APPROVE-WITH-CHANGES)
and captain doctrine. All 8 codex Required Changes adopted:
- **RC1** P0-1 hook contract fixed (env-gated path must run before the `model_settings is
  None` early return in `apply_post_load_transforms`; fingerprint read from the model
  instance, not a config re-read).
- **RC2** P0-1 transient-memory budget added (+~34-35GB peak naive; mitigated to <1GB by
  module-by-module quantize/replace/release loop).
- **RC3** P0-2a gate expanded: reject nonzero `left_padding`, non-null `lengths`,
  nontrivial masks, B>1, non-bf16; per-reason fallback census fields.
- **RC4** P0-2b split into 2b-lite (ts-precombine) and 2b-topk (fused top-22) with
  real-gate-distribution + tie-stress parity bars.
- **RC5** M0 gate rewritten: correct ratio direction (`bf16_time / q_time >= 1.6`),
  weighted model-level token-saving gate, per-module-class reporting, router gate reported
  separately and excluded from P0-1 savings math.
- **RC6** cross-request stateful test battery added to every stateful lever and to the new
  serving-fix workstream.
- **RC7** DQ8-as-offline-checkpoint promoted to the **preferred production path**;
  load-time DQ8 is the probe/emergency deployment only.
- **RC8** engagement logging upgraded to expected-vs-actual per-layer counters + per-reason
  fallback counts (not one-shot lines).
- Codex arithmetic corrections adopted throughout (78.123GB pure weights; 65.263GB DQ8
  target excluding router gate; 0.890GB state/cache; new totals 47.401/48.421GB).
- Codex gaps adopted: DQ8 staged by module class with **lm_head LAST** under isolated
  ablation; composed end-to-end synthetic forward test (new M6); mamba L>1 probe for P2-2
  (new M7); `left_padding=[0]` singleton handling in P0-2a.
- Codex P2-3 REJECT adopted: dense 6-bit/mixed push moved to non-goals.
- **Captain doctrine** (non-negotiable): (a) QUANT-FIRST — serve tensors in containers
  matching source precision; DQ8 module selection follows NVIDIA's own quantization map,
  with NVIDIA-excluded classes behind their own kill-switches and isolated A/Bs;
  (b) PRESERVE NVIDIA'S WEIGHT VALUES — P1-1 expert requant demoted to last-resort;
  (c) **P0-K nvfp4 gather_qmm kernel workstream promoted to first-class**, parallel with
  DQ8 (pays across M3/Super/Ultra and the probable GLM-5.2-NVFP4 465GB campaign);
  (d) no MTP work in this plan (dropped at conversion; fs5 drafter disabled today).
- The exact-container idea nvfp4→affine5-gs16 was **verified UNSUPPORTED and dropped**:
  MLX 0.31.2 `mx.quantize(..., group_size=16, mode="affine")` rejects gs16 (tested in this
  venv: "group size 16 is not supported"; affine = gs32/64/128). gs32 cannot be exact
  because NVFP4 scales are per-16. The 25-level integer-grid math was sound; the container
  doesn't exist. mxfp4-gs32 stays as a **measurement-only** column in M1 (nvfp4→mxfp4 is
  lossy-on-lossy, barred by doctrine (b) as a conversion path).
- New section 7: SERVING-FIX workstream (prefix-cache commit lag + throttle mispredictor),
  separate deliverable, same pipeline.

**Headline correction to the campaign brief (unchanged from v1, numbers refined).** Only
the routed experts (22.1B active params) are 4-bit; the other ~33B active params (mamba
mixers, shared experts, latent projs, attention, lm_head) are **bf16** and read **every
token**. Pure weight reads are **78.123GB/token** (+0.890GB SSM-state/conv/KV traffic =
79.013GB) → **absolute ceiling 10.36-10.48 tok/s at 819GB/s**. We are at 74% of physics.
Host-side work alone cannot reach 12-16 tok/s; the campaign must cut bytes (P0-1, P0-K)
*and* host serialization (P0-2/P0-3).

---

## 1. Measured host-side split (two 10s `/usr/bin/sample` captures during sustained decode)

Files: `~/.claude/jobs/62f9cfe9/tmp/ultra_sample1.txt`, `ultra_sample2.txt`. Decode thread =
`Thread_70414112` in both (sample1 lines 954-5404; sample2 header at line 901). Main thread is
the idle uvicorn kevent loop (sample1 line 37: 5661/6058 in `kevent`) — ignore it. Tokenizer
frames are scattered on worker threads at ≤7 samples each — detokenization is off the hot path.

| bucket | sample1 (of 6058) | sample2 (of 5866) | avg % | ms/token @130ms |
|---|---|---|---|---|
| cv-wait inside `mx.async_eval` → `eval_impl` → `condition_variable::wait` | 2948 | 2601 | 46.5% | **~60** |
| `eval_impl` CPU work (graph traverse + Metal encode: `gpu::eval`, `Matmul::eval_gpu→gemv`, `binary_op_gpu`, alloc) | ~2056 | ~2179 | 35.5% | **~46** |
| Python forward (nn.Module `slot_tp_call` towers ~40 frames deep) | 1005 | 1037 | 17.1% | **~22** |
| other (loop glue) | ~49 | ~49 | 0.9% | ~1 |

Evidence lines (sample1): `async_eval` branch 5008 @d26; cv-wait 2948 @d28-34; encode work
1447+282+109+108 @d28 with `gemv` 146 / `binary_op` 128 visible; python tower 1005 @d20.
Sample2 additionally shows `mlx::core::fast::metal_kernel` call-setup (102 samples @d52)
inside the python tower — the 48 per-token `ssm_update_kernel` invocations
(`mlx_lm/models/ssm.py:83`) paying Python/binding setup per call (~2ms/token).

Concurrent (not on the decode thread): `com.Metal.CompletionQueueDispatch` 442 samples
(7.3%, includes `MetalAllocator::free`), `CommandQueueDispatch` 402 (6.6%).

**Interpretation.** Decode thread ~100% occupied: **~70ms/token CPU-busy** (22 py + 46
encode + 2 misc) + **~60ms blocked**. GPU must execute ~96ms of byte-reads (section 2), so
~36ms of GPU work overlaps the CPU-busy window and **~25-34ms/token the GPU is starved** —
mostly during the 22ms Python build phase when nothing is being encoded/committed
(mechanism identical to M3, tasks/todo.md "HOST-SERIALIZATION PROFILING": MLX v0.31.2
`MAX_ACTIVE_TASKS=10` compile-time constant blocks `async_eval` callers; the one-ahead
pipeline in `BatchGenerator._step`, .venv .../mlx_lm/generate.py:1320-1378, cannot run ahead).

---

## 2. Op-level cost model of one decode token (v2: codex-recomputed decimals)

Config (`~/.omlx/models/unigilby/Nemotron-3-Ultra-oQNVFP4/config.json`): hidden 8192,
vocab 131072, 108 layers = 48 mamba + 48 MoE + 12 attention. Mamba: 256 heads × 64 dim
(intermediate 16384), n_groups 8, ssm_state 128, conv_kernel 4 → conv_dim 18432, in_proj
8192→35072. Attention: GQA 64 heads / 2 KV, head_dim 128, **no RoPE**. MoE: 512 experts
top-22, latent 2048, expert ffn 5120, shared expert 8192↔10240, router n_group=1 (**the
group-limit branch in `group_expert_select` never runs** — routing is one @mx.compile'd
sigmoid+argpartition chain, nemotron_h.py:313-344; no host round-trip, no `.item()`).

### Bytes read per token (bf16=2B/param, nvfp4 gs16=0.5625B/param)

| component | per layer | × | GB/token | ideal ms @819GB/s |
|---|---|---|---|---|
| mamba in_proj 8192×35072 bf16 | 574.6MB | 48 | 27.582 | 33.7 |
| mamba out_proj 16384×8192 bf16 | 268.4MB | 48 | 12.885 | 15.7 |
| MoE shared expert 2×8192×10240 bf16 | 335.5MB | 48 | 16.106 | 19.7 |
| MoE latent projs 2×8192×2048 bf16 | 67.1MB | 48 | 3.221 | 3.9 |
| MoE router gate 512×8192 bf16 (stays bf16) | 8.4MB | 48 | 0.403 | 0.5 |
| MoE routed experts 22×2×(5120×2048) nvfp4 | 259.5MB | 48 | 12.457 | 15.2 |
| attention q/k/v/o bf16 | 276.8MB | 12 | 3.322 | 4.1 |
| lm_head 131072×8192 bf16 | — | 1 | 2.147 | 2.6 |
| **pure weights** | | | **78.123** | **95.4** |
| SSM state r+w f32 (0.805) + conv state (0.011) + KV@6k (0.074) | | | 0.890 | 1.1 |
| **total traffic** | | | **79.013** | **96.5** |

Ceiling: 10.48 tok/s (weights only) / 10.36 tok/s (incl. state/cache). Non-expert weights
65.666GB incl. router gate; **P0-1 DQ8 target pool = 65.263GB (router gate excluded)**.

### NVIDIA's own quantization map (doctrine anchor)

From the conversion record (tasks/oqnvfp4_nemotron.md §12, converter-verified per-tensor
mix; T7 source currently unmounted — LEAD chore: quote and freeze the exact
`hf_quant_config` exclude list into this file when it is next mounted):
- **FP8-shipped by NVIDIA** → mamba `in_proj`/`out_proj` (all 48 layers). We upcast these
  to bf16 at conversion — the "bf16-carrying-FP8 upcast tax", 40.467GB/token of the 78.1.
- **Deliberately excluded (bf16) by NVIDIA** → latent projs, o_proj (and q/k/v), shared
  experts, router, embeddings, lm_head. Per doctrine (a): each of these classes gets its
  own kill-switch and isolated quality A/B — no blanket squash.

### Kernel count ≈ 1800 dispatched/token

48 mamba × ~15 (norm, in_proj, 3 splits, concat, conv1d, silu, cache slice-update,
compute_dt, ssm_kernel, gated-norm ~3, out_proj, residual) ≈ 720; 48 MoE × ~20 (norm, gate
mm, ~4 compiled-select, **5 ts-fold** (nvfp4_ts.py:57-59), latent1, fc1 gqmm, relu2 ~2,
fc2 gqmm, wsum ~3, latent2, shared ×3, adds 2) ≈ 960; 12 attn × ~9 ≈ 108; head+sample ≈ 10.
Measured encode 46ms ⇒ ~26µs/op host cost (M3 measured ~8.6µs bare dispatch + node ctor +
malloc churn; samples show 480 libsystem_malloc mentions).

### Reconciliation vs 130ms observed

GPU-must-run ≈ 96.5ms ideal; fat matmuls run ~96% of ceiling on this box (todo.md:884).
v2.2 RESOLVED: K1 measured the expert path in-stream at **28.48ms/token (1.87× ideal)**,
of which only 2.41ms is expert-kernel-specific (dense-equal-bytes control 1.71×) — the
rest is the diffuse in-stream overhead every op pays. GPU busy ≈ 105ms (dense ~70 +
experts 28.5 + state ~2 + margin); CPU busy 70ms, ~36-45ms overlapped → wall = GPU busy +
starvation ≈ 130ms. ✔ Attack surfaces: **bytes** (96ms floor → P0-1), **host
serialization + diffuse dispatch** (~25-34ms starvation + encode scale → P0-2/P0-3;
the deep diffuse floor belongs to the future mega-kernel campaign).

---

## 3. Microbench suite (offline, synthetic weights, no model load)

Scripts in `~/.claude/jobs/62f9cfe9/tmp/` (GLM scaffold precedent `probeA_gather_qmm.py` /
`probeB_linears.py`: warm-up eval, ≥12 reps, median). Rules (tasks/lessons.md): measure
deltas **in-stream** (`mx.depends`-chained) as well as standalone — savings surviving
pipelining are ~50-60% of naive; activations in **bf16** (live dtype); for expert probes
generate random uint32 codes / uint8 scales DIRECTLY (never `mx.quantize` a 10.7GB bf16
tensor).

**M0-dense — `ultra_probe_dense.py`** (decides P0-1 mode + staging)
Shapes × counts: in_proj 8192→35072 ×48, out_proj 16384→8192 ×48, shared_up 8192→10240
×48, shared_down 10240→8192 ×48, latent 8192→2048 & 2048→8192 ×48, q/o 8192→8192 ×24,
kv 8192→256 ×24, lm_head 8192→131072 ×1; router gate 8192→512 ×48 **reported separately**
(stays bf16, excluded from savings). Modes: bf16 `x@W.T` vs `mx.quantized_matmul` affine8
gs64 vs **mxfp8 gs32** (M3 ledger: mxfp8 beat affine8 273 vs 301µs at M=1) at M=1
(+ M=2,4 columns for M7/P2-2). Output per module class: µs, achieved GB/s, ratio
`bf16_time/q_time`, and the **weighted model-level token saving in ms**.
**GO gate (RC5, direction explicit): weighted projected token saving ≥ 25ms/token**, i.e.
per-shape `bf16_time / q_time ≥ 1.6` (equivalently `q_time ≤ 0.625 × bf16_time`) on the
two mamba classes that carry 40.5GB. Report per-class so staging (P0-1 A→D) can proceed
even if one class underperforms.
**MEASURED (v2.2): GO.** Aggregate **47.2ms/token ≥ 25** with **affine8-gs64** the
winning mode. Per-class: mamba 1.44-1.54× (below the 1.6 gloss — aggregate governs,
ruling ratified); **attn k/v 0.87-0.91× = LOSS → excluded from DQ8**; q/o pending
per-class confirmation before stage C. Live expectation ~24-28ms/token after in-stream
discount → DQ8-alone ≈ 9.4-9.8 tok/s.

**M1-experts — MEASURED (v2.1)** (`m1_kernel_bench.py`, isolated-eval harness, 200 iters,
M=1, fc1 [512,5120,2048] / fc2 [512,2048,5120], top-22, random indices; raw output in the
session task log):

| matrix | nvfp4 gs16 | affine4 gs64 | mxfp4 gs32 | bandwidth-ideal |
|---|---|---|---|---|
| fc1 (129.8MB) | 408.2µs | 404.7µs | 395.1µs | 158.5µs |
| fc2 (129.8MB) | 416.5µs | 423.6µs | 401.3µs | 158.5µs |
| layer total | **824.7µs** | 828.3µs | 796.4µs | **316.9µs** |

**Finding 1 — no mode gap: all three within ±4%.** The Super 0.58× gs16 tax does not
transfer to Ultra's fat latent-2048 shapes at M=1. Even mxfp4's lower scale traffic
(4.25 vs 4.5bpw) buys ~3% — scale-load cadence is NOT the bottleneck here. ⇒ requant to
any other 4-bit container buys nothing (P1-1 dead on speed, not just doctrine); mxfp4 and
5-bit container columns are closed.
**Finding 2 — absolute efficiency looks poor IN ISOLATION: ~38% of peak** (318/311 GB/s
achieved vs 819). Taken at face value that is experts ≈ 39.6ms/token vs 15.2 ideal — a
~24ms/token prize. BUT the harness evals per op; isolated timings are known to inflate vs
in-stream (lessons.md: only ~50-60% of naive deltas survive pipelining), and this same
trap produced the retracted "MoE at 60% bandwidth" claim in the GLM campaign.
**⇒ K1 (below) must attribute the expert path IN-STREAM before any kernel is written.**
Follow-up-run spec (implementer): (i) production-identical call form — x
`expand_dims(-2,-3)`, idx [1,1,22]; **batch legs use rhs_indices shape (M,1,TOP)** (the
(M,TOP) form errored — broadcast contract); (ii) sorted AND unsorted matching live
`do_sort = indices.size >= 64` (false at batch-1 top-22 — switch_layers.py:220-231);
(iii) the in-stream ×48 `mx.depends`-chained variant WITH realistic neighbors (latent
proj in, weighted-sum out) — this IS K1; (iv) M={1,2,4} columns retained only for the
P2-2 economics file.

**M2-ssm — `ultra_probe_ssm.py`**
(a) `ssm_update_kernel` [1,1,256,64], state [1,256,64,128] f32: µs standalone AND ×48
`mx.depends`-chained (ideal ~20µs: 16.8MB r+w). (b) decode conv chain (concat, conv1d k4,
silu, cache slice, 3-way splits) op-by-op. (c) graph-build-only `mx.fast.metal_kernel`
invocation cost (1000 builds, no eval; sample2 says ~45µs/call). (d) v2 additions (codex):
first-decode-token/state-None fallback path, `left_padding`/`lengths` mask variants —
`ssm_update` switches to `ssm_attn` when state is None or L>1 (ssm.py:230-249).

**M3-route — `ultra_probe_route.py`**
Full MoE non-expert chain at [1,1,8192]: gate matmul + `group_expert_select` (n_group=1) +
ts-fold + weighted-sum epilogue; then fused variants (2b-lite precombined ts; 2b-topk
kernel). v2 (codex): drive with a **recorded/realistic gate-logit corpus** (sigmoid-space
distribution) plus tie/near-tie stress vectors, not only mx.random — argpartition edge
cases are the risk. Parity bar: exact index-set equality + final-output rtol.

**M4-dispatch — `ultra_probe_dispatch.py`**
2000 chained tiny binary ops on [1,8192] bf16 → µs/op host floor (M3: ~8.6µs). v2 (codex):
add one `mx.fast.metal_kernel` no-op, one skinny `mx.quantized_matmul`, one skinny
`mx.gather_qmm` variant so the fusion ROI multiplier isn't derived from binary ops alone.

**M5-overlap — falsifier only (codex).**
~1800-op synthetic chain (~40MB reads/op), one `async_eval` vs 4 chunked: time-to-return
vs time-to-complete. A negative result KILLS P0-3; a positive one still requires the live
A/B (real decode mixes GEMV/gather_qmm/custom kernels/cache updates the synthetic misses).

**M6-composed (NEW, codex gap) — `ultra_probe_composed.py`**
Small synthetic nemotron_h model (hidden 256, layers M/*/E ×2, 8 experts, ts sidecars,
nvfp4 switch_mlp) composing **DQ8 + 2b-lite + 2b-topk + 2a fused conv + chunked eval
together**: 200-token greedy parity vs stock, engagement counters == expected on every
step, then each kill-switch flipped one at a time. Catches interaction bugs and dead
fast-paths that component probes miss. This is also the repo unit-test payload
(tests/test_nemotron_ultra_decode.py) — runs without the 334GB model.

**M7-mamba-seq (NEW, codex gap; feeds P2-2 only) — `ultra_probe_mamba_seq.py`**
Mamba layer at L∈{1,2,4}: L=1 uses `ssm_update_kernel`, L>1 switches to `ssm_attn`
(ssm.py:230-249) — measure the L>1 penalty ×48 layers. Any spec-decode verify pays this;
dense/expert M-scaling alone (M0/M1 columns) is insufficient evidence.

---

## 4. Ranked levers

Baseline to mutate: wall 130 = GPU busy ~105 (experts 28.5 MEASURED in-stream (K1),
dense ~70, state ~2) + starvation ~25; CPU busy 70 partially hidden.

### P0-K — expert gather_qmm workstream: PARKED → P2 (v2.2, K1 measured)

K1 ran under golden env (12 reps, ×48 chained, realistic neighbors): unsorted-production
593.4µs/layer (1.87× ideal, spread 21.7%), sorted 520.4µs (1.64×, 5.6%), **dense-equal-
bytes control 543.3µs (1.71×, 0.8%)**. Expert-specific excess = 50.1µs/layer =
**2.41ms/token** — the gather_qmm kernel is ~as bandwidth-efficient as a plain dense
read; the rest of the 1.87× is the diffuse in-stream overhead every op pays. The K3
fused expert-MLP kernel's addressable pool is therefore ~2-4ms/token (expert excess +
one launch + the [22,5120] roundtrip) — it cannot rival DQ8. **Parked to P2.** The K2
source-study and K3 kernel design specs live in the v2.1 changelog entry above and in
git history of this file — reusable by the future cross-model mega-kernel campaign
(where the 1.71× diffuse floor itself is the target, alongside GLM foray §10b's ~32ms).
Cross-model porting note stands: re-run K1-style attribution per model before assuming
transfer. What K1 DID yield for this campaign: the sorted-vs-unsorted delta →
**P0-2d below**.

### P0-1 — Load-time DQ8 of the dense bf16 pool (probe/emergency path; checkpoint is the product)

Target pool 65.263GB (router gate excluded per RC5). Post-quant totals incl. experts+gate+
state: **47.401GB (mxfp8 gs32) / 48.421GB (affine8 gs64)** → ideal 57.9/59.1ms → ceiling
~17 tok/s. Expected wall after full staging: GPU busy ~64-70ms + starvation ~25 →
**~90-95ms → 10.5-11 tok/s** naive; **v2.2 measured-M0 expectation: ~102-106ms →
9.4-9.8 tok/s from DQ8 alone** (mamba class came in at 1.44-1.54×, and only ~50-60% of
isolated deltas survive in-stream); with P0-2 (incl. 2d) + P0-3 → **~78-85ms →
11.8-12.8 tok/s**. The former conditional 14-16 leg is DEAD (K1: no expert-kernel
headroom; P0-K parked).

**Staged by module class (codex gap + doctrine (a)), each with its own switch + isolated
LEAD A/B + quality smoke, in this order:**

| stage | classes | GB saved (mxfp8) | switch | quality prior |
|---|---|---|---|---|
| A | mamba in/out_proj (96 mods) | 40.467→20.866: **−19.6** | `OMLX_ULTRA_DQ8_MAMBA` | NVIDIA shipped these FP8 — we're removing our own upcast tax |
| B | shared experts + latent projs (192 mods) | 19.327→9.966: **−9.4** | `OMLX_ULTRA_DQ8_MOEDENSE` | NVIDIA-excluded → isolated A/B mandatory |
| C | attention **q_proj/o_proj ONLY** (24 mods; k/v EXCLUDED — M0 measured 0.87-0.91× = loss on the skinny 8192→256 shape) | 3.221→1.661: **−1.56** | `OMLX_ULTRA_DQ8_ATTN` | NVIDIA-excluded; build only after q/o per-class M0 numbers confirm ≥1.0× |
| D | lm_head (1 mod) — **LAST, isolated ablation** | 2.147→1.107: **−1.0** | `OMLX_ULTRA_DQ8_LMHEAD` | touches final logits directly (codex) |

- Mode: **LOCKED affine8-gs64 (M0 measured — mxfp8 lost at these shapes)**. Router gate,
  embeddings, norms, conv1d, `switch_mlp.*`, `gate`, and attn k/v NEVER quantized.
  (Table GB figures were computed at mxfp8 bpw; affine8 saves ~3% less — aggregate
  47.2ms/token measured is the authoritative projection input.)
- **Hook contract (RC1)**: extend `apply_post_load_transforms`
  (omlx/utils/model_loading.py:513) so env-gated transforms run BEFORE the
  `model_settings is None` early return (today it returns immediately and only handles
  IndexCache). Engagement condition: any `OMLX_ULTRA_DQ8_*` env set AND instance
  fingerprint matches — read from the loaded model object (`model.model_type ==
  "nemotron_h"`, `model.args.hidden_size == 8192`, switch_mlp is QuantizedSwitchLinear
  mode nvfp4 with `fc1_ts` present), no config file re-read. `batched.py:283` already
  calls the hook after `mlx_lm.load`.
- **Transient budget (RC2)**: naive whole-model `nn.quantize` holds 334GB resident +
  ~33.7-34.7GB new buffers + allocator scratch before old bf16 frees. Mitigation:
  module-by-module loop — quantize one Linear (`QuantizedLinear.from_linear`), `mx.eval`,
  install via `update_modules`, drop the old weight reference, `mx.clear_cache()` every
  ~16 modules → peak extra ≈ largest module pair (574MB bf16 + ~300MB q8) < 1GB. LEAD
  still confirms pool headroom before the restart (other models loaded in-process).
  Final resident drops ~31GB (bonus). Load cost +1-3 min per restart — acceptable for the
  probe path only (see P1-3).
- Engagement (RC8): per-stage counters at load `[ULTRA-DQ8] mamba expected=96 actual=96;
  moedense expected=192 actual=192; ...` — **hard-fail on mismatch** (never half-quantize)
  + first-decode-step census line.
- Offline tests: M6 composed model — expected module set swapped, forward parity ≤ q8
  tolerance, `_TsFoldMoE` still the mixer class + `fc1_ts` present, second stock model
  in-process untouched; per-stage switch flips.
- LEAD legs (per stage): single chained restart command (OPS contract), engagement grep,
  decode A/B short/6k/20k, gsm8k 200-sample smoke; full regate after stage D; Super-120B
  regression probe (engagement lines ABSENT); memory-accounting sanity (resident −31GB vs
  engine_pool admission — 5a26eb1 touched that code).

### P0-2 — Op-count reduction ≈1800 → ≈1050/token (encode 46→~27ms, python 22→~16ms)

All in `omlx/patches/nemotron_ultra_decode/` (per-instance swaps only; precedents:
glm_moe_dsa/decode_kernels.py `_log_once` discipline, nvfp4_ts.py `__class__` swap).
Umbrella `OMLX_ULTRA_DISABLE_DECODE_OPT=1` + sub-switches. All decode-gated with
per-reason fallback census (RC3/RC8).
- **2a mamba glue fusion** (~340 ops): one `mx.fast.metal_kernel` for concat(conv_state,x)
  → depthwise conv k4 → silu → state shift → 3-way split emit, replacing ~7 ops/layer in
  `NemotronHMamba2Mixer._conv` + splits (nemotron_h.py:134-229).
  **Gate (RC3)**: B==1 AND `cache.lengths is None` AND (`cache.left_padding is None` OR
  all-zero — the `[0]` singleton is the common batch-aware state (generate.py:838-861,
  cache.py:691-699); engaging on all-zero is valid and ASSERTED) AND mask is None/all-true
  AND dtype bf16. Census: `fused_conv=48/48` + fallback reasons {batch, lengths, leftpad,
  mask, dtype}. Switch `OMLX_ULTRA_DISABLE_FUSED_CONV=1`. Parity: bit-level vs stock chain,
  500 steps, + batch-2/ragged fallback test (RC6).
- **2b-lite ts-precombine** (low risk, ~4 ops × 48): precompute `combined_ts =
  fc1_ts**2 * fc2_ts` [512] f32 at patch time; fold = ONE gather+mul. Our per-instance
  `__call__` replaces `_moe_call` wholesale on Ultra instances only (never touch the
  shared `_TsFoldMoE` class). Switch `OMLX_ULTRA_DISABLE_TSPRE=1`. Test: numeric identity
  to stock fold.
- **2b-topk fused top-22** (high risk, ~4-6 ops × 48): fused sigmoid+bias+top22(values+
  indices)+normalize+scale kernel over [1,512] (M3 fused-topk precedent, bit-identical
  bar). **Parity bar (RC4): exact index-set equality on a realistic gate-logit corpus +
  tie/near-tie stress + final MoE output parity + fallback counters**; live A/B only after
  all four pass. Switch `OMLX_ULTRA_DISABLE_FUSED_ROUTE=1`.
- **2c qkv pack** (~24 ops): one 8192→8704 matmul + views. **Gated behind M4 + a
  qkv-specific in-stream probe** (codex: payoff small). v2.2 NOTE: with k/v now excluded
  from DQ8 and q/o quantized, a packed projection would need mixed precision in one
  matmul — NOT possible; 2c is only viable as all-bf16 (pre-stage-C) or needs k/v folded
  back in. Deprioritize until stage C settles. Switch `OMLX_ULTRA_DISABLE_QKV_PACK=1`.
  Parity: exact.
- **2d pre-sorted expert indices (NEW v2.2, from K1's measured sorted-vs-unsorted
  delta: 73µs/layer = 3.50ms/token gross in-stream; ~3 extra tiny dispatches/layer ≈
  1ms → net ~2.5ms/token).** Mechanism: MLX's `SwitchMLP.__call__` engages its sort path
  only at `indices.size >= 64` (switch_layers.py:222) — batch-1 top-22 always runs
  UNSORTED gather_qmm; sorted `rhs_indices` take the kernel's fast path.
  Implementation (inside our per-instance `_moe_call` replacement, which 2b-lite already
  owns): after ts-fold, `order = mx.argsort(inds, axis=-1)`; permute **(inds, scores)
  JOINTLY** via `take_along_axis`; then BYPASS `switch_mlp.__call__` and call
  `smlp.fc1(x, inds_sorted, sorted_indices=True)` → relu² → `smlp.fc2(...,
  sorted_indices=True)` (1:1 port of the SwitchMLP body minus its sort machinery,
  keeping the `expand_dims(x, (-2,-3))` semantics), then the weighted-sum epilogue with
  the PERMUTED scores.
  **Correctness ruling (checked nemotron_h.py:385-424 + switch_layers.py:176-236):**
  the epilogue `(y * scores[..., None]).sum(axis=-2)` pairs rows positionally — joint
  permutation of (inds, scores) preserves every pair and the sum is permutation-
  invariant mathematically; ts-fold gathers `combined_ts[inds]` positionally so folding
  BEFORE the sort and permuting the folded scores is exact; top-22 indices are distinct
  (argpartition over distinct positions) so `sorted_indices=True`'s precondition holds.
  Output is allclose-but-NOT-bit-identical (fp accumulation order changes in the sum) —
  parity bar: exact expert-SET equality + output allclose (bf16-appropriate rtol),
  documented as such.
  Gate: engage only when `indices.size < 64` (stock path already sorts above that);
  batch fallback counter. Switch `OMLX_ULTRA_DISABLE_SORTED_ROUTES=1`; census
  `sorted_routes=48/48`. Offline test: expert-set equality + allclose + batch-3
  fallback. LEAD leg: in-stream A/B via the M3-route probe extension, then live A/B.
  Linkage: when 2b-topk's fused kernel lands it emits its 22 indices ALREADY SORTED
  in-register (free), superseding 2d's 3 extra dispatches — fold 2d's gate into it.

### P0-3 — Chunked intra-token `async_eval` (starvation fix; no wheel)

Today ZERO kernels are encoded during the ~22ms Python build — the GPU drains and starves
(~20-25ms/token). Wrap THIS instance's `__call__` to `mx.async_eval(h)` every K layers
(default 27) during decode (L==1) only; MLX multi-output semantics materialize sibling
cache/state arrays with h — scheduling only, values identical.
- Expected: ~10-20ms/token recovered (M5 is the falsifier; positive M5 ⇒ still needs live
  A/B). Kill-switch `OMLX_ULTRA_EVAL_CHUNK=0` (default off until LEAD A/B), engagement
  `[ULTRA-CHUNK] eval_chunk=27 chunks=4/4` per-step counter.
- **Test battery (codex + RC6)**: token-identical 200-step greedy vs unchunked (M6);
  abort/retry mid-decode; cache-corruption recovery path; prefix-cache store→reuse cycle
  (omlx materializes cache arrays on the owner thread before background store,
  scheduler.py:8949-9026 — chunking changes WHEN arrays become concrete; must be
  value-identical); SSM/KV state identity after 1000 greedy steps; request-A-then-shorter-B.
- LEAD leg: A/B K ∈ {0, 54, 27, 12}, one restart each.

### P1 (1-2 weeks; each gated on a P0 probe)

**P1-3 → PROMOTED (RC7): offline DQ8 checkpoint = the production path.**
One-shot tool (pattern: oqnvfp4_nemotron_convert.py, far simpler): read the oQNVFP4
checkpoint, `mx.quantize` the stage-A..D tensor classes to the M0-winning mode, write
config `quantization` entries per module (upstream loader then quantizes-before-
load_weights — no boot-time pass, no transient peak, .venv .../mlx_lm/utils.py:359-380).
Emit `Nemotron-3-Ultra-oQNVFP4-dq8` (~303GB). **Disk gate**: internal margin was ~298GB
BEFORE this output — likely does NOT fit alongside the 333GB original; options (LEAD
decision): emit to T7 then swap, or delete original after full regate on the new build.
Load-time DQ8 (P0-1) remains the fast probe and the emergency path if disk is blocked.

**P1-2 mamba decode megakernel v2** (only if M2 shows remaining per-layer glue + ssm
python-invocation ≥ ~6ms/token after 2a): fold conv+silu+splits+compute_dt+SSM-update+
state-write into 1-2 kernels (kills the 48×~45µs metal_kernel call setup too). Tests
(codex): state dtype preservation, left_padding/lengths fallback, first-token state-None
fallback, 1000-step recurrent drift.

**P1-1 expert requant — REMOVED (v2.1).** M1 measured nvfp4/affine4/mxfp4 within ±4% at
Ultra shapes: requant to another 4-bit container buys nothing and is lossy-on-lossy.
Measured equal, rejected. (The kernel-efficiency prize lives in P0-K instead.)

### P2 (explicit lead sign-off)

**P2-1 banked MAX_ACTIVE_TASKS cap wheel** (M3 campaign asset). Attacks the same
serialization as P0-3 properly. Risks: process-global (GLM native kernels pinned to
nanobind 2.12.0 ABI — re-verify), `uv run` silently reverts patched wheels (lessons.md).
Only if P0-3 pays but leaves ≥10ms on the table. Also the fallback home for P0-K if a
metal_kernel can't express the fix (wheel-level kernel patch).

**P2-2 speculative decode economics re-check (probe only; NO build; NO MTP).**
Ultra is 84% dense bytes — the M3 expert-scaling wall doesn't transfer numerically. The
discriminator = M0/M1 M∈{1,2,4} columns **+ M7 mamba L>1 probe (codex: the L>1 path
switches ssm_update_kernel→ssm_attn; recurrent state is the hard part — per-step state
checkpointing ~0.4GB×K + rollback plumbing)**. Note: omlx `_is_mtp_compatible` doesn't
admit nemotron_h (model_loading.py:386-397) and stock sanitize drops `mtp.*` — consistent
with doctrine (d): no MTP work. Drafter question (Nano-30B tokenizer identity) only if the
lead ever reopens.

---

## 5. Risk register

| risk | exposure | mitigation |
|---|---|---|
| ts-fold broken by DQ8 | predicate touching `switch_mlp`/`gate` mis-scales experts or breaks strict load | predicate excludes them; M6 asserts `_TsFoldMoE` engaged + `fc1_ts` present + parity; `nn.quantize` replaces leaves via `update_modules`, parent class swap survives (codex-verified, quantized.py:22-95) |
| DQ8 transient OOM at load | +34GB naive peak on a box hosting other models | module-by-module quantize/release loop (<1GB extra); LEAD headroom check; production = offline checkpoint (P1-3) with zero transient |
| ts double-fold / stale fold | 2b-lite coexisting with `_moe_call` gathers | per-instance `__call__` REPLACES `_moe_call` wholesale; identity test |
| routing instability | 2b-topk vs argpartition ties | RC4 bar: exact index-set on realistic corpus + tie stress + output parity + fallback counters before any live A/B |
| batch/ragged correctness | 2a/2b/2c/chunking assume batch-1 decode | RC3 gates incl. `left_padding` all-zero assert; per-reason fallback census; batch-2/ragged offline tests; RC6 cross-request battery (A-then-shorter-B, prefix hit after store, extraction with state arrays, abort/retry) |
| partially materialized cache/state | P0-3 changes when arrays become concrete; later-layer error leaves partial state | abort/retry + cache-corruption recovery + store→reuse tests in the battery; chunking value-identical by construction (M6 verifies) |
| quality regression from 8-bit | stages B/C/D quantize NVIDIA-excluded classes | staged isolated A/Bs, lm_head last (codex); per-stage switches allow shipping A alone; full regate after D |
| other in-process Nemotrons (Super, Nano) | shared nemotron_h module, shared `_TsFoldMoE`, shared compiled `group_expert_select` | per-instance swaps only + hidden_size==8192 fingerprint; never rebind module-level fns; LEAD Super probe expects ABSENT engagement lines |
| prefix-cache vs DQ8 | logits differ from any pre-quant expectations | load-time change ⇒ per-process cache rebuilt; no mixed state. SF-1 changes index publish timing — its own battery below |
| dead-patch shipping | fp16-gate-vs-bf16-live class of bug | RC8: expected-vs-actual per-layer counters + per-reason fallback counts, live-grepped by LEAD before any benching |
| kernel ABI / wheel traps | P2-1 / P0-K wheel fallback | nanobind 2.12.0 ABI recheck; `uv run` revert trap documented workaround; separate venv for trials |

## 6. Non-goals (do not re-litigate without new data)

- `mx.compile` of the decode step — 3× dead (GLM foray §10b 1.06×; M3 flat; GLM flat).
- MLX task-cap raise via env — compile-time constant; wheel route only via P2-1.
- Buffer env resizing — 500→2000 flat on M3; golden 4000/4000 stands (sole exception: one
  lead-run re-sweep AFTER P0-2 lands; 10 minutes, not a workstream).
- int8/mxfp8 KV or latent cache — settled (capacity feature, not speed).
- Speculative decode BUILD at batch-1; ANY MTP work (doctrine (d): MTP dropped at
  conversion, fs5 drafter disabled; `_is_mtp_compatible` excludes nemotron_h anyway).
- **Dense 6-bit / mixed-precision push — REJECTED by codex review** (~5ms for outsized
  quality risk); reopen only after DQ8 ablations show large margin AND P0-2/P0-3 live
  deltas are known.
- nvfp4→mxfp4 or →affine4 expert CONVERSIONS — **measured speed-equal (±4%) at Ultra
  shapes in M1 (v2.1), on top of being lossy-on-lossy. Rejected outright**; the former
  P1-1 is deleted.
- affine5-gs16 exact container — verified unsupported in MLX 0.31.2 (gs16 rejected;
  gs32 can't be exact over per-16 NVFP4 scales). Dropped; and M1's no-mode-gap result
  makes any container swap moot regardless.
- Quantizing router gate or embeddings; TTFT/prefill throughput work; batch>1 serving
  optimization; hand-rewrites of MLX dense matmul/qmv (they run ~96% of ceiling).

## 7. SERVING-FIX workstream (separate deliverable, same pipeline — NOT a speed lever)

**SF-1 prefix-cache commit lag.** Today the token→block index entry becomes visible only
when the store worker completes `block_aware_cache.store_cache(...)` (worker body
scheduler.py:2000-2067; submitted at :9068-9070 after the inference-thread
boundary/collect/dispatch phases). A follow-up request arriving during the store window
misses the cache unless the freshness bridge catches it — and the bridge only engages at
`_CACHE_FRESHNESS_WAIT_MIN_PROMPT_TOKENS = 8192` with `_CACHE_FRESHNESS_WAIT_TIMEOUT_S =
4.0` (scheduler.py:5729-5732), which under-covers big-model store times.
Fix (two parts):
1. Publish the index entry **on the inference thread immediately after
   `mx.eval(*pre_eval_arrays)`** (the `store_cache_main_dispatch` phase, ~scheduler.py:9026)
   — the arrays are concrete there; only host memcpy + SSD persist remain on the worker.
   Requires: entry states "hot/in-memory" until the worker confirms persist; the worker's
   failure path RETRACTS the entry (no index pointing at bytes that never landed);
   lookup path already distinguishes hot cache (`hot_cache_write_back` machinery, :9030).
2. Widen the freshness bridge: min-prompt 8192 → model-scaled (e.g., scale with measured
   store duration / prefill rate for the loaded model), timeout 4s → cover the real store
   time (EWMA of recent store durations + margin).
Kill-switch `OMLX_DISABLE_EARLY_INDEX_PUBLISH=1`; engagement counter for early-published
vs worker-published entries. Tests: full RC6 battery (request A then shorter B DURING A's
store; prefix hit immediately after store dispatch; abort mid-store retracts entry;
extraction with SSM state arrays; batch-2).

**SF-2 prefill-throttle mispredictor.** `_predicted_chunk_transient`
(scheduler.py:3258-3292) takes the MAX of (last measured per-token delta, EWMA, kv-aware
static estimate) × 1.3 safety. A one-off allocation spike (cache materialization, load
event, allocator churn) poisons `last_delta_bytes`/EWMA and the MAX keeps predicting huge
transients → chunk sizes collapse → prefill crawls. Fix: 3-line clamp of BOTH
tracker-derived signals to `K × static_estimate` (K≈8, env `OMLX_TRANSIENT_CLAMP_K`,
0 = off) inside the function; static path and `_PREFILL_TRANSIENT_SAFETY`/
`_PREFILL_ABORT_MARGIN` untouched (the Metal async OOM SIGABRT is the failure mode the
margin guards — clamp must stay generous). INFO log when the clamp engages (with clamped
vs raw values). Tests: unit test on the predictor with a poisoned tracker; LEAD leg:
long-context prefill A/B confirming no OOM aborts and restored chunk sizes.

## 8. Execution order & telemetry

1. ~~Day 0 probes~~ DONE (M0 measured → P0-1 GO affine8; K1 measured → P0-K parked;
   SF-1/SF-2 implemented + reviewed).
2. **Now**: (a) P0-1 stage A build (loadquant patch, affine8-gs64, module-by-module
   transient loop, counters) + offline M6 tests → LEAD A/B + gsm8k smoke; (b) P0-2b-lite
   ts-precombine + **2d sorted routes** (both live in the same `_moe_call` replacement —
   one patch package) + offline parity; (c) SF required tests T1/T2, then SF live legs.
3. P0-1 stages B → C(q/o-only, pending per-class numbers) → D, each behind its own A/B +
   smoke; full regate after D. Decide P1-3 checkpoint emission (disk plan) as soon as
   A+B prove out.
4. 2a behind M2; 2b-topk behind M3 parity bars (emits sorted indices natively,
   superseding 2d's extra ops); 2c deprioritized (mixed-precision pack conflict, see
   note). P0-3 behind M5 falsifier, then LEAD chunk sweep.
5. P1/P2 strictly behind their gates (P1-1 deleted — measured equal; P0-K parked — K1
   measured no kernel headroom). Every live A/B logs tok/s @ short/6k/20k + engagement
   census (expected-vs-actual + fallback reasons) and lands in the todo.md ledger with
   kill-switch states spelled out.
