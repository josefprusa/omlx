# OVERLAP LEVERS #1 (mx.compile) + #2 (MLX task-cap) — crew plan 2026-07-04

## Measured context (do not re-derive)
- Decode is host-serialized: M3 37ms/token = 52% GPU-wait + 18% Metal encode + ~29% python/graph-ctor.
  GLM 48ms/token = 69% GPU-wait + 16.5% encode + ~15% python. Host work does NOT overlap GPU.
- Root cause verified in MLX v0.31.2 transforms.cpp: `static constexpr int MAX_ACTIVE_TASKS = 10;`
  eval_impl blocks ANY caller (async_eval included) past it: gpu::finalize(open streams) + wait_for_one().
  Second branch: blocks while get_active_memory() > get_memory_limit() && tasks>0.
- PRE-FLIGHT RESULT (lead, today): `MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=200000` →
  M3 26.3 → 28.43 tok/s (+8%) env-only. The 40MB default MB cap forced early commits (why OPS=2000
  alone was flat). THIS ENV IS LIVE ON THE SERVER NOW — unvalidated on GLM.
- Baselines today (probe: `cd ~/.claude/jobs/62f9cfe9/tmp && M3_MODEL=<id> python3 m3_probe.py 8 500 <nonce>`):
  M3-fs5 26.3 (OPS=500) / 28.43 (bigbuf); GLM-Alis 20.8 (OPS=500). MTP ON for fs5 (production).
- Targets: M3 wall → ~GPU ≈ 20-22ms (45-50 tok/s); GLM → ~33ms (~30 tok/s).

## LEVER #2 — owner: overlap-builder (owns the SERVER exclusively)
1. Env sweep (quick, ~30min): grid OPS {1000,4000,16000} × MB {1000, 50000, 200000} on M3; then best
   combo on GLM (327GB — watch memory enforcer log for ceiling pressure; bigger buffers retain
   transients longer). Track TTFT too (bigger buffers delay first kernel). Pick production combo.
2. MLX source patch: clone ml-explore/mlx at tag v0.31.2. Change MAX_ACTIVE_TASKS to read env
   `MLX_MAX_ACTIVE_TASKS` (default 10 = stock). FIRST read mlx/scheduler.{h,cpp} and report what
   n_active_tasks actually counts + where the backpressure check sits relative to command-buffer
   commits (hand to codex-reviewer for cross-check). Build wheel (MLX_BUILD_METAL=ON; needs xcrun
   metal toolchain — verify first). Install into repo .venv.
3. ABI LANDMINE (from memory, real): omlx GLM native ext (_ext.so/.metallib/.dylib, git-ignored)
   was built against the stock wheel. After wheel swap REBUILD:
   `rm -rf build; CMAKE_ARGS="-DPython_EXECUTABLE=$(pwd)/.venv/bin/python" OMLX_WITH_CUSTOM_KERNEL=1
   .venv/bin/python setup.py build_ext --inplace --with-custom-kernel` (nanobind pinned 2.12.0, ABI v19).
   Then verify [GLM-DKO] engagement lines appear live — "imports fine" ≠ "kernels engage" (house lesson).
4. A/B ladder: MLX_MAX_ACTIVE_TASKS {10, 32, 64, 128} × best env combo. Bench M3 short + one 16k run,
   GLM short. /usr/bin/sample the winner (expect cond_wait share to collapse and wall → GPU time).
5. GATES before declaring: (a) temp0 token-identical outputs stock vs patched (same prompt/nonce),
   (b) M3 quick NIAH 12/12 at 32k (script exists: jobs tmp/niah_bench.py), (c) all three models
   healthy, (d) no memory-enforcer ceiling breaches over a 10-min soak, (e) kill switch documented:
   `uv pip install mlx==0.31.2` + stock env line + ext rebuild.
6. If raising the cap exposes the memory-limit branch (get_active_memory spikes), cap lower or
   coordinate with lead — do NOT raise mx memory limit to compensate without asking.

## LEVER #1 — owner: compile-builder (OFFLINE ONLY until Phase B go)
Phase A (start now, no server access): minimal spike scripts (scratchpad), each PASS/FAIL + microbench:
  a. mx.compile wrapping quantized_matmul + gather_qmm (affine gs64 AND nvfp4/mxfp8 modes).
  b. mx.compile wrapping an mx.fast.metal_kernel custom primitive — use the real fused_index kernels
     from omlx/patches/mlx_vlm_minimax_m3_compat/vendor/.../fused_index.py.
  c. Cache-state pattern: slice_update on captured buffers via compile inputs=/outputs= state,
     offset carried as mx.array (NO python-int branches inside the step). Prove correctness over
     300 simulated steps incl. a 256-boundary buffer growth (recompile expected — measure retrace cost).
  d. Retrace cost at realistic scale: dummy 60-layer graph, measure per-bucket recompile ms.
  e. Interplay: compiled fn should shrink op count → fewer buffers → less #2 pressure. Quantify on the dummy.
Phase B (ONLY after lead relays #2 results): if #2 delivers wall≈GPU at batch-1, #1's batch-1 win is ~0 —
  rescope with lead (batch≥2 throughput / CPU burn / skip). If #2 partial: full integration behind
  OMLX_M3_COMPILE=1 (default OFF), census counters bypassed inside compiled path (they're python
  side-effects — keep census on fallback path), bucket recompile at cache-growth boundaries,
  gates = argmax-agreement 200 tokens temp0 vs uncompiled + probe benches + NIAH 12/12 @32k.

## codex-reviewer (codex exec, read-only)
- Task 1 (now): read mlx v0.31.2 mlx/scheduler.{h,cpp} + transforms.cpp; report n_active_tasks
  semantics, backpressure placement, and risks of raising the cap (memory growth, completion-handler
  contention, relevance of omlx issues #300/#888 M4 'completeMemory prepare count underflow').
- Task 2 (on request): review overlap-builder's MLX diff + compile-builder's spike conclusions.

## OPS CONTRACT (non-negotiable, from campaign)
- Server work = ONE chained background command that ends with server relaunch + health probe.
  Never leave the server down while idling. Only overlap-builder touches the server; bench windows
  announced to lead. Standard launch:
  `env OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=500 uv run omlx serve --log-level info 2>&1 | tee <log>`
  in tmux session `omlx` window 4 (`omlx:4`), send-keys pattern; PID from log line "Started server process".
- Sample profiler: `/usr/bin/sample <pid> 20 1 -file out.txt` during an active probe decode (no sudo).
- NO git commits/pushes. Keep omlx changes minimal + env-gated. MLX clone lives outside the repo.
- Report to lead via SendMessage at each numbered step's completion; short numbers-first messages.
- Token discipline: measure, don't philosophize; reuse existing probe/bench scripts.

## STATUS LOG
- [x] 2026-07-04 eve — LEVER#1 PHASE A: 5/5 spikes PASS (compile-builder). Quant ops + custom metal
  kernels + stateful cache all compile bit-identical; 60-layer dummy: 4931→3007 ops (-39%), retrace
  43us/token amortized, census-on-fallback rule empirically confirmed. FEASIBLE. Artifacts:
  tasks/compile_spikes/ (PHASE_A_SUMMARY.md). Phase B holding on #2 gate; refactor inventory
  (PHASE_B_INVENTORY.md) being prepped meanwhile.
- [x] 2026-07-04 eve — LEVER#2: scheduler semantics pinned (lead + builder independently agree):
  task = committed cbuf; gate inside eval_impl tape loop after gpu::eval; two branches (count>10,
  memory). Patch written at ~/mlx-src (env MLX_MAX_ACTIVE_TASKS, default 10 = stock). Packaging
  insight: patch lives in mlx-metal stage-2 wheel → nanobind mlx.core ABI untouched by construction.
  Step 1.5 added: profile under winning env BEFORE building wheel. Env sweep running.
- [x] 2026-07-04 eve — PHASE_B_INVENTORY.md done (compile-builder): ONE root scalar (cache.offset,
  python int) fans to ~13 graph-entry sites incl. the fused_topk total_len bake (fix #1); 16+2 census
  sites all env-gated (dead-code under compile in prod, keep fallback); only 3 real bucket axes
  (cap-growth / sparse-crossover / L); seam = dedicated compiled decode_step(h, offset_arr) over the
  60-layer stack with per-layer {keys,values,index_keys} as compile state, host wrapper picks regime
  (do NOT compile the branchy MiniMaxAttention dispatcher in place). Largest mechanical change:
  KVCache buffers-as-state + slice_update writes (Spike-C pattern on the real cache).
- [x] 2026-07-04 eve — GOLDEN ENV FOUND: OPS=4000/MB=4000 (user-driven refinement). M3 decode
  saturates at MB=4000 (28.30 tok/s, full +7.6%); GLM 16k prefill peak 365GB = 80GB under soft
  (largest MB passing the >=60GB rule; 8000 gives 49GB, 50000 BREACHED at 479GB). MB accounting:
  4GB ~ one M3 layer's weights (commit-per-layer). GLM env win +10.6% (20.8->23.1). 58k-depth
  validation added to final gates. uv-run sync trap caught (patched wheel silently reverted;
  fix = --no-sync; logged in lessons.md). Wheel built+ABI-safe; cap A/B still gated on step-1.5 sample.
- [x] 2026-07-04 night — CAP A/B DEFINITIVE (patched binary, patch-live proven): cap 10→64 kills the
  backpressure wait ENTIRELY (8550→0 samples) yet tok/s FLAT (28.27→28.05 short; 16k WORSE) and GLM
  cached-sha byte-identical across stock/cap10/cap64. VERDICT: backpressure wait was overlapped/off
  critical path; real wall = main-thread python graph-build (~95% busy) + encode ~4.6ms + GPU.
  LEVER #2b DEAD as throughput lever. DECISIONS: ship env lever (golden 4000/4000, works on stock);
  shelve wheel at ~/mlx-src + SHELF.md (insurance for batch pile-up regimes); production posture =
  STOCK mlx + golden env (plain uv run re-sync = the restore mechanism). Gate(a) PASSED.
  PHASE B GO issued to compile-builder: full mx.compile integration, projected M3 35→~22-25ms
  (≈40-46 tok/s); OMLX_M3_COMPILE=1 default OFF; L=1 first; MTP path stays eager.
- [x] 2026-07-04 night — LEVER #2 CLOSED, SHIPPED: production = STOCK mlx 0.31.2 + golden env
  (OPS=4000/MB=4000). M3 26.3→28.4 (+8%), GLM 20.8→23.0 (+10.6%), Nemotron 137 tok/s. GLM prefill
  peak PLATEAUS 390GB at 37k AND 59k (chunked prefill bounds working set) → MB=4000 safe to arbitrary
  depth. ALL GATES PASS: byte-identity, NIAH 12/12 effective (16k retest 3/3 @600tok — first run was
  output-truncation), 3-model health, 10-min soak 85 reqs 0 breaches, kill-switch TESTED. Wheel
  SHELVED at ~/mlx-src + SHELF.md + ov_KILL_SWITCH.md. Zero repo edits for #2.
  LEVER #1 window opened: compile-builder standalone fs5 parity gate running; overlap-builder retained
  idle as bench owner for the upcoming server A/B.
- [x] 2026-07-04 night — REAL fs5 PARITY GATE PASS (compile-builder): 220/220 argmax, max|Δlogit|=0.0
  all steps, nvfp4 weight_scale_2 ts-carry survives the trace (per-expert tensor gathers). Standalone
  preview 1.09x — uninterpretable (default mlx env both legs). STRATEGIC GAP surfaced by crew:
  production M3 decode = MTP L=2 verify → L=1 compiled hook doesn't engage in production as-built.
  DECISION: 3-leg server A/B at golden env (MTP-on eager / MTP-off eager / MTP-off compiled) to size
  the true compiled ceiling — may flip production posture to MTP-off+compile (user call on numbers).
  Option (a) compile L=2..8 verify bucket = logged future work (likely ranking-neutral for batch-1).
  [M3COMPILE] one-shot engagement log being added (trace-time side effect as live-path gate).
- [x] 2026-07-04 night — LEVER #1 VERDICT (definitive): compiled decode = FLAT at batch-1
  (mid 24.31→24.47, 16k 24.23→24.26) despite 86× LESS host eval work (eval_impl samples 7635→89,
  gpu::eval 2238→40). mx.compile WORKS; the host was never the critical path at golden env —
  M3 BATCH-1 DECODE IS GPU-BANDWIDTH-BOUND. Same conclusion as the cap A/B, now proven twice.
  BONUS FINDINGS: (i) MTP-on == MTP-off within 0.1 tok/s at ALL contexts at temp0 (+ temp>0 skips
  drafting by design) → fs5 EAGLE = dead weight in production, user call pending; (ii) token-identity
  gate caught CROSS-REQUEST KV CONTAMINATION in the compiled path (bucket buffers not reseeded per
  request — NIAH missed it, unique prompts always grew fresh buckets). Reseed fix mandated even for
  shelving. DISPOSITION: compiled path → SHELF next to the cap wheel (default OFF), revisit for
  batch≥2 (host stacks per request) / CPU-thermal / as foundation for GPU-dispatch-gap work.
  RESIDUAL STRUCTURE: 41ms/token@16k vs ~20ms naive bandwidth floor → the missing 2× lives in
  GPU-side dispatch gaps + small-kernel underutilization (240 separate qmm dispatches/token), NOT host.

## SHELF — LEVER #1 compiled_decode (default OFF; proven machinery, wrong bottleneck @ batch-1)
WHAT: a compiled single-token (L=1) decode step over the whole 60-layer MiniMax-M3 stack.
  File: omlx/patches/mlx_vlm_minimax_m3_compat/vendor/mlx_vlm/models/minimax_m3_vl/compiled_decode.py
  Hook: 1 gated call in MiniMaxM3Model.__call__ (delegates to CompiledDecoder.forward, else eager).
  KILL SWITCH: OMLX_M3_COMPILE unset/0 (DEFAULT) = zero code-path change. =1 arms it.
  Offline suite (tasks/compile_spikes/): spike_phaseb_smoke.py (non-batch, both quant modes, +growth),
  spike_phaseb_batch.py (scheduler-built MiniMaxM3BatchKVCache via merge, +growth +delta-logit),
  spike_phaseb_reuse.py (cross-request bucket reuse). compiled_decode.py sha256 = 9959bdc0...4039.

DESIGN: offset carried as an mx.array ARG (1 trace/bucket, no per-token retrace); KV+index cache
  buffers are compile inputs=/outputs= state, written via slice_update at the dynamic offset; reuses
  the REAL submodules so math == eager; fused_index kernels driven directly with mx.array params;
  buckets by cache capacity (rebuild ~11-20ms per 256-growth, ~43us/token amortized); a once-per-trace
  [M3COMPILE] logger.info is the live-path engagement gate.

PARITY: BIT-IDENTICAL to eager offline — non-batch (fp16/affine gs64 b4/b8; and the real fs5 nvfp4+ts
  standalone gate: 220-token argmax + step-1 delta-logit=0) AND the scheduler-built batch cache
  (delta-logit=0 across a 256-growth) AND cross-request reuse (token-identical). Live A/B confirmed
  correctness modulo the contamination bug (now fixed + re-gated).

LANDMINE FIXES (all host-boundary; per-token trace stays retrace-free) — the integration gotchas:
  1. per-step scalars (total_len/cur_block/q_start/offset) must be mx.array, never python ints baked
     into the trace (else per-token retrace + unbounded compile-cache growth). offset = mx.array arg.
  2. _build must seed by BUFFER SHAPE (mx.pad), never keys[:, :, :offset, :] — live batch caches
     carry offset as an mx.array so a python slice raises "Slice indices must be integers". (_seed_to_cap)
  3. writeback must NOT set the wrapper's .offset — MiniMaxM3BatchKVCache.offset is a READ-ONLY
     property (delegates to inner .kv_cache.offset); setting it raises -> the scheduler reads it as
     unrecoverable cache corruption. Unwrap to the inner settable offset + guard. (_advance_offset)
  4. regime gate must accept a scalar mx.array offset (_as_int) AND an all-True bool mask (the batch
     path's L=1 create_attention_mask is all-True = no-op; compiled re-derives causality from offset);
     REJECT padded batches (per-row offsets aren't single-seq-safe).
  5. RESEED PER REQUEST, not per cap-bucket: a shorter new request reusing an existing bucket (no
     growth -> no rebuild) decodes on the PRIOR request's KV (cross-request contamination; NIAH missed
     it — unique prompts always grew fresh buckets). Detect a new sequence (cache keys no longer our
     state buffer OR offset discontinuity) -> reseed. (_continuation)

WHEN TO REVISIT (the regimes where the validated 86x host-work cut actually pays):
  - batch>=2 decode: host graph-build stacks per-request while the GPU is shared -> host becomes the
    wall. THIS is the regime for compile; batch-1 is bandwidth-bound so the cut is invisible there.
  - CPU-burn / thermal-headroom value (host near-idle frees power/thermal budget for the GPU).
  - As FOUNDATION for the residual 2x (41ms->~20ms @16k = GPU dispatch gaps + small-kernel
    underutilization, 240 qmm dispatches/token): compile's op-count cut is a prerequisite for any
    mega-kernel / graph-submit / fewer-larger-dispatch work. Would also want the L=2..8 MTP verify
    bucket compiled (the _multi kernels exist; logged, likely ranking-neutral at batch-1).

## FINAL WRAP (2026-07-04 night) — CAMPAIGN CLOSED
Production verified: stock mlx 0.31.2, golden env, MTP-on, fs5 warm 28.88 tok/s, kill-switches intact.
FINAL TABLE (M3-fs5, golden env, decode tok/s):
  ctx      MTP-on-eager  MTP-off-eager  MTP-off-COMPILED(9959bdc0)
  544      28.2          28.2           28.2 (no-engage by design, <4096)
  ~2k      27.3          27.3           —
  ~5.5k    —             24.3           24.5
  16k      24.3          24.2           24.3
VERDICTS: (1) batch-1 decode = GPU-bandwidth-bound; cap AND compile both flat despite doing their jobs
(compile: 86x less host eval work). Env lever = the only batch-1 win (M3 +8%, GLM +10.6%). (2) MTP-on
== MTP-off everywhere at temp0; temp>0 skips drafting → fs5 EAGLE currently dead weight (USER CALL).
(3) Compiled path CORRECT after 4 batch-cache fixes (final: reseed-per-request; self-controlling
regression test; reuse-NIAH clean 3/3 no-leak) — SHELVED default-off. (4) Future campaigns: batch≥2
A/B (compile's regime), GPU-dispatch-gap/mega-kernel (the residual 2x), mxfp8-KV long-ctx (capacity).

## CORRECTION (2026-07-04 late night, user-prompted recheck) — "MTP ≈ 0" WAS AN ARTIFACT
User challenged the MTP verdict; recheck found the stupid mistake: fs5 model_settings carries
force_sampling=true + temperature=1.0 (inherited from oQ4) → server FORCE-OVERRIDES every request's
temperature to 1.0 BEFORE the EAGLE temp==0 routing guard → "vlm_mtp routing skipped: temperature=1"
on EVERY request incl. explicit temp-0 probes → THE DRAFTER NEVER ENGAGED, not in tonight's benches
NOR this morning's "live K-sweep break-even". All "MTP-on" rows in the final table = base path.
(House lesson violated: no engagement evidence demanded for the MTP leg — compile leg had one.)
REAL MEASUREMENT (force_sampling lifted temporarily, engagement verified via vlm_mtp stats lines):
- LIVE ACCEPTANCE IS EXCELLENT: 86.9% @544ctx, 82.6% @4.8k (rounds 214/219) — vendor-grade.
- BUT NET SLOWER: 27.6 vs 28.2 short, 20.8 vs 24.3 @~5k (−15%). The serial vlm_mtp wrapper
  (bypasses BatchGenerator, yield-int sync per round, unpipelined draft+verify) eats the entire
  1.83×-tokens/round gain and more.
=> CORRECTED VERDICT: EAGLE is NOT a bummer — the drafter is great; the EXECUTION PATH is the
bummer. Fix = pipeline the vlm_mtp wrapper (async rounds / route through BatchGenerator / compile
L=2 verify bucket). Ceiling math @~5k: ~47ms/round ÷ 1.83 tok ≈ 26ms/token ≈ 38 tok/s (1.55× over
base) if wrapper overhead dies. NEW #1 CANDIDATE CAMPAIGN. Production note: user's real traffic is
forced temp=1.0 where drafting is skipped by design (rejection sampling deferred) — shipping this
needs [2b] rejection sampling OR accepting temp0-only benefit.
Production restored: force_sampling=true, MTP-on, golden env, fs5 warm.

## NEXT CAMPAIGN DEFINED (user requirement 2026-07-04): EAGLE AT PRODUCTION TEMP=1
Requirement: MTP must pay at MiniMax's recommended sampling (temp=1.0, top_p=0.95, forced) — temp0-only
is insufficient. Two mandatory workstreams:
(A) REJECTION SAMPLING in vlm_mtp verify ([2b] deferred item): speculative sampling accept
    min(1, p(x)/q(x)) + resample max(0,p-q) on reject; apply the SAME sampling processors
    (temp/top_p) to BOTH target p and drafter q for exactness. Removes the temp0 guard entirely.
    Expect acceptance to drop from 87% greedy to ~0.6-0.75 at temp1 (measure, don't assume).
(B) WRAPPER PIPELINING (tonight's finding): serial vlm_mtp wrapper eats the entire speculative gain
    (87% acceptance still nets -15%). Pipeline rounds / async the draft+verify chain / compiled L=2
    verify bucket as stretch (shelved compile work = foundation).
GATES (hard): engagement evidence via vlm_mtp stats lines on EVERY bench leg (the artifact rule);
distribution-exactness test for rejection sampling (statistical vs base sampling); live A/B at
temp1.0/top_p0.95 matched contexts; NIAH + reasoning smoke; acceptance telemetry documented.
CEILING (honest): if α_temp1 ≈ 0.65 and wrapper overhead dies → ~1.35-1.5× at engaging contexts;
if α_temp1 disappoints (<0.5), verdict flips to NO-GO and we document why.
