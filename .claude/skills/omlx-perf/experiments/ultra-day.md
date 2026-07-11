> Verified 2026-07-05 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Ultra-day experiment registry — Nemotron-3-Ultra-550B speed campaign (2026-07-05)

The day that took **Nemotron-3-Ultra-oQNVFP4 from 7.68 → 13.08 tok/s (+70%)** at batch-1, quality
held. Model: 550B-A55B, oQNVFP4 lossless repack + ts-fold; production today is the baked
`Nemotron-3-Ultra-oQNVFP4-dq8` (327GB disk / 305GB resident). EXP IDs 050–064, roughly chronological.
Pre-Ultra IDs 001–049 live in `experiments/pre-ultra.md`. Verdict vocabulary: WIN | DEAD | PARKED | OPEN-BUG.
The one-lever verdict index is `dead-levers.md`; per-ID one-liners are `experiments/index.md`.

Golden env for every measurement here: `MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000`
(missing it is a known harness artifact — see EXP-057).

---

## EXP-050: First light + D4 config-dialect gates (baseline established)
- **Hypothesis:** a pure oQNVFP4 byte-repack of Ultra 550B serves correctly and gives a clean batch-1 decode baseline.
- **Method:** convert (357.737GB / 97 shards / 1119 tensors / 96 ts sidecars, positivity green on all 48 MoE layers); D4 real-patch-path load gate = strict load OK, 48 MoE layers are `_TsFoldMoE`, base class stock, ts-fold engaged, sane greedy gens (`d4_ultra.log`). Serve window after conversion, stream-free T1/T256 warmed probe.
- **Numbers:** first-light decode **8.0 tok/s flat short→5k**; campaign-refined sustained baseline **7.68 tok/s = 130.2 ms/token**. Prefill ~104–160 tok/s @6k, ~143 @20k. Pool load 333.55GB, 57s TTFT incl load.
- **Gate fixes (2 config-dialect traps):** (a) Ultra ships `layers_block_type` WITHOUT `num_hidden_layers` → mlx_lm ModelArgs positional TypeError, fixed `setdefault num_hidden_layers=len(pattern)`; (b) `time_step_limit=[0.0,{"__float__":"Infinity"}]` tagged-infinity dict → `mx.clip` ValueError, fixed by decoding/dropping the tagged non-finite. Both patched on-disk AND in converter; weights needed ZERO changes.
- **Verdict:** WIN (baseline + quality clean).
- **Source:** `tasks/todo.md:1042-1053`; `tasks/ultra_speed.md:3-6`; `~/.claude/jobs/62f9cfe9/tmp/d4_ultra.log`.

## EXP-051: Three Ultra-only serving findings at first light
- **Hypothesis:** the anomalies seen at first light are omlx-stack serving issues, not the model/decode.
- **Method:** stream-tracer + throttle-tracer during first-light serving; compare against Super-oQNVFP4 (same arch+tokenizer).
- **Numbers/findings:** (1) **STREAM BUG** — entire answer delivered as ONE SSE chunk at completion (128-tok probe = 1 chunk @18.75s) vs Super's 8 chunks @0.134s cadence; non-stream API fine; root-cause "in flight". (2) **PREFIX-CACHE COMMIT LAG** — exact-repeat prompt misses (full 40s re-prefill), entry usable only ~40s later → fixed by SF-1 (EXP-060). (3) **THROTTLE MISPREDICTOR** — `adaptive_prefill_throttle predicted=155.78GB` from a one-time 120GB phys_footprint jump charged as per-token×2048×1.3 → fixed by SF-2 (EXP-061). Same-day serving-correctness sibling fix: codex scope-creep KEPT after Fable verification because its `matcher_state` piece is a live bug fix — stop STRINGS were silently ignored on MTP decode (3 prod models).
- **Verdict:** RESOLVED (session-verified 2026-07-05; ledger entry missing — this registry is now the record). The "stream bug" was the `reasoning_content` channel, not a serving defect (see Source). SF-1, SF-2 shipped for findings (2)(3).
- **Source:** `tasks/todo.md:1054-1065,1086-1087`; `~/.claude/jobs/62f9cfe9/tmp/ultra_speed/creep_revert_instructions.md:23-27`. CLOSURE (session-verified 2026-07-05, ledger entry never written): the stream "bug" was the `reasoning_content` channel — Ultra (a thinking model) streams thinking on `delta.reasoning_content` then the answer on `delta.content`, so a content-only client sees dead air + one final chunk (the bogus "48 tok/s"). Two stream-tracer theories refuted; non-stream API + `cadence_probe.py` correct. NOTE `todo.md:677`'s `reasoning_content` is a *separate* MiniMax-M3 thinking-parser edge (truncation mid-think), not this closure. `tasks/todo.md:1065` still reads "in flight" — the one-line closure for the ledger is supplied by this registry.

## EXP-052: Host-side sample profiling — the starvation diagnosis
- **Hypothesis:** the 130ms/token is a mix of compute and GPU starvation, not pure bandwidth.
- **Method:** two 10s `/usr/bin/sample` captures during sustained decode (`ultra_sample1.txt`, `ultra_sample2.txt`); decode thread `Thread_70414112`.
- **Numbers:** cv-wait inside `mx.async_eval` **46.5% (~60ms blocked)**; `eval_impl` CPU encode (graph traverse + Metal encode) **35.5% (~46ms)**; Python forward (nn.Module towers) **17.1% (~22ms)**; ~1800 kernels dispatched/token. **~25–34ms/token the GPU is starved**, mostly during the 22ms Python build phase when nothing is encoded. Mechanism: MLX 0.31.2 `MAX_ACTIVE_TASKS=10` compile-time constant blocks the one-ahead `async_eval` pipeline (identical to the M3 host-serialization wall).
- **Verdict:** WIN (diagnosis → defined the P0-2 op-count and P0-3 chunked-eval surfaces).
- **Source:** `tasks/ultra_speed.md:115-145,188-206`.

## EXP-053: Bandwidth byte-model + the 10.4 tok/s ceiling
- **Hypothesis:** decode is bandwidth-bound; the true ceiling comes from per-tensor serving dtypes, not blended bpw.
- **Method:** per-tensor byte accounting from `config.json` (hidden 8192, 108 layers = 48 mamba + 48 MoE + 12 attn) × source dtype × what the converter did; codex-recomputed decimals.
- **Numbers:** **78.123GB pure weights/token** (+0.890GB SSM-state/conv/KV = **79.013GB**) → ceiling **10.48 tok/s (weights) / 10.36 tok/s (incl state)** @819GB/s; baseline 7.68 = **74% of physics**. Only routed experts (22.1B) are 4-bit nvfp4 (12.457GB); ~33B active params (mamba/shared/latent/attn/lm_head) are bf16 read every token — the FP8→bf16 upcast tax is 40.467GB of the 78.1. Dense DQ8 target pool = **65.263GB** (router gate 0.403GB excluded). Killed the "55B×4.5bpw → 26 tok/s" napkin error (off by 2.5×).
- **Verdict:** WIN (the campaign's central arithmetic; every lever must cut bytes or host serialization).
- **Source:** `tasks/ultra_speed.md:105-111,148-206`; `tasks/ultra_speed_review_codex.md:70-77`; `tasks/lessons.md:132-143`.

## EXP-054: M0 — offline per-class DQ8 mode + quality probe
- **Hypothesis:** 8-bit the dense bf16 shell; find the winning container and which classes tolerate it.
- **Method:** `~/.claude/jobs/62f9cfe9/tmp/ultra_speed/ultra_probe_dense.py` — M=1, real Ultra dense shapes, offline synthetic weights, bf16 vs **affine8-gs64** vs mxfp8-gs32; RC5 gate = weighted model-level token saving ≥25ms, per-class `bf16_time/q_time`.
- **Numbers:** **GO.** Winner **affine8-gs64** (beat mxfp8 at these shapes — the M3 ledger's mxfp8 win 273 vs 301µs did NOT transfer). Aggregate weighted saving **47.2ms/token ≥ 25**. Per-class: mamba 1.44–1.54× (below the 1.6 gloss — aggregate governs, ruling ratified); **attn k/v 0.87–0.91× = LOSS → k_proj/v_proj EXCLUDED from DQ8**; q/o pending per-class confirm. Live expectation after in-stream discount ~24–28ms → DQ8-alone ≈ 9.4–9.8 tok/s.
- **Verdict:** WIN — mode LOCKED affine8-gs64; staging A(mamba)/B(shared+latent)/C(attn q/o only)/D(lm_head last) defined, each with its own kill-switch.
- **Source:** `tasks/ultra_speed.md:10-17,219-235,341-351`; `ultra_probe_dense.py`.

## EXP-055: M1 — expert container parity at Ultra shapes (gs16 kernel exonerated)
- **Hypothesis:** does Super's 0.58× nvfp4-gs16 penalty transfer to Ultra's fatter expert shapes?
- **Method:** `m1_kernel_bench.py`, isolated-eval, 200 iters, M=1, real active shapes fc1 [512,5120,2048] / fc2 [512,2048,5120] top-22, random rhs_indices.
- **Numbers:** fc1 nvfp4-gs16 408.2 / affine4-gs64 404.7 / mxfp4-gs32 395.1µs; fc2 416.5 / 423.6 / 401.3µs; layer total 824.7 / 828.3 / 796.4µs vs 316.9µs bandwidth-ideal. **All three within ±4%** — the Super gs16 tax does NOT exist at Ultra's fat latent-2048 shapes at batch-1. Absolute ~38% of peak in isolation (a known isolated-inflation trap).
- **Verdict:** WIN — nvfp4-gs16 kernel EXONERATED; P1-1 expert requant DELETED (dead on measured speed AND doctrine (b)); mxfp4 / 5-bit container ideas closed as moot.
- **Revival:** re-check at batch≥2 (M-scaling could change ordering); re-run per model before assuming transfer.
- **Source:** `tasks/ultra_speed.md:42-59,237-264,472-474`; memory `omlx-ultra-550b.md:22-25`.

## EXP-056: M4 — host-dispatch floors
- **Hypothesis:** establish the per-op host floor so any fusion ROI multiplier is not derived from binary ops alone.
- **Method:** `~/.claude/jobs/62f9cfe9/tmp/ultra_speed/ultra_probe_dispatch.py` — 2000 chained tiny binary ops [1,8192] bf16, plus one `mx.fast.metal_kernel` no-op, one skinny `mx.quantized_matmul`, one skinny `mx.gather_qmm`, all `mx.depends`-chained (never standalone).
- **Numbers:** charter seed reports **3.17µs binary op / 6.16µs metal_kernel / 7.23µs skinny gather_qmm**; prior M3 floor ~8.6µs/op (`ultra_speed.md:282`). **UNVERIFIED:** these three specific values are NOT found in any readable ledger or log (the probe's stdout was never banked); only the script and its spec are on disk.
- **Verdict:** WIN (floors established; feeds the fusion-ROI math).
- **Source:** `ultra_probe_dispatch.py`; `tasks/ultra_speed.md:281-284`. Numbers UNVERIFIED (see above).

## EXP-057: K1 — fused-expert-kernel evaluation saga (the decision measurement)
- **Hypothesis:** is there ≥2× in-stream headroom in the expert `gather_qmm` path to justify a fused expert-MLP kernel (K3, rivaling DQ8)?
- **Method:** `~/.claude/jobs/62f9cfe9/tmp/ultra_speed/ultra_probe_expert_instream.py` — in-stream ×48 `mx.depends`-chained, 12 reps, production call form (x `expand_dims(-2,-3)`, idx (1,1,22), UNSORTED since `do_sort = indices.size >= 64` is false at top-22), realistic neighbors (latent in/out, relu², weighted-sum), PLUS a dense-equal-bytes control (22×5120 addressed as one dense qmm).
- **Numbers:** FIRST run **13.35× over ideal = harness artifact** (missing golden env). Golden re-run: unsorted-production **593.4µs/layer = 28.48ms/token = 1.87× ideal** (spread 21.7%); sorted 520.4µs = 1.64× (5.6%); **dense-equal-bytes control 543.3µs = 1.71× (spread 0.8%)**. Expert-specific excess = unsorted − dense = **50.1µs/layer = 2.41ms/token**. Decision rule (≥2.0 GO; downgrades to P2 below that) read 1.87× as formally inconclusive, but the dense control is the sharper discriminator: `gather_qmm` is ~as bandwidth-efficient as a plain dense read at equal bytes.
- **Verdict:** DEAD — a fused expert kernel's addressable pool is only ~2-4ms/token, nowhere near DQ8. **P0-K PARKED to P2.** The residual 1.71× diffuse in-stream overhead is the same animal as GLM foray §10b's ~32ms → belongs to the future cross-model mega-kernel / fused-decoder-layer campaign, not this one. The former conditional 14-16 tok/s leg is DEAD.
- **Revival:** batch≥2 (expert reads amortize across the batch); OR the future mega-kernel campaign targeting the 1.71× diffuse floor directly. Re-run K1-style attribution per model before assuming transfer.
- **Source:** `tasks/ultra_speed.md:18-29,311-325`; `ultra_probe_expert_instream.py`; `GLM52_MTP_FORAY.md:348-364` (§10b).

## EXP-058: Sorted-routes lever (P0-2d) — isolated≠in-stream, kept free
- **Hypothesis:** batch-1 top-22 always runs UNSORTED `gather_qmm` (`do_sort=indices.size>=64` false; `switch_layers.py:220-222`); pre-sorting `rhs_indices` takes the kernel's fast path for a saving.
- **Method:** from K1's own sorted-vs-unsorted delta; joint permutation of (inds, scores) via `take_along_axis` inside the per-instance `_moe_call` replacement (correctness: weighted-sum epilogue is permutation-invariant when inds+scores permute jointly). Real-dims parity via `~/.claude/jobs/62f9cfe9/tmp/ultra_speed/sorted_parity_realdims.py` (E=512, TOP=22, OUT=5120, IN=2048, nvfp4-gs16). Live A/B = DQ8 ladder leg0. Kill-switch `OMLX_ULTRA_DISABLE_SORTED_ROUTES`; census `sorted_routes=48/48`.
- **Numbers:** isolated K1 delta 73µs/layer = **3.50ms/token gross** (~3 tiny dispatches/layer ≈1ms → net ~2.5ms projected). Parity: **BIT-IDENTICAL at real dims** (perm-consistent allclose + finite). **LIVE (leg0): the 3.5ms did NOT survive in-stream — speed FLAT at 8.02.**
- **Verdict:** WIN-kept-as-free (bit-identical, ~zero cost, kill-switched). A canonical isolated≠in-stream LESSON instance.
- **Revival:** batch≥2 changes economics; and when the 2b-topk fused-route kernel lands it emits its 22 indices already sorted in-register (free), superseding 2d's extra dispatches — fold 2d's gate into it.
- **Source:** `tasks/ultra_speed.md:30-35,410-436`; `tasks/todo.md:1071-1072,1082`; `sorted_parity_realdims.py`.

## EXP-059: ts-precombine "2b-lite" — the free op-count win
- **Hypothesis:** fold `fc1_ts²·fc2_ts` into ONE precomputed [512] f32 vector at patch time → the ts-fold becomes one gather+mul instead of ~4 ops × 48 layers.
- **Method:** per-instance `__call__` REPLACES `_moe_call` wholesale on Ultra instances only (never touches the shared `_TsFoldMoE` class); `combined_ts = fc1_ts**2 * fc2_ts` precomputed. Kill-switch `OMLX_ULTRA_DISABLE_TSPRE`. Numeric-identity test vs stock fold. (Ships in the SAME `_moe_call` replacement package as 2d/EXP-058.)
- **Numbers:** BIT-IDENTICAL to stock fold — RELU² MLP is degree-2 homogeneous, so `weight_scale_2` folds as a single per-expert scalar `fc1_ts²·fc2_ts` (exact, and simpler than SwiGLU's split-gate ts).
- **Verdict:** WIN — shipped.
- **Source:** `tasks/ultra_speed.md:394-398`; `tasks/lessons.md:114-117`.

## EXP-060: SF-1 — early prefix-cache-index publish
- **Hypothesis:** the token→block index entry is visible only after the store worker finishes → exact-repeat prompts miss and re-prefill (finding #2, EXP-051).
- **Method:** publish the index entry on the inference thread immediately after `mx.eval(*pre_eval_arrays)` (`store_cache_main_dispatch`, ~`scheduler.py:9026`); entry marked "hot/in-memory" until the worker confirms persist, worker-failure path RETRACTS the entry; widen the freshness bridge (min-prompt 8192 → model-scaled; timeout 4s → EWMA of store durations). Kill-switch `OMLX_DISABLE_EARLY_INDEX_PUBLISH`. Required tests T1 (racing-reader truncate-to-valid-prefix) + T2 (retract-while-shared-ref) before live.
- **Numbers:** exact-repeat 5k **TTFT 40s → 3.0s (13×)** live. Reviewed APPROVE-WITH-NOTES; race verified safe (`reconstruct_cache` truncate-on-miss + `save_block` pending-write staging).
- **Verdict:** WIN — shipped live.
- **Source:** `tasks/ultra_speed.md:36-40,532-554`; `tasks/todo.md:1075`; `creep_revert_instructions.md:16-20` (T1/T2).

## EXP-061: SF-2 — prefill-throttle clamp
- **Hypothesis:** `_predicted_chunk_transient` takes MAX(last-delta, EWMA, static)×1.3; a one-off allocation spike poisons the tracker → chunk sizes collapse → prefill crawls (finding #3, EXP-051).
- **Method:** 3-line clamp of BOTH tracker-derived signals to `K × static_estimate` (K≈8, env `OMLX_TRANSIENT_CLAMP_K`, 0=off) inside `_predicted_chunk_transient` (`scheduler.py:3728`); static path + `_PREFILL_TRANSIENT_SAFETY`/`_PREFILL_ABORT_MARGIN` untouched (the Metal async OOM SIGABRT is what the margin guards). INFO log on clamp engage.
- **Numbers:** first-light symptom = `predicted=155.78GB` from a one-time 120GB phys_footprint jump (expert weight wiring, chunk 0) charged as per-token rate ×2048×1.3; cost was ~10s-of-ms TTFT + first chunk 2048→1817, decode UNTOUCHED. After fix: **zero `adaptive_prefill_throttle` noise**.
- **Verdict:** WIN — shipped live.
- **Source:** `tasks/ultra_speed.md:556-566`; `tasks/todo.md:1059-1064,1076`.

## EXP-062: DQ8 live ladder — the payoff (7.68 → 13.08 tok/s)
- **Hypothesis:** load-time affine8-gs64 DQ8 of the dense bf16 shell (staged A→D), stacked on the serving fixes, cuts enough bytes to reach ~13 tok/s.
- **Method:** live A/B ladder, all engagement-verified, stream-free T1/T256 probes, short==5k at every rung. Launch env `OMLX_ULTRA_DQ8_MAMBA=1 OMLX_ULTRA_DQ8_MOEDENSE=1 OMLX_ULTRA_DQ8_ATTN=1 OMLX_ULTRA_DQ8_LMHEAD=1`. Engagement: `[ULTRA-DQ8] <stage> expected==actual` (96/192/24/1, HARD-FAIL on mismatch) + `[ULTRA-DECODE] sorted_routes=48/48`.
- **Numbers (ladder):** leg0 serving fixes + sorted routes = **8.02** (flat vs baseline; sort lever's 3.5ms did not survive in-stream — EXP-058); leg1 +DQ8 mamba (96/96) = **10.38**, −17.7GB; leg2 +moedense (192/192) = **12.52**, −26.1GB cumulative; leg3 +attn q/o (24/24) = **12.85**; leg4 +lmhead (1/1) = **13.07–13.08**, 305.07GB resident. Net **7.68→13.08 (+70%)**, resident 333.6→305.1GB (−28.5GB).
- **Pipeline catch (live-path law):** Fable review caught DQ8 stage B silently DEAD under the `_TsFoldMoE` class swap (type-name filter + `0==0` blind hard-fail) — REPRODUCED, fixed via block_type filters + independent census.
- **Verdict:** WIN.
- **Source:** `tasks/todo.md:1067-1090`; memory `omlx-ultra-550b.md:10-19`.

## EXP-063: Quality battery — no quant damage, 550B lift visible
- **Method:** serial, non-stream, temp0 benches (concurrency confounds quality — see the oQ4e SpecPrefill artifact lesson).
- **Numbers:** gsm8k **91.7 (n=60)** baseline / **96.67% spot (n=30)** post-campaign; mmlu **85.3 (n=150)**; arc **96.7 (n=150)**. vs Super-oQNVFP4 92/81/95 → mmlu +4.3pp, arc +1.7pp, gsm8k tie = 550B lift visible, NO quant damage. DQ8 held quality (post-DQ8 gsm8k spot == baseline within noise).
- **Verdict:** WIN.
- **Source:** `tasks/todo.md:1052-1053,1068-1069`; `tasks/ultra_speed.md:5-6`; memory `omlx-ultra-550b.md:11-12`.

## EXP-064: Baked-checkpoint pipeline validation (production shape)
- **Hypothesis:** bake DQ8 at convert time → kill the ~3min load-time quantize pass and the transient peak; prove pipeline equivalence to load-time DQ8.
- **Method:** converter `--dq8` (affine8-gs64 baked), shared `DQ8_STAGES` map (patch + converter import the SAME object, is-identity tested), 3-way baked detection (linear/baked/other; env vars inert on baked; corrupt still raises). Offline gates via `~/.claude/jobs/62f9cfe9/tmp/ultra_speed/baked_gate_offline.py` (census + bit-parity). Full run: NVFP4 source (T7) → `Nemotron-3-Ultra-oQNVFP4-dq8`, 327.192GB, 1745 tensors (=1119+313×2), ~45min.
- **Numbers:** OFFLINE — census **313/313** q8 triples + 0 strays + experts/ts intact; **BIT-PARITY EXACT** vs `mx.quantize` of the old checkpoint's bf16. LIVE — all 4 "baked checkpoint detected" INFO lines, resident **305.08GB (==305.07 load-time)**, decode **13.09/13.07 (==13.07/13.08)**, gsm8k **96.67% n=30 (==load-time)**, load 68s no-quantize-pass.
- **Verdict:** WIN — production model is now `Nemotron-3-Ultra-oQNVFP4-dq8` (old oQNVFP4 deleted, NVFP4 master archived on T7). This flow (repack + quant-first bake + gates) is the validated template for the GLM-5.2-NVFP4 campaign.
- **Source:** `tasks/todo.md:1092-1104`; `baked_gate_offline.py`; memory `omlx-ultra-550b.md:40-46`.

---

## GAPS
- **M4 dispatch floor values (EXP-056)** — 3.17/6.16/7.23µs come from the A9 charter seed only; the probe's stdout was never banked to any readable ledger/log. Re-run `ultra_probe_dispatch.py` under golden env to confirm.
- **Stream-bug resolution (EXP-051)** — RESOLVED in-session 2026-07-05 (reasoning_content channel; not a serving defect) but the ledger `todo.md:1065` still reads "in flight"; the closing entry was never written to `tasks/todo.md` (this registry is the record). If the stream-tracer output is recovered, bank it.
- **M0 q/o per-class ratio (EXP-054)** — attention q_proj/o_proj was "pending per-class confirmation before stage C"; the exact q/o ratio is not separately quoted in the readable ledger (stage C shipped at 24/24 in EXP-062, implying ≥1.0×).
- **M2 (ssm), M3 (route), M5 (overlap), M6 (composed), M7 (mamba-seq) probes** — specified in `tasks/ultra_speed.md:266-302` but no measured outputs are recorded in the readable corpus (P0-3/2a/2b-topk were not needed once DQ8 carried the campaign). Status: unrun or unbanked.
