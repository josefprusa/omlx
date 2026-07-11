> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Future campaigns — mapped but NOT run

Five workstreams that are **designed, arithmetic-backed, and asset-banked but not executed**. Each entry: goal · why it pays (cited) · banked assets (exact paths) · entry conditions · first steps. These are the standing backlog when someone asks "what's next on the box." Nothing here is live — do not present any number below as measured production behavior.

---

## 1. GLM-5.2-NVFP4 (465 GB NVIDIA release) — ✅ EXECUTED 2026-07-06

**DONE — see `models/glm52.md §11` (EXP-067/068).** Landed `GLM-5.2-oQNVFP4` at 398.9 GiB resident, 400k-fp16-KV fits with ~93 GiB slack; decode 16.3-16.5 tok/s (0.71× of 3.5bpw, napkin-exact — the "faster than 3.5bpw" estimate below was wrong, the quality-per-GB trade is the actual product); gsm8k n=60 wash vs 3.5bpw. **Residue:** mmlu/arc A/B (the real quality question), 400k soak, `--shared-nvfp4` lever. Two corrections to the arithmetic below: MTP-drop and shell-DQ8 double-counted (~22 GB overlap → real out is ~428 GB not 414), and the M3 fused-shared recipe does NOT transfer (GLM runs shared experts as separate modules by design; they went affine8-gs64). Original charter kept for the record:

**Goal.** Serve `nvidia/GLM-5.2-NVFP4` on the box using the **validated Ultra DQ8 doctrine** (lossless NVFP4 expert repack + bf16-shell → DQ8 + MTP-drop), landing a **~425 GB-resident, 256k-native-fp16-KV** GLM-5.2 at an estimated **~17-19 tok/s** — faster decode and more headroom than today's 3.5bpw build, at NVIDIA-calibrated 4-bit quality.

**Why it pays (arithmetic, `tasks/glm_quant_matrix.md` footnotes [1]/[2]/[3]).** The NVFP4 release is **464.865 GB** = 362.39B U8 (packed FP4 expert weights) + 45.30B F8_E4M3 (block scales) + **28.55B BF16 (all non-expert weights)** + 19 456 F32. Crucially, **NVFP4 quantizes ONLY the routed experts** and leaves everything else BF16 — all MLA attention, the DSA lightning indexer, AND the **entire MTP layer 78**. That is why a "4-bit" export is still 465 GB, and it is exactly the lever:
- **DQ8 the 28.55B BF16 shell** (affine8-gs64, the Ultra stage-A/B mechanism): 28.55B × 2B = 57.1 GB → ~28.6 GB → **save ~28.5 GB**.
- **Drop the MTP layer 78** (shipped-but-unquantized, its own full decoder block + **256 bf16 routed experts**, `config.json:194` `n_routed_experts:256`, `tasks/glm_quant_matrix.md:3`): **est. ~22 GB saved** (projection — no source states the exact layer-78 magnitude; `memory/omlx-ultra-550b.md:37` backs only the MTP-drop → 256k-native-KV concept). GLM production runs `mtp_enabled=false` anyway (models/glm52.md §8 — MTP is a measured 0.85× loss).
- Net: ~465 GB source → ~**414 GB on disk** after both cuts → **~415-425 GB resident** (the charter's ~425 GB estimate; resident depends on the working set), leaving room for **256k native fp16 MLA-KV** on the 512 GB box (weights + KV < ~464 GB cap).

**The SwiGLU catch (do not copy the Nemotron fold).** GLM experts are **SwiGLU (gate/up/down)**, not Nemotron's relu². The relu²-homogeneity trick that collapses `weight_scale_2` into a **single per-expert router-score multiply** (`tasks/oqnvfp4_nemotron.md §4`) **does NOT transfer**. SwiGLU needs the **M3-proven** mechanism: fold `down_ts` into the router scores **and** an **output-side elementwise ts fold** on the gate_up branch (`memory/omlx-glm52-decode-opts.md`, M3 oQNVFP4; the router-only fold is insufficient because the nonlinearity sits between gate_up and down). Template = MiniMax-M3's `omlx/tools/oqnvfp4_convert.py` (SwiGLU), **not** `oqnvfp4_nemotron_convert.py` (relu²).

**Banked assets.**
- Quant map (gold): `tasks/glm_quant_matrix.md` — full per-tensor split across all 4 GLM-5.2 releases + the NVFP4 `hf_quant_config.json` `ignore` list (§3: `lm_head`, `embed_tokens`, `layers.0*/1.*/2.*`, per-layer `self_attn*`+`shared_experts*`, `layers.78*`; `quant_algo=NVFP4 group_size=16 kv_cache_quant_algo=FP8`; producer `modelopt 0.46.0`).
- Validated pipeline template: conversion.md + the Ultra converter `omlx/tools/oqnvfp4_nemotron_convert.py --dq8` (repack + quant-first bake + census/bit-parity gates) — proven end-to-end on Ultra (`tasks/todo.md` "DQ8 CHECKPOINT PRODUCTIZATION").
- Byte-repack pre-flights already green (MiniMax-M3): NVIDIA-NVFP4 → MLX is a **pure byte-copy** (max|diff|=0.0), `tasks/todo.md` "PRE-FLIGHT 2 PASSED".
- The glm_moe_dsa loader + native kernels already run mixed-precision GLM (models/glm52.md).

**Entry conditions.**
- **Disk-shuffle prerequisite** (like Ultra's gate): a 465 GB standalone load cannot coexist with a loaded server model — need an **offline window** + the source staged on the **T7 external** (`tasks/oqnvfp4_nemotron.md §12` disk-fit discipline).
- Converter generalized to `glm_moe_dsa` arch + **SwiGLU ts-fold** (the one net-new code vs the Nemotron converter).
- fp8 block-scale handling for GLM's `_scale_inv` naming (already in the oQ `_block_dequant_fp8` path, `tasks/todo.md` "PRE-FLIGHT 2").

**First steps.** (1) Freeze the `ignore` list from `hf_quant_config.json` into the converter (start from `glm_quant_matrix.md §3`). (2) Port the M3 SwiGLU ts-fold into a `glm_moe_dsa` converter path; assert per-expert `weight_scale_2 > 0`. (3) D2 bit-exactness on a range-fetched expert triplet (repack == ModelOpt formula, max|diff|=0.0). (4) DQ8 the 28.55B shell, drop `layers.78*`, emit config `quantization` per-module map. (5) D4 offline real-path load test (coherent gen, non-interference with other loaded models). Then live A/B vs the 3.5bpw build.

---

## 2. Batch≥2 / speculative revival

**Goal.** Make speculative decoding (EAGLE-3 on MiniMax-M3, MTP on GLM) actually **pay**, by moving off the batch-1 regime where it is structurally capped at break-even. Everything is built; the only missing ingredient is concurrency.

**Why batch-1 is dead, and why batch≥2 flips it (`tasks/eagle_temp1.md` WORKSTREAM B; `GLM52_MTP_FORAY.md §11`).** Live-profiled on M3, block_size=2 (L=2 verify), 83-86% accept:
- verify GPU-wait = **75 ms @5k = 1.83× a base single-decode (41 ms)**, commits 1.83 tokens → **GPU-level break-even**; **87 ms @16k = 2.12×**, commits 1.85 → **loss**.
- **Mechanism:** verify of L=(K+1) tokens reads ~(K+1)× the **dominant routed-expert weights** (distinct experts per token in a MoE), while dense/attention weights are read ~once. So verify cost scales ~1:1 with committed tokens → **spec is capped at break-even at batch-1 even at α=1, sampling-independent** (so rejection sampling / temp1 can't rescue it — it's strictly worse, fewer accepts for the same verify cost).
- **Batch≥2 changes the denominator:** expert-weight reads **amortize across the batch** (the literature's MoESD result — "even 2-4 concurrent requests flips economics positive as-is", `tasks/todo.md` "SPEC-VERIFY ROADMAP"). Draft-side work is separately capped at **1.54×** (proven on the marginals). Agent fan-outs already produce the concurrency.

**Banked assets (exact paths).**
- **EAGLE-3 drafter (vendor-grade):** `$OMLX_COLD_STORAGE/omlx-quant-work/MiniMax-M3-EAGLE3` (6.5 GB bf16, `LlamaForCausalLMEagle3`, source `Inferact/MiniMax-M3-EAGLE3`). Live greedy acceptance **86.9% @544ctx / 82.6% @4.8k** — the drafter is NOT the problem (`tasks/eagle_temp1.md`).
- **omlx adapter:** `omlx/patches/mlx_vlm_mtp/eagle3_minimax.py` — fc_norm triple + POST-norm recurrence, target-hidden taps at **layers 2/30/57** (`fc_norm.0/1/2`), prefix-KV seeded during prefill (mandatory for chain acceptance), mxfp8 drafter default-on (`OMLX_EAGLE3_MXFP8`, accept delta <1pp). Settings: `vlm_mtp_draft_block_size=2`, `vlm_mtp_enabled=false` (models/minimax-m3.md §4).
- **Phase-1 keeper:** multi-query flash verify kernel (`flash_sparse_sdpa_multi` in `fused_index.py`) — argmax-exact vs MSA-prefill, 14% faster at every L (`tasks/eagle3_build.md §PHASE 1 SUMMARY`).
- **α harness (verify economics at temp1):** vendored verbatim to `scripts/archive/eagle_gate0_alpha.py` + `gate0_ops.sh` (from the now-ephemeral `jobs/tmp`; sanitize before reuse — `scripts/archive/README.md`) — offline-validated (top_p filter/renorm == production to <1.6e-7; α anchors exact) but **never run live** (auto-mode guard blocks the shared-server stop). Spec: `tasks/eagle_temp1.md §GATE 0`.
- **Compiled decode step:** vendored `compiled_decode.py` behind `OMLX_M3_COMPILE=1` (default off; bit-identical offline incl. growth boundary; flat at batch-1 but **batch≥2 is its real regime**, `tasks/todo.md` "LEVER#1 VERDICT").
- **Cap wheel (SHELF):** `~/mlx-src` + `SHELF.md` — `MLX_MAX_ACTIVE_TASKS` env-configurable MLX v0.31.2 stage-2 build, ABI-safe. Flat at batch-1 (backpressure wait was already overlapped); may matter once host work stacks at batch≥2.
- **GLM MTP head:** `omlx/patches/mlx_lm_mtp/glm_moe_dsa_model.py` (inert, correct, per-position accept logging), `OMLX_MTP_VERIFY_FAST` (models/glm52.md §8).
- **Batch-KV integration know-how (the M3 batch path is already hardened):** `tasks/overlap_levers.md:154-168`, `tasks/compile_spikes/PHASE_B_STATUS.md §3b-3d`. The four landmines any batch≥2 / mega-kernel integration over the real cache WILL re-hit live in `gotchas.md` ("batch-KV cache"); the offline repro is `type(c).merge([c])` per layer (rebuilds the live `MiniMaxM3BatchKVCache`, no server; `tasks/compile_spikes/spike_phaseb_batch.py:52-54`). Compiled-decode regression suite: `compile_spikes/spike_phaseb_{smoke,batch,reuse}.py` + `phaseb_real_parity.py` + `run_parity_window.sh`, beside vendored `compiled_decode.py`.
- **Verify-economics + concurrency harnesses** (vendored verbatim to `scripts/archive/`, sanitize first — the skill otherwise ships only the *serial* `acc_bench_serial.py`): acceptance-α teacher-forced `gate_mtp.py`/`eagle_accept.py`/`eagle_gate0_alpha.py`/`alpha_prompts.py`/`shadow_trace.py`; verify-width / L-sweep `width_bench.py`/`eagle_gate1_both.py`; MoE same-vs-diverse M-scaling `probeA_gather_qmm.py`/`probeB_linears.py`; N-stream aggregate tok/s `m3_parallel.py`; concurrent-vs-serial quality `acc_bench.py` (6-worker; exposed the SpecPrefill 92%→15% cliff). These are #2's literal first steps.

**Entry conditions.** A real batch≥2 serving path exercised under concurrent requests. **Enable it:** `OMLX_CONTINUOUS_BATCHING=true` (default `"false"`, `config.py:199`) + `OMLX_MAX_CONCURRENT_REQUESTS≥2` (alias `OMLX_MAX_NUM_SEQS`, `settings.py:930`; effective default 8) — decode then runs the tight 0.03s concurrent burst budget vs 0.1s single-stream (`omlx.md §EngineCore decode-burst`). The compiled L=2 verify graph is the "compile's real regime" foundation. `tasks/eagle_temp1.md §WORKSTREAM A` (rejection sampling, `OMLX_VLM_MTP_REJECTION`) only unblocks temp>0 — it does NOT beat the wall alone.

**First steps.** (1) Stand up a batch-2 A/B harness (two concurrent greedy streams). (2) Re-run the verify-economics profile (`OMLX_VLM_MTP_PROFILE`) at batch-2 — does the (K+1)× expert read amortize to <1× per committed token? (3) If yes, compile the L=2 verify bucket (`compiled_decode.py` foundation) and measure MoE verify-batching. (4) Then the α harness (`gate0_ops.sh`) becomes meaningful for temp1 economics.

---

## 3. Mega-kernel / fused decoder layer (cross-model, the biggest remaining structural lever)

**Goal.** Collapse each decoder layer's ~dozens of small MLX ops into **one fused Metal kernel per layer**, eliminating the inter-kernel dispatch/dataflow overhead that is the residual on **all three** large models. Months-class, but it is the only thing that could cross ≥1.3× at batch-1 AND unlock MTP economics.

**Why it pays (the same diffuse overhead measured three ways).**
- **Ultra (`tasks/ultra_speed.md §4 K1`):** the dense-equal-bytes control ran at **1.71× ideal** — `gather_qmm` is as efficient as a plain dense read, so the 1.71× is **pure diffuse in-stream overhead every op pays** (expert-specific excess is only 2.41 ms/token and already ruled out). This 1.71× floor is explicitly assigned to this campaign.
- **GLM (`GLM52_MTP_FORAY.md §7/§10b`):** the L=2 verify marginal decomposes to **W≈41 ms/token** with **<0.1 ms FLOPs and ~0 extra weight reads** — a **~32 ms diffuse residual** across 78 layers that `mx.compile` cannot fuse (measured **1.06×** — the overhead is in opaque big kernels, not fusible glue). "The only thing that could plausibly cross ≥1.3× is a fully-fused GLM-DSA decoder-layer Metal kernel."
- **MiniMax-M3 (`tasks/todo.md` "LEVER#1 VERDICT"):** residual ~2× @16k (41 ms vs ~20 ms bandwidth floor) = GPU-side dispatch gaps / small kernels (240 qmm dispatches/token), needs Metal-level fusion not host work.

**The MTP unlock (why this is worth months).** If the L=2 verify forward became as per-token-efficient as the L==1 decode (W → ~10 ms), GLM's `t_verify(2) ≈ 34 ms ≈ 1.2× decode` → at α=0.66 that is a **~1.4× decode win** (`GLM52_MTP_FORAY.md §7`). The mega-kernel is thus a prerequisite for spec decode paying at batch-1, and it stacks with batch≥2 (#2).

**Banked assets.** K1/K2/K3 kernel-design specs (`tasks/ultra_speed.md §4 P0-K` + git history of that file); the GLM native-kernel `csrc` infrastructure (`omlx/custom_kernels/glm_moe_dsa/`, nanobind 2.12.0 / ABI v19 pinned, metal.md); the M3 fused-index/topk kernels as fusion precedents (`fused_index.py`). ThunderMittens is a **reference only** (dead MLX-0.21 fork, no dependency; `tasks/todo.md` "TM CAPABILITY MAP").

**Banked methods (the mega-kernel's correctness + success gates, not yet ported to `scripts/`).**
- **Whole-model teacher-forced parity** — `tasks/compile_spikes/phaseb_real_parity.py`: eager records the decode stream, feed the SAME stream to the candidate from a deep-cloned warm cache, then step-1 Δlogit + all-step max|Δlogit| + argmax-agreement + first-divergence + steady median/min ms/tok. This is #3's "bit-parity vs the eager chain" gate.
- **Op-count collapse** (the success metric) — `tasks/compile_spikes/spike_de_scale.py:125-129`: count primitives via `mx.export_to_dot` (Ultra spike E: 4931→3007 ops, 301 fused `Compiled*`). A mega-kernel must move THIS number, not just wall-time.
- **Dense-GEMM FLOP ceiling** ("is a hand kernel even worth building") — `scripts/archive/prefill_moe_ceiling.py` (+`prefill_moe_ceiling2.py`, real `_gather_sort` path): quant `gather_qmm` TF/s at M=8192 vs a dense fp16 GEMM of identical FLOPs. If the quant path already ≈ the dense ceiling, no hand kernel beats MLX. The scalar-frozen-constant retrace trap + `metal_kernel` compile-traceability are in `mlx.md §mx.compile`.

**Entry conditions.** Explicit lead sign-off (P2, months-class). Re-run **K1-style in-stream attribution per model** before assuming transfer — the diffuse floor is real but its split varies (M3 attributed to expert divergence; GLM's identical-vs-diverse control shows diffuse per-op dominates — the two are not identical, `GLM52_MTP_FORAY.md §11`).

**First steps.** (1) Pick one model/layer type (GLM MoE layer or Ultra mamba+MoE) and prototype a single fused decoder-layer kernel. (2) Bit-parity vs the eager chain. (3) In-stream A/B — does the 1.71×/~2× diffuse floor actually shrink? Only then commit to the full forward rewrite.

**Puzzle-class caveat (EXP-092/095/098, 2026-07-09):** for Puzzle-class models (fast trunk, 88 thin
layers, small expert reads) this campaign is **RESOLVED-NEGATIVE**: decode is COMPUTE-bound, eager
overhead pools are pipeline-hidden (Law 19), pool A survived at only +2.9%, the fused router lost
7.3× (GLM router-fusion DEAD verdict reinstated), pool C faulted under pipelining (Law 18). The
GLM/Ultra framing above may still hold where expert reads are fat and the diffuse floor is real —
expert-read-size vs launch-floor is the gating variable; re-derive per model before funding.

---

## 4. mxfp8 KV-cache for MiniMax-M3 (capacity play, design rules banked)

**Goal.** An mxfp8 (or int8) MLA/index-KV cache for M3 as a **RAM-capacity** lever for very long context, NOT a speed lever — with the design rules already derived from the GLM int8-MLA-KV post-mortem so the same wall isn't hit twice.

**Why it's a capacity play only (the same-wall lesson, `memory/omlx-glm52-int8-mla-kv.md`).** On Apple GPUs there is **no fp8/int8 MMA** (simdgroup_matrix is fp16/bf16/fp32 only), so any quantized KV must **dequant-on-read** before the kernel. On GLM this made prefill **~2× slower** (dequant transient death-spirals the throttle) with **zero speed upside** — the KV saving (42%) only matters when fp16 literally doesn't fit. mxfp8 hits the **identical** dequant wall with no quality upside over token-identical int8 → **don't test the GLM/mxfp8-latent path on this box**. For **M3 specifically** (`tasks/todo.md` "REASONING @103k"): mxfp8-KV = a RAM play but **~0.7 ms/tok SLOWER** as a bolt-on (dispatch-bound compact path); the **speed-positive variant is int8 *index keys*** (not the latent); the **zero-risk variant is quantizing only the M3 SSD prefix-cache blocks** (no live-decode cost).

**Banked assets.** The full GLM int8-MLA-KV implementation (`omlx/patches/glm_moe_dsa/int8_latent_cache.py` — **NOT at HEAD; on sibling branch `fix/glm5.2-native-kernels` only**, recover via `git show fix/glm5.2-native-kernels:omlx/patches/glm_moe_dsa/int8_latent_cache.py`; dequant-on-read, boundary-snapshot tier) as a design reference; the ThunderMittens **fp8 MLA-KV reference** (insert + partitioned dequant-on-read decode, e4m3+UE8M0/64, 1.25× penalty) which **de-risks** the port (`tasks/todo.md` "TM CAPABILITY MAP" find #1). Design rules + the 5-clamp throttle-stack post-mortem: `memory/omlx-glm52-int8-mla-kv.md`, kv-cache.md.

**Entry conditions.** A concrete need for M3 context beyond what fp16 KV fits on 512 GB (today M3 reaches 128k comfortably, models/minimax-m3.md §6). Absent that need, this is **parked by design** — the int8-MLA lesson says the trade is only worth it when fp16 doesn't fit.

**First steps.** (1) Pick the variant by risk: SSD-blocks-only (zero-risk) → int8-index-keys (speed-positive) → full mxfp8 latent (capacity, slower). (2) Reuse the GLM boundary-snapshot handler pattern; avoid extending TurboQuant (wrong path, `memory/omlx-glm52-int8-mla-kv.md`). (3) Gate on a real >128k M3 workload before building the latent path.

---

## 5. SpecPrefill 6-way concurrency bug (Nemotron-Super) — MITIGATED 2026-07-07 (bug still unfixed)

> **Production exposure CLOSED:** user ordered SpecPrefill disabled fleet-wide — `specprefill_enabled=false` set on all four enabled models (Qwen3.5-122B-oQ4, Super-oQ4e incl. the fleet default, GLM-5.1-2.9bit, MiniMax-M2.7-6bit), settings backup `model_settings.json.bak-specprefill-off`, server bounced. The underlying `specprefill.py` concurrency bug below remains UNDIAGNOSED — this section stays as the repro/triage map if SpecPrefill is ever wanted again (its TTFT win is the only reason to reopen).

**Goal.** Root-cause and fix the accuracy collapse when the fleet-default **Nemotron-3-Super-oQ4e** runs SpecPrefill under concurrent load. This is the one item here that is an **active production hazard**, not a speed opportunity.

**Symptom (`tasks/oqnvfp4_nemotron.md §11`, "Artifact flagged — NOT quantization").** `acc_bench`'s **concurrent (6-worker) gsm8k scored oQ4e 15%**; **single-shot AND serial both 92%**. It is a **SpecPrefill-under-concurrency degradation**, reproducible, and it is **not** a quantization defect (the weights are fine — proven by the serial score and the tied oQNVFP4 A/B). Any concurrent request stream hitting Super's SpecPrefill path is exposed; **agent fan-outs qualify**.

**Why it matters.** Super-oQ4e is `is_default=true` in `~/.omlx/model_settings.json` with `specprefill_enabled=true` (draft = `NVIDIA-Nemotron-3-Nano-30B-A3B-oQ4`, `keep_pct=0.2`, `threshold=2048`). A 6× accuracy cliff under the exact concurrency that agent workloads produce is a silent correctness failure. Until fixed, **report Super's serial gsm8k number and treat concurrent SpecPrefill as suspect** (models/nemotron-super.md §9).

**Banked assets.** The SpecPrefill implementation `omlx/patches/specprefill.py` (~32.6 KB); the reproducing harness `acc_bench.py` in concurrent (6-worker) mode (vendored to `scripts/archive/`); the clean serial/single-shot baselines (92%). Draft model on disk at the settings path.

**Entry conditions.** None — this is fixable now; it needs an offline reproduce-and-isolate pass (a QA/Builder split per the workflow doctrine).

**First steps.** (1) Reproduce: `acc_bench` gsm8k, 6 concurrent workers, Super-oQ4e, capture per-request drafts. (2) Isolate whether the degradation is **shared draft-cache/KV collision** across concurrent requests (SpecPrefill keeps draft state; concurrency may cross-contaminate the kept-token mask or the draft cache) vs a scheduler batching interaction. (3) Bisect: disable `specprefill_enabled` under 6-worker load — if accuracy returns to 92%, the fault is inside `specprefill.py`'s concurrent path, not the base MoE. (4) Fix, then re-gate serial AND concurrent gsm8k both at 92%. Cross-ref gotchas.md (concurrency-correctness) and ops-runbook.md.

## M3 int8 GQA-KV (recon complete 2026-07-07, NOT started)
MiniMax-M3 is plain GQA (60L x 4KVh x 128, ~120KB/tok fp16) — Int8MLAKVCache does NOT drop in. Scope: (a) new plain-GQA int8 cache (mlx-lm stock QuantizedKVCache quantizes K+V — rebuild w/ start-threshold + native block persist doctrine); (b) make_cache wiring at vendor language.py:2400-2404 (quantize only MiniMaxM3KVCache.kv_cache; index_keys stays fp16 per indexer doctrine); (c) port int8 arming into engine/vlm.py (missing entirely; args via model.language_model.args); (d) GQA preflight KV accounting (existing block is DSA-only, no-ops harmlessly); (e) BLOCKER: compiled_decode.py CompiledDecoder reads raw kv.keys/values buffers (seeds/writes back directly) — int8 must bail off the compiled fast path (decode regression vs 27.5 tok/s) or fund an int8-native kernel. ALSO: M3 settings entry has NO max_context_window (inherits 1M) and weights leave big RAM headroom — capacity case is weak; revisit only on a real context wall. Win if done: ~47% KV cut (~23GB @400k).

## CANCELLED: restore reconstruct batching (EXP-084)
The 83s was model-load conflation; real 130k restore = 1.3-1.8s ([restore-profile] telemetry now permanent). Do not revive without fresh telemetry showing load_loop/build actually slow.

## Tier-clamp bounded test (small, from EXP-084)
The secondary watermark tier-clamp (scheduler.py ~4375) pins deep-prefill chunks at 1024 whenever
current >= soft watermark — permanent condition with 424GB weights. The estimator gate beneath it is
now honest (intercept+slope + conservativeness floor), so the crude bucket may be removable. Bounded
expectation: ~10-20% deep-prefill (chunking overhead only — the depth decline itself is indexer O(N*K)
physics). Test: env-gated bypass, 128k cold A/B, watch enforcer.

## MLX v0.32.0 (2026-07-07) — upgrade triggers, mapped 2026-07-08 (not yet adopted; box stays on 0.31.2)
- **qmv_wide (#3764)** — small-batch quantized matvec, M∈[2,8): old `qmv` RE-READ the full weight per
  vector (batch-2 decode paid 2× shell bytes!); qmv_wide dequants each group once and amortizes.
  M3 Ultra measured (PR table): affine4 1.1-1.4×, int8 1.2-1.7×, nvfp4 1.4-1.7×, mxfp8 1.4-2.0× at M=2-8.
  DIRECTLY re-prices campaign #2 (batch≥2) AND the MTP-verify economics that killed EXP-090/GLM-MTP
  (verify cost was weight-read bound at M=K+1). **MEASURED on Puzzle oQ48 (EXP-093, side venv
  .venv-mlx032): whole-model batched decode B=4 137→168 agg tok/s (1.23×, per-stream 34→42), B=8
  179→237 (1.32×); kernel-level in_proj qmm M=4/8 = 1.61-1.65×; single-stream FLAT (54.2 vs 54.6).**
  Re-run the GLM/Hy3 MTP ladders on 0.32 before re-verdicting them.
- **#3637 fused SDPA vector kernel for asymmetric Q/V head dims (192,128)** — exactly MLA decode geometry;
  bench vs GLM's custom flash-decode + gather+SDPA (<64k) paths. **#3455 MLX_SDPA_BLOCKS** env = new knob.
- **#3448 ST_F8_E8M0** safetensors dtype → native mxfp8 checkpoint loading (mxfp8 experiments unblock).
- Micro (Puzzle-relevant but ~1-2% at best): #3663 vectorized axis-0 concatenate (mamba conv update does
  40/token), #3754 rms_single_row register caching. Does NOT touch the launch-granularity floor (EXP-092).
- **UPGRADE COSTS:** MLX 0.32 bumps **nanobind → 2.13.0** (#3722) — the GLM native-kernel ext is ABI-pinned
  to 2.12.0 (commit 3c224ed): MUST rebuild the ext (env-setup.md) or GLM kernels die silently (Law 11 grep).
  qmm/gemv kernel codegen restructured (#3424, #3705) — µs-level floors may shift; re-run kernel_parity +
  T1/T256 fleet ladder; golden-env semantics re-verify. mlx-vlm stays pinned regardless (vendor-shadowing
  landmine, memory/omlx-mlx-vlm-064-verdicts). Adopt in a SIDE VENV first, one model, before fleet.

## Hermes-agent-trace MTP re-distill (Puzzle) — the LAST spec-decode lever, bar is HIGH
- **Bar (EXP-098):** sustained a1 >= **0.85** on served content — NOT the 0.65 distill gate. Even
  α=1 loses per-cycle unless full-accept cycles chain (pipelined full-accept floor 26.1ms/cycle;
  EMA≥0.75 gate must stay armed). Current head: a1 0.668 (EXP-096, trained on just 16 GENERIC
  self-gen seqs, 10 steps, early-stop overfit ceiling — agent traces never used).
- **Plan:** re-distill on real multi-turn agent traces (hermes/herdr; tool calls, JSON, code edits —
  the most draftable served distribution). Assets ready: trainer `_src/puzzle_campaigns/distill/`
  (chunked --resume), sidecars `~/.omlx/mtp_sidecars/puzzle75_mtp_*`, spec loop + EMA-gated pipeline
  `_src/puzzle_campaigns/spec_loop/spec_decode.py`, alpha harnesses `scripts/puzzle_mtp_alpha*.py`.
  **Missing asset: the trace corpus itself (ask the user where hermes/herdr traces live).**
- Success gate: a1 battery ≥0.85 held-out THEN spec ladder ≥1.10× overall (Law 17 harness rules).

## Mamba per-token scan-state exposure (Puzzle spec-decode fallback lever)
- The reject re-forward is the residual cycle cost after clone-on-verify; exposing per-token SSM
  scan states in the vendored mixer would allow partial rollback instead of full re-forward.
  EXP-098 judge estimate **+15-20% on a1≈0.6 categories**; invasive mixer surgery, unbuilt.
- Fund ONLY if the hermes redistill lands a1 in the 0.75-0.85 gray zone where reject tax is the
  remaining blocker.

## Puzzle-75B oQ48 productionization checklist (all cheap; ops-runbook §10 has the procedure)
1. :8000 admin reload (or restart) — pool globbed before the model landed; still undiscovered.
2. `model_settings.json` row: alias (`puzzle-75b`?), sampler (card: T=1.0/top_p=0.95), thinking
   default — user approves.
3. First warm decode: engagement grep `[PUZZLE-FUSE] mamba=40/40` (pool A auto-engages,
   kill switch OMLX_PUZZLE_DISABLE_FUSED_MAMBA=1).
4. Optional: batch≥2 on MLX 0.32 (B=4 = 168 agg tok/s measured, EXP-093) — gated on the shared
   nanobind-2.13 GLM-ext rebuild + fleet re-verify window (§MLX v0.32.0 above).
