> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# GLM-5.2 (glm_moe_dsa, 3.5bpw) — dossier

753B MoE with **MLA attention + DeepSeek-style sparse "lightning indexer" (DSA)**. This box runs a mixed-precision MLX build with **native Metal kernels** (the whole point of branch `glm5.2-native-kernels-v0.4.5`). Decode is measured **out** — every cheap lever is shipped or has a measured negative. Codenames: **DSA** = Deepseek Sparse Attention (indexer selects top-2048 keys); **MLA** = Multi-head Latent Attention (compressed KV latent); **MTP** = Multi-Token Prediction head (grafted, disabled).

## 1. Identity
- **Serving artifact:** `~/.omlx/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw`. `model_type=glm_moe_dsa` (a DeepSeek-V3.2 derivative). Disk **334.1 GB** (index total_size; ≈311 GiB `du`), **resident ~305-306 GB**. Native context **1 048 576 (1M)** (config), settings caps it at **400 000**. vocab 154880.
- Base weights = `avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw` (76 shards) + one grafted **MTP shard** from `inferencerlabs/GLM-5.2-MTP-MLX-Q4` (`tasks/glm_quant_matrix.md` footnote [1]).
- Source lineage: `zai-org/GLM-5.2` (native BF16, 1.507 TB) — see `tasks/glm_quant_matrix.md` for the 4-release quant comparison (native / FP8 / NVIDIA-NVFP4 / this LOCAL build).

## 2. Geometry (`GLM52_MTP_FORAY.md §1`, `tasks/glm_quant_matrix.md`)
- 753.33B total / ~40B active. hidden 6144, vocab 154880.
- **78 hidden layers: first 3 dense (`first_k_dense_replace=3`), 75 MoE** + one MTP/nextn layer (layer 78).
- MoE: **256 routed experts, top-8, + 1 shared expert**, `moe_intermediate=2048`.
- **MLA** attention: `kv_lora_rank=512`, `qk_rope_head_dim=64`, 64 heads.
- **DSA** sparse indexer: `index_topk=2048` keys, 32 index heads, head_dim 128 (the "lightning indexer"). Native training precision **BF16** (this matters — see §5 dtype trap).

## 3. Quant layout — per-tensor dtype split (`tasks/glm_quant_matrix.md`, LOCAL column — the primary source)
MLX affine, **`group_size=64` throughout**. "3.5 bpw" is a routed-expert-dominated weighted average, **not** a blended container.

| family | container | tensors |
|---|---|---|
| routed experts (256× gate/up/down, layers 3-77) | **affine 3-bit gs64** (lowest-bit; ~751B of params) | 225 |
| shared expert, all MLA projs, dense MLP (L0-2), MTP block | **affine 4-bit gs64** | 702 |
| embeddings, lm_head | **affine 6-bit gs64** | 2 |
| routers (`mlp.gate` + `e_score_correction_bias`), all RMSNorm, **main-stack DSA indexer** | **BF16 / F32** (kept full precision) | — |

Bit tally over 929 quantized tensors (`tasks/glm_quant_matrix.md` footnote [1]): 3-bit ×225, 4-bit ×702, 6-bit ×2. **MTP layer 78** is affine 4-bit, grafted from `inferencerlabs/GLM-5.2-MTP-MLX-Q4` (attn/switch_mlp/shared/indexer/eh_proj all 4-bit; norms+router BF16; loaded via omlx MTP patch, not the base quant dict). Decode bandwidth floor ~16 ms of 47.6 ms/token; `qmm`/`gather_qmm` run 83-105% of bandwidth (`memory/omlx-glm52-decode-opts.md`).

## 4. Serving (`~/.omlx/model_settings.json` → `GLM-5.2-Alis-MLX-Dynamic-3.5bpw`)
```
max_context_window 400000, temperature 1, top_p 0.95, enable_thinking true,
active_profile_name "glm52", turboquant_kv_enabled false (→ fp16 MLA KV),
specprefill_enabled false, mtp_enabled false, vlm_mtp_enabled false,
is_default false, trust_remote_code false
```
- **Golden env** (fleet-wide): `MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000` — bumped GLM short **20.8→23.1 tok/s** (`memory/omlx-glm52-decode-opts.md`; `GLM52_MTP_FORAY.md §11`).
- **Native kernels are a hard dependency:** they only load when the `_ext.so` is built against **nanobind 2.12.0 (ABI v19)** matching mlx-metal 0.31.2; nanobind 2.13.0 (v20) → silent fallback to pure-MLX (slower). Rebuild + pre-load ABI probe in `memory/omlx-glm52-native-kernels.md`; kill switch `OMLX_GLM_DISABLE_NATIVE=1`. See metal.md / preflight.md.
- **KV:** fp16 MLA latent. On 512 GB, fp16 KV fits even 1M context (306 GB weights + ~90 GB KV < ~464 GB cap) — the int8 MLA-KV path is built but OFF (§8). Cross-ref kv-cache.md.

## 5. Engagement grep set (`omlx/patches/glm_moe_dsa/`)
- Native kernels loaded: log `GLM MoE DSA native kernels available from <path>`; `is_native_available()` True; `12/12 tests/test_glm_moe_dsa_patch.py` pass (skips when native off → skip==fallback signal).
- Decode opts (`decode_kernels.py`, kill `OMLX_GLM_DISABLE_DECODE_OPT=1`): fused s==1 indexer-scores + `mh_qmm_m1` embed_q + flash-decode sparse-MLA (gated `K>=98304`, tune `OMLX_GLM_FLASH_DECODE_MIN_K`).
- Fused indexer counter: `[GLM-DKO] indexer-scores ENGAGED dtype=bfloat16` / `[GLM-DKO] ... BAIL ...` (`decode_kernels.py:139-142`).
- **WATCH (dtype trap, the live-path law):** the fused decode path was fp16-gated while the live model runs **bf16** → the kernel NEVER engaged live until templatized to `{half,bfloat}` (`tasks/todo.md` "GLM SAME-DISEASE FIX"). Any GLM fast-path must show a live counter (profiling.md).

## 6. Measured speed (batch-1)
| context | native-kernel decode (pre-golden) | note |
|---|---|---|
| short (~1k) | 20.8 → **21.7** tok/s (+4.3%, decode opts) | golden env → **23.1** |
| 14.5k | 19.5 → 19.9 (+2.1%) | |
| 57.7k | 18.2 → **18.5** (+1.6%) | |
Source: `tasks/todo.md` "Live A/B", greedy-identical. MTP-off baseline (`GLM52_MTP_FORAY.md §1`): 21.0 short / 19.6 @16k / 18.2 @64k. **Prefill at ceiling** everywhere (MoE 89% peak, projections 89%, indexer 24.7 TF/s ~88% flat in K, sparse_mla topk-bound). Native kernels give **1.57× prefill** vs dense (117.7 s vs 184.7 s @20k) AND make long context feasible — native completes @~53k where **dense OOMs @~44k** (`memory/omlx-glm52-native-kernels.md`). GLM L=1 decode @golden = 43.3 ms/token: 53.6% GPU-wait + 17.5% encode + 12.5% python (`GLM52_MTP_FORAY.md §11`). vs the release's published GLM-5.2-oQ4 (418 GB): our 3.5bpw (306 GB) is ~comparable prefill, **+27% decode** (bandwidth-bound), −112 GB memory.

## 7. Quality
Measured accuracy benches (gsm8k/mmlu/arc) for this 3.5bpw build were **not run** in the corpus — **GAP**. Known: **3.5bpw average quality < the release oQ4 (4-bit)** build (`memory/omlx-glm52-native-kernels.md`), the deliberate trade for −112 GB and faster decode. Synthetic layer parity vs native reference: max_abs 2e-3, topk overlap 99.95% (fp16 tie noise). Long-context reasoning/NIAH ladders in `tasks/todo.md` are MiniMax-M3's, not GLM's — do not attribute them here.

## 8. Levers — LIVE vs DEAD vs PARKED
- **LIVE:** native DSA Metal kernels (sparse-MLA, exact-block, indexer, weighted-sum); decode opts (+2-4%, `OMLX_GLM_DISABLE_DECODE_OPT`); golden env; 4-bit V-up kernel (committed, but <1% e2e — real but marginal).
- **BUILT then ABANDONED:** **int8 MLA-KV cache** (`Int8MLALatentCache`, committed capability, default OFF) — 42% smaller KV but **~2× slower prefill** (dequant-on-read death-spirals the throttle through a 5-clamp stack); fp16 already fits 1M on 512 GB so int8 is only worth it on a smaller box or >1M ctx (`memory/omlx-glm52-int8-mla-kv.md`). mxfp8 latent = same dequant wall (no fp8 MMA on Apple), don't.
- **DEAD (measured negative — see dead-levers.md, don't re-litigate):** **MTP self-speculation** (0.85× — verify L=2 is ~1.9× a decode from diffuse MLX op overhead, `GLM52_MTP_FORAY.md`); fused router kernel (`mx.compile`'d select already wins 36µs); custom MoE gemv (gather_qmm 83-84% bw); dual-stream shared-expert overlap (23% slower + numerically unsafe); flash-decode <64k (MLX gather+SDPA SLC-hot wins); `mx.compile` of verify (1.06×); native MoE 3-bit kernel (parity by construction); block-union prefill.
- **PARKED:** fused GLM-DSA decoder-layer **mega-kernel** — the only thing that could cross ≥1.3×, months-class (`future-campaigns.md` #3); batch≥2 (`future-campaigns.md` #2).

## 9. Known bugs / watch-items
- **GLM-DKO mask-gate skip:** the fused decode path is gated on `mask is None` (`deepseek_v32.py:271`); restored-prefix / batch decode carries a bool mask → silent skip at long ctx with **no BAIL log** (the log lives inside the kernel fn, below the gate). Third instance of "fast-path gate upstream of instrumentation" (`tasks/todo.md`). Follow-up: census the indexer call site (mask None vs array per step @58k).

## 10. Conversion provenance
Third-party MLX Dynamic 3.5bpw build (`avlp12`) + grafted 4-bit MTP layer from `inferencerlabs/GLM-5.2-MTP-MLX-Q4`. Native-kernel enablement (nanobind pin + ABI probe + kill switch) and the decode/V-up kernels are ours (this branch). Full per-release quant map: `tasks/glm_quant_matrix.md`; conversion mechanics: conversion.md.

## 11. GLM-5.2-oQNVFP4 (second build, 2026-07-06 — future-campaigns #1 EXECUTED)

`~/.omlx/models/unigilby/GLM-5.2-oQNVFP4` — 427.75 GB disk / **398.9 GiB resident** / 3130 tensors / 77 shards, converted from `nvidia/GLM-5.2-NVFP4` (465 GB, source kept at `~/glm52-nvfp4-src/`) by `omlx/tools/oqnvfp4_glm_convert.py`. Recipe: routed experts **byte-exact NVFP4 gs16** (ModelOpt-formula dequant max|diff|=0.0) pre-fused `switch_mlp.gate_up_proj` + f32 ts sidecars; BF16 shell → **affine8-gs64** (MLA projs, shared experts, dense L0-2, embed/lm_head); **kv_b pre-absorbed at convert time** → quantized `embed_q (64,512,192)`/`unembed_out (64,256,512)` (avlp12 disk convention, native V-up/embed kernels ENGAGE); indexer/router/norms BF16; **MTP layer 78 + input_scale dropped**. Runtime ts-fold = `omlx/patches/glm_moe_dsa/nvfp4_ts.py` (sanitize wrap, per-instance `_TsSwitchGLU` swap; gate_up_ts pre-activation + down_ts on output rows — one hook covers native weighted-sum + both score paths); gated on config `omlx_moe_nvfp4_ts` in `model_loading.py`; kill `OMLX_GLM_DISABLE_NVFP4_TS`; grep `[GLM-TS] ... ENGAGED`. Gates: `scripts/glm_oqnvfp4_gate_offline.py` (G1 census exact + G2 parity, ALL PASS) + `scripts/glm_oqnvfp4_g3_load.py`.

**Measured (2026-07-06 same-session A/B vs 3.5bpw, golden env, t1t256):**
| | oQNVFP4 | 3.5bpw |
|---|---|---|
| decode short / 16k / 64k | 16.25 / 16.46 / 19.01* | 22.74 / 23.09 / — |
| gsm8k n=60 / mmlu n=250 / arc n=150 (paired, temp0) | 93.3 / **82.0** / 94.7 | 95.0 / **85.6** / 96.7 |
| 16k cold prefill | 125 s | 92 s |
| resident | 398.9 GiB | ~306 GB |
(*64k rung restore-variance suspect — T1 leg carries SSD-restore jitter; re-probe if load-bearing.) Decode = 0.71× of 3.5bpw, **napkin-exact** (≈31.4 vs ≈21.5 GB token-reads, law 3). 400k fp16 KV fits: enforcer ceiling 491.7, model 398.9 → ~93 GiB slack (KV ≈ 93 KB/token: 78×1152 latent + 21×256 indexer-full layers only). Settings entry mirrors 3.5bpw (400k, fp16 KV, mtp off).

**VERDICT AMENDED (EXP-071, production-settings A/B): quality is a WASH — 3.5bpw stays production on EFFICIENCY grounds only.** With NO request-side overrides (server defaults: temp 1.0, top_p 0.95, thinking ON), seed-paired mmlu n=50: oQNVFP4 **86.0** vs 3.5bpw **84.0**, zero truncations — the temp-0 −3.6pp FLIPS sign. The temp-0/8-token protocol structurally penalized the reason-first model (the RCA's truncation-artifact hypothesis, confirmed). Bench-protocol law reinforced: **for thinking models, quality A/Bs must include a production-settings leg (no sampler/thinking overrides) before a verdict.** Roster call unchanged in action: 3.5bpw is −122 GB and +40% decode at tied quality.

**Earlier verdict for the record (EXP-069 + EXP-070 RCA, temp-0 protocol): 3.5bpw stays production; pipeline EXONERATED by measurement; the gap is noise-inflated with at most a small real "error-character" effect.** The quality battery is uniformly against NVFP4 (−1.7/−3.6/−2.0pp) but every delta is sub-2σ (mmlu margin ≈9 items, McNemar z≈1.8; a live paired 40-item slice TIED 31–31). RCA workflow findings (5 investigators + adversarial judge): (1) avlp12 is **plain static-recipe RTN affine** — NOT calibrated/tuned (all 225 expert tensors exactly 3-bit gs64, codes match stock `mx.quantize` RTN to 3 decimals; the earlier "sensitivity-tuned" framing was WRONG). (2) Our serving path measured clean end-to-end: full ts-fold chain 0.51% output rel-RMS vs fp32-dequant ideal (M3-style fp32-through-activation chain: 0.39% — difference 20–40× below the ~10% weight-error floor); gather_qmm nvfp4 exact at GLM shapes (rel-RMS 0.0017); router/norms/bias tensors **byte-identical** across builds; cross-build expert distance 0.210 vs 0.217 expected for independent RTN — independently corroborates the repack. (3) Paradox: NVFP4 gs16 has HALF the weight-RMS error of affine3-gs64 (0.105 vs 0.190) yet loses — surviving hypothesis: NVFP4's symmetric zero-point-free grid + block-correlated E4M3 errors don't preserve group means (affine's bias does), a property weight-RMS doesn't rank; plus a minor harness artifact (max_tokens=8 truncates GLM's reason-first answers; 1 of 2 NVFP4-only losses in the live slice was truncation). (4) The M3 "win" this contrasts with is equally under-powered — the flip may partly dissolve at higher n. Discriminating follow-ups (ranked): stage-2 256-tok re-ask (script banked), logit-KL vs a common high-precision reference (~200 prompts, kills or confirms error-character), mmlu n≥1000 McNemar, expert-swap ablation, re-power the M3 battery. Build + source parked pending disk call. Untested residue: 400k soak, `--shared-nvfp4`. **Watch-item:** server died SILENTLY (no shutdown log, no signal line) at 20:45:25 after ~500 rapid short requests + one pool swap, 3.5bpw loaded — reliability signal, cause unknown (cf. PR #2103 boundary-snapshot retention).

## 12. Alis 4.5bpw era + int8 MLA-KV (2026-07-07/08 — CURRENT PRODUCTION)
**Flagship:** `GLM-5.2-Alis-MLX-Dynamic-4.5bpw` (~424GB, avlp12). oQNVFP4 build parked (quality WASH).
**Indexer RoPE fix (EXP-073, all GLM builds):** indexer uses NON-interleaved RoPE (`indexer_rope_traditional=False`)
+ LayerNorm eps 1e-6 — HF ignores the config's interleave flag; vendored ModelArgs defaults retrofit old checkpoints.
**int8 MLA-KV (EXP-077/080-082, LIVE):** latent-only int8-gs64 past start=4096; the historic "2x prefill" was
omlx scheduler clamps, NEVER the quant (fork A/B: dequant ~2%). Decode = gather-then-dequant fast path (2048
rows not K) -> 19.5 tok/s @64k BEATS fp16 18.5; prefill 1.02x fp16 @16k; L>1 q8 sparse kernel wired (bit-identical,
no wall win — deep prefill is throttle/indexer-bound, but keeps transient flat). SSD blocks persist int8-NATIVE
(element-count format dispatch; legacy fp16 restores via streamed requant). Counters: `[INT8KV]`, `restore format=`.
**Quality certification (EXP-085, int8 LIVE, thinking on):** gsm8k 93.3 (60) / arc 96.7 (120) / mmlu 95.0 (200) —
matches/beats every 3.5bpw reference. int8 KV = quality-neutral. logCoT 26/51k: 12/12 capability (one 51k
auto-fail was a degenerate-loop artifact; rerun-passed in 1025 tok).
**MTP (EXP-076/078/079, FINAL):** native head, chained K=2, lossless; a1=92%/2.70 tok-cycle on code, net +3-7%
short-ctx. The fork's own MTP measures 0.82x on this box (card's +15% unreproducible). CONTEXT-GATED:
`OMLX_MTP_MAX_CONTEXT` default 2048 (16k = 0.6x; 8k prose = 1.79 cycle ≈ break-even — gate stays).
**Depth (EXP-081/084):** 256k end-to-end PROVEN (prime 2086s, decode confirmed; restore streamed per-layer,
peak 469 vs hard 480). 128k decode: 15.6-22.7 (probes disagree — confirm pending). Marginal prefill declines
126->68->63 tok/s (64k->128k->256k): indexer O(N*K) physics, NOT chunking (chunks were 1024 throughout);
tier-clamp removal = bounded ~10-20% candidate only.
**Serving env:** golden env + `OMLX_MTP_DRAFT_K=2 OMLX_MTP_VERIFY_FAST=1`; settings: int8_mla_kv_enabled/bits/start,
max_context_window 600000, memory guard custom 506GB (soft .93/hard .97); preflight_guard OFF fleet-wide.
**Fleet math:** 4x256k agents = 484GB — fits under hard 490.8 (int8 only; fp16 would need +60GB).
