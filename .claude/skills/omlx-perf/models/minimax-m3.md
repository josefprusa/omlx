> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# MiniMax-M3 (minimax_m3_vl, oQNVFP4-fs5) — dossier

428B MoE **vision-language** model (23B active) with **MiniMax Sparse Attention (MSA)**. The fastest of the ≥400B-class models (GLM/Ultra/M3) at short context — no longer the fleet's fastest outright: Puzzle-75B-oQ48 decodes 54.3 tok/s vs M3's 27.5 (`models/nemotron-puzzle.md`, EXP-091). **Production build is `oQNVFP4-fused`** (2026-07-06 — fused-shared + uniform affine8 attention, no 5-bit band; chosen over fs5 for higher attention precision + cleaner recipe at a measured ~5% decode cost, §6/§7). `fs5` (fused + 5-bit attn band) was the ~5%-faster prior production, **deleted 2026-07-06** (regenerable from the T7 NVFP4 source with `--attn5-layers 17-44` if the 5% is ever wanted). Also the home of the **EAGLE-3 drafter** (banked, break-even at batch-1) and a long "silent sparse" debugging saga (the live-path law's origin). Codenames: **MSA** = MiniMax Sparse Attention (per-layer index → 128-block pool → top-16 blocks); **fs5** = fuse-shared-nvfp4 variant 5; **oQ** = our NVFP4/mxfp8 byte-repack converter.

## 1. Identity — variants (`oQNVFP4-fused` is PRODUCTION; all other builds DELETED 2026-07-06)
| artifact | location | role |
|---|---|---|
| `MiniMax-M3-oQNVFP4-fused` | 230 GB internal (`~/.omlx/models/unigilby/`) | **PRODUCTION** — fused shared + uniform affine8 attn (no band); the converter's new fused default. Sole MiniMax build on internal (~5% slower than the deleted fs5 at tied quality, chosen for attn precision, §6/§7) |
| NVIDIA source `MiniMax-M3-NVFP4` | 233 GB, T7 `omlx-quant-work/` (88 files) | original NVFP4 weights — the archive; rebuild ANY variant from here via `oqnvfp4_convert.py` |
| ~~`oQNVFP4-fs5`, `-fs`, `oQNVFP4`, `oQ4`~~ | — | **DELETED 2026-07-06** (freed ~917 GB for the GLM-5.2-NVFP4 conversion). Regenerable from the T7 source: fs5 = `--fuse-shared-nvfp4 --attn5-layers 17-44`; `-fs`/`-fused` = fused default (no flags); oQ4 = the int4 recipe. Settings rows removed. |

`model_type=minimax_m3_vl`, vocab 200064, **60 layers**, hidden 6144, native context **1 048 576 (1M)** (config). Vision tower present (VL); text path is the measured hot path.

## 2. Geometry (`tasks/todo.md` M3 section; `tasks/eagle3_build.md §PHASE 2`)
- 428B total / 23B active. **60 layers: 3 full-attention (layers 0/1/2) + 57 MSA-sparse.**
- MoE **fs5 fused mode**: 129-expert switch, shared expert folded at index 128; swiglu `a=1.702, L=7.0, b=1.0`; `nvfp4_ts=True`.
- MSA per layer: index scores over K → 128-block pooling → top-16 blocks (`index_topk`), engage floor `OMLX_M3_SPARSE_MIN_K=4096`.
- Attention head_dim 256 (GLM sdpa256 patch inert here: `head_dim==256 && L>1` guard). `rope_theta=5e6`, silu, untied.

## 3. Quant layout — per-tensor dtype split (fs5 recipe, `tasks/todo.md` "oQ-NVFP4 GOAL MET")
fs5 = **fuse-shared-nvfp4 + attn5-layers 17-44** (an oQ4 sensitivity mirror) + runtime `weight_scale_2` carry.

| family | container |
|---|---|
| routed experts | **NVFP4 gs16** — NVIDIA's calibrated experts, **byte-repacked** into MLX (proven bit-exact both formats, max|diff|=0.0) |
| shared expert | **fused into the nvfp4 switch** (fs5); + per-expert `weight_scale_2` runtime fold |
| attention (layers 17-44 sensitive band) | oQ4 recipe: **8-bit early / 5-bit late**; routers+norms **bf16** |
| routers, norms | bf16 / f32 |

Runtime **ts-fold** (`nvfp4_ts`): `weight_scale_2` folded into routing weights for down_proj + a post-mul on gate_up (fp32), exact SwiGLU carry (`language.py:1983` folds `down_ts` into scores, max|diff|=0.0). Kill/attribution `OMLX_M3_DISABLE_NVFP4_TS`. Byte-ceiling not separately tabulated for M3 — GAP (decode is dispatch/host-bound, not pure-bandwidth, §6/§8).

## 4. Serving (`~/.omlx/model_settings.json` → `MiniMax-M3-oQNVFP4-fs5`)
```
trust_remote_code true, temperature 1.0, top_p 0.95, top_k 0, min_p 0.1,
force_sampling true, vlm_mtp_enabled FALSE,
vlm_mtp_draft_model "$OMLX_COLD_STORAGE/omlx-quant-work/MiniMax-M3-EAGLE3",
vlm_mtp_draft_block_size 2
```
- **Production launch line** (`memory/omlx-glm52-decode-opts.md`, final state): `env OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000 uv run omlx serve`. The **MB=4000 cap (~one M3 layer of weights)** is load-bearing — it prevents the Metal command-buffer timeout that an earlier note "REQUIRES OPS=500" was working around (that OPS=500 note is superseded by the golden-env campaign; `tasks/todo.md` "CAMPAIGN CLOSED"). `OMLX_M3_DEBUG_PATH=256` keeps the census permanently on for engagement visibility.
- **force_sampling=true overrides request temp to 1.0** before the EAGLE temp0 guard → the drafter never engages in normal serving (§8/§9).

## 5. Engagement grep set (`omlx/patches/mlx_vlm_minimax_m3_compat/.../language.py`)
- Census (permanent): `[M3CENSUS] step=<N> cache=MiniMaxM3BatchKVCache zeropad=True offset=<K> counts={...}` — healthy live decode shows `fused_hit=57`, `fused_topk=57` (all 57 sparse layers), maskless compact. `fused_none`/`scores_fallback` counts must be **0** (they were 57/57 during the fp16-gate bug — the dtype trap, §9).
- Packed projections: census `pack_full=56 pack_qkv=3 pack_none=1` (matches config; values log-confirmed 2026-07-04 — grep the LIVE server log, the emitter keys may have drifted in the tree).
- Native ext: `minimax_msa_topk` (the only native kernel; attention is MLX by design). No ABI-probe line = silent fallback (nanobind lesson, preflight.md).
- EAGLE verify (when drafter on): `verify_ok=57/57`; `[M3DBG]` one-shot path dump.
- Knobs: `OMLX_M3_SPARSE_MIN_K=4096`, `OMLX_M3_COMPACT_MAX_DENSITY`, `OMLX_M3_DISABLE_FUSED_INDEX`/`_FUSED_TOPK`/`_PACKED_PROJ`, `OMLX_M3_ENABLE_FLASH_SPARSE` (opt-in, slower).

## 6. Measured speed (batch-1, census-verified fused_hit; card ref 21.7 decode / 214 prefill)
| context | oQ4 (final) | oQNVFP4-fs5 (production) |
|---|---|---|
| short | **27.5** / golden **28.4-28.9** | 27.09 |
| 9.5k | 22.5 | 21.66 |
| 16k | **22.0** / golden 24.3 | 21.64 |
| 66k | **19.1** | — |
| 103k / 128k | — | 18.0 @103k / 17.48 @128k; golden 18.8 @126k |
Prefill **342 tok/s** (`tasks/todo.md` "M3 PUSHED FURTHER"; the **+25%** was the 273→342 optimization gain, and 342 is **+60% over the card's 214**). fs5 is ~1.6-3.7% slower decode than oQ4 but better quality (§7). GLM's sdpa256 patch is inert here. Dense-forced crossover: MSA sparse wins ≥10k (dense 13.1 @69k — MLX sdpa has no GQA K/V sharing, reads 16×). Decode is **host/dispatch-bound**, not pure-bandwidth (see §8, profiling.md).

**2026-07-06 same-session A/B** (`-fused` vs `fs5`, live, temp-1): decode short **27.26** (fused) vs **28.54** (fs5); 16k **25.43** vs **27.08** → **fs5 ~5-6% faster** (the 5-bit band's speed win, confirmed same-session, consistent with the 07-04 ladder fs 25.75→fs5 27.09). ⚠️ The oQ4/fs5 numbers in the table above are **2026-07-04 and NOT same-session comparable** — the decode tree got materially faster since (fs5 16k **21.64→27.08**); always A/B in ONE session (the stale-ledger trap that misread fs5 as "no penalty" mid-campaign, gotchas.md).

## 7. Quality (`tasks/todo.md` "oQ-NVFP4 GOAL MET"; acc_bench, serial)
| bench | oQ4 | oQNVFP4-fs5 | default oQNVFP4 |
|---|---|---|---|
| gsm8k | 92.7 | **95.3 (+2.7pp)** | — |
| mmlu | 79.3 | 81.3 (+2.0pp) | 83.0 |
| arc | 94.3 | — | 96.7 (+2.3pp) |

Long-context: **NIAH 12/12 to 103k** (3-needle @13/26/51/103k), multi-hop reasoning **18/18** across 26k/51k/103k, temp0 (`tasks/todo.md` "NIAH LADDER" / "REASONING @103k"). The lossless NVFP4 repack + exact ts-fold *improved* quality over oQ4.

**2026-07-06 same-session A/B** (mmlu 250 / arc 150 / gsm8k 30, temp-1 sampling): `-fused` mmlu **80.8** / arc **92.0** / gsm8k **96.7**; `fs5` mmlu **78.8** / arc **94.7** / gsm8k **96.7** → quality a **statistical wash** (mmlu +2 fused, arc +2.7 fs5, gsm8k tie — all within n/sampling noise). Dropping the 5-bit attn band recovers **no measurable quality** → "quality recovery" is a dead lever (`dead-levers.md`, EXP-066). Production chose `-fused` anyway for the attention-precision margin + cleaner uniform-affine8 recipe, accepting the ~5% speed (§6). To get fs5's speed back on a `-fused`-style build, add `--attn5-layers 17-44` at convert time (free ~5% at neutral quality).

## 8. Levers — LIVE vs DEAD vs PARKED
- **LIVE:** MSA sparse (≥4096); fused index kernel + fused topk kernel + packed q/k/v/index projections (all bit-exact, `_PACKED_PROJ` cumulative @16k 19.6→22.0 +12%); nvfp4 ts-fold; golden env; census.
- **DEAD (measured negative, dead-levers.md):** EAGLE-3 spec decode at batch-1 (break-even — verify L=2 GPU-wait 1.83×@5k / 2.12×@16k, MoE expert-read-bound, K-sweep flat, alpha=1 can't beat it); flash-sparse-SDPA (opt-in, 64 TGs under-occupy the 80-core GPU, slower); int8/mxfp8 index-KV as bolt-on (~0.7 ms/tok slower, dispatch-bound); MLX `MAX_ACTIVE_TASKS` cap raise (flat); `mx.compile` decode (flat at batch-1 — host already overlapped, wall = GPU + per-op dispatch); **DQ8 dense-shell bake** (EXP-065, 2026-07-06: **PROVEN NO-OP** — `oqnvfp4_convert.py` emits an affine8-gs64 shell BY CONSTRUCTION, so oQNVFP4/fs5 already ship **130/130** text-shell Linears (q/k/v/o+index+dense-mlp+shared+lm_head+embed) as affine8-gs64, **0 GiB bf16 text weight to bake**; the cost-model's 1.80 GiB is ~1.61 GiB vision-tower bf16 [off the text hot path] + affine8 sidecars miscounted. Ultra's 66→47 GB win does NOT transfer; decode is dispatch-bound → `scripts/m3_dq8_offline_gate.py`, `scripts/dq8_costmodel.py`, `dead-levers.md`).
- **PARKED (banked assets):** EAGLE-3 **drafter** (vendor-grade 86.9%/82.6% accept, `future-campaigns.md` #2) → pays only at **batch≥2**; MoE verify-batching; the residual ~2× diffuse dispatch @16k → mega-kernel (`future-campaigns.md` #3); mxfp8 KV capacity play (`future-campaigns.md` #4).
- **BATCH>1 (already hardened, parked for concurrency):** M3's B>1 decode ships physical-positions correctness, per-row mask gather, and zero-pad flag propagation (`tasks/todo.md:390-392`). BUT the compact gate (`original_mask is None and B==1`) never fires under batched decode → it falls to **slow masked-dense** (894µs/layer vs 741 compact / 620 dense @3.7k, `memory/omlx-glm52-decode-opts.md:79-82`); and `_build_sparse_mask` (`language.py:913`) had a logical-q-vs-physical-k bug that dropped left-padded rows' newest keys (fixed; comment `language.py:1626-1643`). Only measured concurrent datapoint: **4-way padded = 11.5 tok/s aggregate incl. prefills** (`tasks/todo.md:388-389`) — the fleet's sole concurrent number. Revival = batch≥2 (`future-campaigns.md #2`).

## 9. Known bugs / watch-items
- **SSD restore-fallback class:** the prefix-cache restore path could rebuild a plain `KVCache` for metadata-less blocks (`prefix_cache.py:1946/2500`), which would drop sparse. **Empirically refuted** (A18: restored 16k prefix kept `MiniMaxM3BatchKVCache` + `fused_hit`) but the code class exists — re-test if you touch prefix-cache restore. The infamous "19.6 @3.6k sag" that chased this was **bench contamination** (script requested GLM) — verify which model the server LOADED per bench (grep log).
- **fp16-gate dtype trap (fixed):** the fused index kernel required fp16 while the live model runs bf16 (`torch_dtype`) → `fused_none=57/57`, O(K) fp32 fallback, anomalous slope. Fixed dtype-generic; **this is the origin of the live-path-verification law** (profiling.md).
- **vlm_mtp disabled in settings:** the EAGLE drafter is dead weight in normal serving (force_sampling=true skips it at temp0); keep OFF until a batch≥2 or rejection-sampling revival (`future-campaigns.md` #2/#5-adjacent).

## 10. Conversion provenance
`omlx/tools/oqnvfp4_convert.py` — NVIDIA's calibrated NVFP4 experts byte-repacked into MLX (both nvfp4 and mxfp8 proven byte-identical, max|diff|=0.0, lo-nibble-first), `--attn5-layers 17-44` (oQ4 sensitivity band), fuse-shared. Two sanitize layers exist (vendored model + `engine/vlm.py:_pack_mlx_unpacked_moe_weights`) — the engine one needed the config-driven skip. Full mechanics in conversion.md; EAGLE-3 build log `tasks/eagle3_build.md`.

## Overnight battery + int8 verdict (2026-07-08)
**Intelligence (EXP-085, thinking on):** gsm8k 93.3 (60) / arc 96.7 (120) — DEAD HEAT with GLM-4.5bpw at
4-6x the speed (11.8 vs 47-68 min legs); mmlu 90.5 vs GLM 95.0 (the separation: GLM = knowledge, M3 = speed).
logCoT shared-prefix (harder variant, cross-family distractors): 18/18 PERFECT at 26k/51k/103k; non-prime
questions 8-19s each (prefix cache).
**Vision:** CONFIRMED live (shapes/colors/positions + text verbatim, 51s wall incl. 40s model swap).
**int8 KV: NOT-MLA (recon 2026-07-07, banked in future-campaigns.md).** M3 caches plain GQA K/V
(60L x 4KVh x 128, ~181 KB/tok fp16 incl. lightning-indexer keys) — Int8MLAKVCache does NOT drop in.
Blocker: CompiledDecoder reads raw kv.keys/values buffers. Weak capacity case anyway: no ctx cap
(inherits 1M), 4x256k = 420GB fits easily. Revisit only on a real wall.
**Fleet role:** speed king of the ≥400B class + vision + long-ctx reasoning champion; swaps with GLM in ~40-60s (never co-resident: 654GB). Puzzle-75B-oQ48 is now the fleet-fastest outright (54.3 vs 27.5) but is a smaller weight class — not a substitute for M3 vision/long-context.

## thinking_mode semantics (measured 2026-07-08)
Template renders differ for real: enabled pre-opens <mm:think> (forced), disabled pre-closes (impossible),
adaptive = no pre-fill + soft instruction (model self-decides). Measured policy under adaptive: NEVER fully
skips (181ch even on "2+2"); modulation is in totals — trivial 53 vs 89 tok (~40% saved), hard 594 vs 846
(thinks LESS than forced, still correct). Mapping doctrine: effort minimal/low/medium -> adaptive (economy,
not off), none -> disabled (only true silence). MONSTER TEST (novel 11-clue XOR/IFF logic grid ->
6-stage arithmetic chain, verified unique, FINAL=395): BOTH modes correct; adaptive scaled UP past
enabled (5961 vs 4532 think chars, 3003 vs 2450 tok) — adaptive is bidirectional self-budgeting
(saves ~40% on trivial, spends MORE than forced when it judges the problem hard). Full spectrum:
trivial 181 chars -> monster 5961. Harness: scratchpad/monster.json.
