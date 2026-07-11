> Verified 2026-07-05 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Nemotron-3-Ultra-550B — dossier

The fleet's largest model and the campaign's headline win: **7.68 → 13.08 tok/s (+70%)** with quality held. Production artifact is the DQ8-baked checkpoint. Codenames: **DQ8** = load-time/baked dequant-to-affine-int8 of NVIDIA's bf16-upcast shell; **oQNVFP4** = our lossless NVFP4 byte-repack into MLX (see conversion.md).

## 1. Identity
- **Source:** `nvidia/Nemotron-3-Ultra-NVFP4` — 352 GB, 113 shards (`tasks/oqnvfp4_nemotron.md §12`).
- **Serving artifact:** `~/.omlx/models/unigilby/Nemotron-3-Ultra-oQNVFP4-dq8`. `model_type=nemotron_h` (NemotronH mamba2/MoE/attention hybrid). Disk **327.2 GB** (safetensors index total_size), 1745 tensors; **resident 305.08 GB** (`tasks/todo.md` "DQ8 CHECKPOINT PRODUCTIZATION"). Native context **262 144 (256k)** (`config.json max_position_embeddings`).
- **NOT in `~/.omlx/model_settings.json`** — the baked checkpoint landed on disk 2026-07-05 23:16, after settings.json was last written (17:10). It serves on engine-pool defaults; GAP: no persisted per-model settings row, no confirmed `model_alias` (Super's is `nemotron-3-super`).

## 2. Geometry (`tasks/ultra_speed.md §2`, verified vs config.json)
- 550B total / ~55B active (A55B). hidden 8192, vocab 131072.
- **108 layers = 48 mamba2 + 48 MoE + 12 attention** (`layers_block_type` list; `num_hidden_layers=None` in config — a dialect trap, see §10).
- Mamba2: 256 heads × 64 dim (intermediate 16384), n_groups 8, ssm_state 128, conv_kernel 4, conv_dim 18432, in_proj 8192→35072.
- Attention: **GQA 64 q-heads / 2 KV heads, head_dim 128, NO RoPE**.
- MoE: **512 experts, top-22**, latent 2048, expert ffn 5120, shared expert 8192↔10240, router n_group=1 (`group_expert_select` group-branch never runs → routing is one `@mx.compile`'d sigmoid+argpartition, no host round-trip). Activation **relu²** (no gate).

## 3. Quant layout — per-tensor dtype split (campaign law: never a blended bpw)
Byte accounting from `tasks/ultra_speed.md §2` "Bytes read per token"; NVIDIA map from `tasks/oqnvfp4_nemotron.md §12`.

| tensor family | container (production baked) | note |
|---|---|---|
| routed experts (512×fc1/fc2, top-22) | **NVFP4 gs16** (289.9 GB) — the only weights NVIDIA quantized | untouched by DQ8; format fully exonerated (§8) |
| mamba in_proj/out_proj (96 mods) | **affine8 gs64** (DQ8 stage A) | NVIDIA shipped these FP8; we upcast at convert then DQ8 removes the tax |
| shared experts + latent projs (192) | **affine8 gs64** (DQ8 stage B) | NVIDIA-excluded → isolated A/B was mandatory |
| attention **q_proj/o_proj only** (24) | **affine8 gs64** (DQ8 stage C) | **k/v EXCLUDED** — M0 measured 0.87-0.91× = loss on the skinny 8192→256 shape |
| lm_head | **affine8 gs64** (DQ8 stage D, last) | isolated ablation, touches logits |
| router gate, embeddings, all norms, conv1d, switch_mlp scale sidecars | **bf16 / f32** — NEVER quantized | |

**Bandwidth ceiling worked once** (`tasks/ultra_speed.md §2`): pre-DQ8 pure-weight reads **78.123 GB/token** (+0.890 GB SSM-state/conv/KV) → **10.36-10.48 tok/s at 819 GB/s** (we were at 74% of that = 7.68). DQ8 shrinks the dense pool 65.263→~48 GB → total ~47 GB/token → ceiling ~17 tok/s; measured 13.08 (host + diffuse overhead eat the rest, §6/§8).

## 4. Serving
- **Golden env** (fleet-wide, env-setup.md): `MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000` (`tasks/ultra_speed.md:4`).
- **Ultra launch adds four DQ8 switches** (`tasks/todo.md` "ULTRA SPEED CAMPAIGN"): `OMLX_ULTRA_DQ8_MAMBA=1 OMLX_ULTRA_DQ8_MOEDENSE=1 OMLX_ULTRA_DQ8_ATTN=1 OMLX_ULTRA_DQ8_LMHEAD=1`. On the **baked** checkpoint these are **INERT** — "baked checkpoint detected" is expected, the tensors are already int8 on disk. They matter only if you ever serve the un-baked `oQNVFP4` build (load-time DQ8 probe path, ~1-3 min extra load).
- Cross-ref ops-runbook.md for the full launch line; env-setup.md for golden-env rationale.

## 5. Engagement grep set (`omlx/patches/nemotron_h_dq8.py`, `omlx/patches/nemotron_ultra_decode/`)
Baked-checkpoint healthy load prints **four** stage lines + decode counters:
```
[ULTRA-DQ8] mamba    baked checkpoint detected (96 modules); ...
[ULTRA-DQ8] moedense baked checkpoint detected (192 modules); ...
[ULTRA-DQ8] attn     baked checkpoint detected (24 modules); ...
[ULTRA-DQ8] lmhead   baked checkpoint detected (1 modules); ...
[ULTRA-DECODE] MoE fast path installed on 48/48 MoE layer(s)
[ULTRA-DECODE] sorted_routes=48/48
```
Load-time (un-baked) path instead prints `[ULTRA-DQ8] mamba expected=96 actual=96` etc. — **hard-fail on mismatch** (`... MISMATCH`, never half-quantize). If you see `fingerprint mismatch -- probe skipped`, the env vars fired against a non-Ultra model. See profiling.md / omlx-live-path-verification law: no counter = not engaged.

## 6. Measured speed (batch-1, stream-free T1/T256 probes; short == 5k at every rung)
| stage | tok/s | resident |
|---|---|---|
| baseline (oQNVFP4, golden env) | **7.68** (130.2 ms/tok) | 333.6 GB |
| +serving fixes +sorted routes | 8.02 | |
| +DQ8 mamba (96/96) | 10.38 | −17.7 GB |
| +DQ8 moedense (192/192) | 12.52 | −26.1 GB |
| +DQ8 attn q/o (24/24) | 12.85 | |
| +DQ8 lm_head (1/1) → **production** | **13.07-13.09** | **305.08 GB** |

Source: `tasks/todo.md` ULTRA ladder; identical on load-time-DQ8 and baked builds. Prefill ~104-160 tok/s @6k, ~143 @20k (`tasks/todo.md` "FIRST LIGHT"). TTFT: fresh 6k ~57s incl. 550B pool load; exact-repeat 5k **40s→3.0s** after serving-fix SF-1.

## 7. Quality (serial, non-stream, temp0)
| bench | score | n | source |
|---|---|---|---|
| gsm8k | **96.67%** | 30 (DQ8 spot) | `memory/omlx-ultra-550b.md`; identical load-time vs baked |
| gsm8k | 91.7% | 60 (oQNVFP4 baseline) | `tasks/todo.md` FIRST LIGHT |
| mmlu | 85.3% | 150 | " (+4.3pp vs Super) |
| arc | 96.7% | 150 | " (+1.7pp vs Super) |

**No quant damage** — DQ8 spot gsm8k ≥ baseline; the 550B lift over Super-120B is visible on mmlu/arc.

## 8. Levers — LIVE vs DEAD vs PARKED
- **LIVE:** the 4-stage DQ8 (baked); sorted-expert-routes (`sorted_routes=48/48`, bit-identical, isolated 3.5ms did NOT survive in-stream but kept free + kill-switched `OMLX_ULTRA_DISABLE_SORTED_ROUTES=1`).
- **DEAD:** fused expert-MLP kernel (K3) — K1 measured expert path 1.87× ideal but expert-**specific** excess only 2.41 ms/token (dense-equal-bytes control 1.71×); `gather_qmm` is ~as efficient as a dense read, kernel pool only ~2-4 ms → cannot rival DQ8, **P0-K parked to P2** (`tasks/ultra_speed.md §4 P0-K`). Requant experts to mxfp4/affine4/5-bit — all within ±4% at Ultra's fat shapes (M1), format exonerated, dead on speed too.
- **PARKED (future mega-kernel campaign):** the residual **1.71× diffuse in-stream overhead** every op pays (same animal as GLM foray §10b's ~32ms) → `future-campaigns.md` #3. P0-2 op-count reduction (1800→~1050) and P0-3 chunked `async_eval` (starvation fix) are designed in `tasks/ultra_speed.md §4` but unbuilt.
- See dead-levers.md for the cross-model index; experiments/ultra-day.md for the EXP ledger.

## 9. Known bugs / watch-items (`tasks/todo.md` "FIRST LIGHT", three Ultra-only serving findings — omlx stack, not the model)
- **Stream "bug" — RESOLVED (2026-07-05):** the one-SSE-chunk / dead-air was the `reasoning_content` channel, not a serving defect — Ultra streams thinking on `delta.reasoning_content`, answer on `delta.content`; count both (profiling.md §3). Non-stream API always fine. (Ledger `todo.md:1065` still reads "in flight" — closure never banked.)
- **Prefix-cache commit lag** → fixed by SF-1 (early cache-index publish; 5k exact-repeat TTFT 40s→3.0s).
- **Throttle mispredictor** (benign) → fixed by SF-2 (clamp measured term in `scheduler.py:_predicted_chunk_transient:3728`; one-time 120 GB expert-wiring jump was charged as per-token rate ×2048).

## 10. Conversion provenance
Built by `omlx/tools/oqnvfp4_nemotron_convert.py --dq8` (affine8-gs64 baked at convert time via the shared `DQ8_STAGES` map, imported identically by patch + converter). Pipeline: NVFP4 source → lossless nvfp4 byte-repack of experts + bf16 dequant of the FP8/bf16 shell + relu² **ts-fold** (per-expert `weight_scale_2` folded into router scores; exact for relu², `tasks/oqnvfp4_nemotron.md §4/§12`) → DQ8 bake. Two config-dialect fixes were required (Ultra ships `layers_block_type` without `num_hidden_layers`; `time_step_limit` as a tagged-infinity dict). Offline gates: census 313/313 q8 triples + bit-parity EXACT vs `mx.quantize` of the old checkpoint's bf16. **Weights needed zero changes — pure repack held.** This flow is the validated template for GLM-5.2-NVFP4 (`future-campaigns.md` #1). Full mechanics in conversion.md; per-tensor map in `tasks/oqnvfp4_nemotron.md §12`.
