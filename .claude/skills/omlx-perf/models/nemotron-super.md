> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Nemotron-3-Super-120B-A12B — dossier

Same NemotronH arch as Ultra at ~1/4 scale. Served in **two containers** with the same weights-precision doctrine: the fleet-default **affine-int4 (oQ4e)** build and the **NVFP4 (oQNVFP4)** pathfinder that de-risked the Ultra converter. Super is also where the **NVFP4 small-shape 0.58× tax** and the **SpecPrefill 6-way concurrency bug** were characterized.

## 1. Identity — two variants
| | oQ4e (FLEET DEFAULT) | oQNVFP4 (pathfinder) |
|---|---|---|
| path (`~/.omlx/models/`) | `NVIDIA-Nemotron-3-Super-120B-A12B-oQ4e` | `unigilby/Nemotron-3-Super-oQNVFP4` |
| disk | 68.6 GB | 77.5 GB (index 79.3 GB) |
| container | affine int4 gs64 experts | NVFP4 gs16 experts + bf16 shell |
| settings row | yes, `is_default=true`, alias **`nemotron-3-super`** | none (offline-tested, not the served default) |

`model_type=nemotron_h`, vocab 131072, native context **262 144 (256k)** (config.json). Source of the NVFP4 build: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (80.3 GB, 17 shards; `tasks/oqnvfp4_nemotron.md §1`).

## 2. Geometry (`tasks/oqnvfp4_nemotron.md §1`, verified vs oQ4e headers)
- 120B total / ~12B active (A12B). hidden 4096.
- **88 layers = 40 mamba2 + 8 attention + 40 MoE** (`hybrid_override_pattern`, len-88 string).
- MoE: **512 experts, top-22**, `moe_latent=1024`, `moe_intermediate=2688`, shared expert 5376, **relu²** activation (no gate; experts operate in latent space: `fc1_latent_proj` 4096→1024 → SwitchMLP 1024→2688→1024 → `fc2_latent_proj` 1024→4096 → + shared).
- Mamba 128 heads; attention 32 heads. **`switch_mlp.fc1` = up_proj, `fc2` = down_proj** (stock loader stacks experts).

## 3. Quant layout — per-tensor dtype split
NVFP4 build (`tasks/oqnvfp4_nemotron.md §1-3`): the campaign law in miniature — **only the 40 960 routed-expert tensors stay NVFP4** (bit-exact byte-repack); everything else is bf16.

| family | oQNVFP4 | oQ4e |
|---|---|---|
| routed experts (fc1/fc2, top-22) | **NVFP4 E2M1 gs16** + E4M3 block-scales + fp32 `weight_scale_2` sidecars (`fc1_ts`/`fc2_ts`) | **affine int4 gs64** |
| mamba in/out_proj, latent projs, o_proj, shared experts | **bf16** (FP8 source dequantized once, exact — E4M3 ≤ bf16 mantissa) | affine int4 / bf16 |
| q/k/v, gate, embeddings, lm_head, norms | bf16; `gate.e_score_correction_bias`/`A_log` **f32** | bf16 / f32 |

Per-token bandwidth math is worked for **Ultra** (`tasks/ultra_speed.md §2`); Super scales down by ~4× (hidden 4096 vs 8192, latent 1024 vs 2048). Not separately tabulated — GAP if a Super byte-ceiling is ever needed.

## 4. Serving (`~/.omlx/model_settings.json` → `NVIDIA-Nemotron-3-Super-120B-A12B-oQ4e`)
```
temperature 0.8, top_p 0.95, max_context_window 1000000, max_tokens 32768,
model_alias "nemotron-3-super", is_default true, force_sampling false,
turboquant_kv_enabled true, turboquant_kv_bits 8, turboquant_skip_last true,
specprefill_enabled true, specprefill_draft_model .../NVIDIA-Nemotron-3-Nano-30B-A3B-oQ4,
specprefill_keep_pct 0.2, specprefill_threshold 2048,
mtp_enabled false, vlm_mtp_enabled false, trust_remote_code false
```
The oQNVFP4 build has no settings row (served ad-hoc for the D5 A/B). Golden env applies fleet-wide (env-setup.md). oQ4e runs **SpecPrefill** with the Nano-30B-oQ4 draft (that is also the source of the concurrency bug, §9).

## 5. Engagement grep set
- **oQNVFP4 only:** the runtime **ts-fold** fires once at load — one INFO line from `omlx/patches/nemotron_h_nvfp4_ts.py:apply_nemotron_h_nvfp4_ts_patch`, and all 40 MoE mixers become `_TsFoldMoE` (`tasks/oqnvfp4_nemotron.md §10/§11`; log line 82 fired on oQNVFP4 load, **absent** on oQ4e → non-interference confirmed live). Kill/attribution switch `OMLX_NEMO_DISABLE_NVFP4_TS=1` (still binds the ts params but skips the fold → mis-scaled, a debug switch not a correctness mode).
- **oQ4e:** no ts line; SpecPrefill draft engagement appears in the TTFT path. Gate on the ts line to know **which** Super is loaded.

## 6. Measured speed — the 0.58× headline (`tasks/oqnvfp4_nemotron.md §11`, live D5 A/B, `nemotron-3-super`)
| metric | oQ4e (affine) | oQNVFP4 (nvfp4) |
|---|---|---|
| decode short / 5k(warm) | **49.4 / 50.4 tok/s** | **29.8 / 29.4 tok/s (0.58×)** |
| TTFT ~5k | 3.9 s (SpecPrefill) | 7.2 s (no spec configured) |

**REVALIDATED 2026-07-08 (EXP-091):** the small-shape verdict drove Puzzle-75B's container choice —
oQ48 quantized experts to affine4-gs64 from the BF16 master instead of serving NVIDIA's NVFP4, at
even skinnier shapes than Super's (`models/nemotron-puzzle.md` §3-4). **Why 0.58×:** nvfp4 `gather_qmm` (gs16 = 4× the scale reads of affine gs64, plus E4M3 + fp4 decode) vs the mature affine-int4 kernel — a **small-shape** effect on Super's skinny latent-1024/moe-2688 experts. It is a **kernel-opt opportunity, not correctness**, and it **does NOT transfer to Ultra** (M1: nvfp4/affine4/mxfp4 within ±4% at Ultra's fat latent-2048 shapes — `tasks/ultra_speed.md §3 M1`, `memory/omlx-ultra-550b.md`). The TTFT gap is just oQ4e's SpecPrefill; oQNVFP4 has none configured.

## 7. Quality (`tasks/oqnvfp4_nemotron.md §11`, serial)
| bench | oQ4e | oQNVFP4 | n |
|---|---|---|---|
| gsm8k (serial) | 92% (11/12) | 92% (11/12) | 12 |
| mmlu | 81% | 80% | 100 |
| arc | 95% | 94% | 100 |

**Statistically tied on all three** — the lossless NVFP4 repack + exact relu² ts-fold is correct in live serving. Quality was proven equal *before* the 0.58× speed gap was accepted.

## 8. Levers — LIVE vs DEAD vs PARKED
- **LIVE (oQ4e, default):** SpecPrefill (draft=Nano-30B-oQ4) for prefill/TTFT; TurboQuant 8-bit KV (`turboquant_skip_last`). See kv-cache.md for TurboQuant mechanics.
- **LIVE (oQNVFP4):** relu² ts-fold (mandatory for correctness).
- **DEAD/parked:** the nvfp4 kernel 0.58× tax is a real **kernel-opt** lever but unbuilt; it is the same NVFP4-gather_qmm workstream that Ultra's K1 later parked to the mega-kernel campaign. Requant experts away from NVFP4 = barred by doctrine (preserve NVIDIA's weight values) and moot on speed at fat shapes.

## 9. Known bugs / watch-items
- **SpecPrefill 6-way concurrency → gsm8k 92% → 15% (OPEN, PRODUCTION RISK).** `acc_bench`'s *concurrent* (6-worker) gsm8k scored oQ4e **15%**; single-shot AND serial both **92%** (`tasks/oqnvfp4_nemotron.md §11`, "Artifact flagged — NOT quantization"). It is a **SpecPrefill-under-concurrency** degradation, not a quant defect. Affects the fleet-default Super whenever concurrent requests hit the SpecPrefill path (agent fan-outs qualify). Full write-up + first diagnostics in `future-campaigns.md` #5.
- Reason to prefer the **serial** number when benching oQ4e: SpecPrefill concurrency contaminates.

## 10. Conversion provenance
oQNVFP4 built by `omlx/tools/oqnvfp4_nemotron_convert.py` — the **pathfinder** for Ultra: D2 per-class bit/near-exactness (NVFP4 experts byte-identical, max|diff|=0.0 vs ModelOpt formula; FP8→bf16 single-rounding < bf16 eps), D3 full conversion (88 layers, 923 tensors, 79.270 GB), D4 offline real-path load test — all PASSED (`tasks/oqnvfp4_nemotron.md §7`). The relu² ts-fold (`fc1_ts²·fc2_ts` into router scores) is exact because relu² is degree-2 homogeneous and `weight_scale_2>0` (asserted at convert time, §4). oQ4e is an older affine build (`oQ` pipeline). Full mechanics in conversion.md.
