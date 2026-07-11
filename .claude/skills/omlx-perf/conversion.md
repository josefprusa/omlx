> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# conversion.md — the "oQNVFP4" source→serve pipeline

The validated recipe that turned NVIDIA's `Nemotron-3-Ultra-550B-NVFP4` into production
`Nemotron-3-Ultra-oQNVFP4-dq8` (13.07–13.09 tok/s, gsm8k 96.67% n=30). Written as the
template for the **next** conversion: `nvidia/GLM-5.2-NVFP4` (465GB). If a step below is
model-specific, the "GLM deltas" section says how it changes.

## Doctrine (quant-first)

1. **Serve tensors in their source-precision containers.** NVIDIA already calibrated the
   quant; a bit-exact byte-repack preserves it. Never requantize NVIDIA's quantized
   weights — fix the kernel instead (`omlx-ultra-550b.md`; laws.md cross-ref).
2. **NVIDIA's exclusion/`ignore` list IS the sensitivity map.** Whatever they left BF16
   (attention, indexer, router, MTP) is precision-sensitive — keep it high-precision;
   only shrink it later with reversible affine8 (DQ8), never below.
3. **Every fidelity claim is a gate, not a hope** (§Gate battery). Cross-ref laws.md.

## Pipeline at a glance (decision tree)

```
source tensor
├─ NVFP4 (U8 codes + E4M3 weight_scale + f32 weight_scale_2)  → STAGE 1 byte-repack + STAGE 2 ts-carry
├─ per-tensor FP8 (E4M3 weight + f32 weight_scale)            → dequant→bf16 (E4M3 LUT × scale)
├─ MXFP8 (F8_E4M3 + E8M0 weight_scale_inv)                    → dequant→bf16 (mx.dequantize mxfp8)
└─ already bf16                                               → passthrough
then, to cut token-read bytes on the bf16 shell:              → STAGE 3 DQ8 bake (affine8-gs64)
DROP always: input_scale (w4a16≠w4a4), k_scale/v_scale (KV deploy), mtp.* (unless MTP-serving)
```
Two converters, do not cross them: `omlx/tools/oqnvfp4_convert.py` = MiniMax-M3 (SwiGLU MoE,
fused gate_up); `omlx/tools/oqnvfp4_nemotron_convert.py` = Nemotron-H (relu² MoE). GLM-5.2 is
SwiGLU → fork the **M3** converter, not Nemotron's (§GLM deltas).

## Stage 1 — NVFP4 byte-repack (NOT a requantization)

NVIDIA ModelOpt NVFP4 = U8-packed FP4 (E2M1) codes + E4M3 per-16 `weight_scale` + f32
per-tensor `weight_scale_2`. MLX `mode="nvfp4"` state = exactly `(weight uint32, scales
uint8-E4M3)`, no bias, no per-tensor slot. The repack is a pure byte reinterpretation
(`omlx/tools/oqnvfp4_convert.py:repack_nvfp4`): `codes_u8.view(uint32)` → weight (8 codes/word),
`weight_scale` bytes as uint8 → scales. Stack over experts for `gather_qmm`. Verified
**byte-identical, max|diff|=0.0, low-nibble-first** vs an independent ModelOpt-formula
reference (`tasks/oqnvfp4_nemotron.md §7 D2`; `tasks/oqnvfp4_build.md` PROVEN).

- `input_scale` is **DROPPED**: MLX runs **w4a16** (bf16 activations); TRT-LLM runs w4a4.
  w4a16 is strictly higher fidelity, so the activation scale is unused (`oqnvfp4_nemotron_convert.py`
  module docstring; `oqnvfp4_nemotron.md §3` rows 4/8).
- raw dtype reads go through integer numpy views (`oqnvfp4_convert.py:tensor_from_bytes`);
  ml_dtypes is NOT installed, so FP8 decode uses a hand-built LUT (below).

Per-tensor FP8 → bf16 (`oqnvfp4_nemotron_convert.py:dequant_fp8_pertensor` + `e4m3fn_lut`):
256-entry E4M3FN uint8→f32 LUT (self-checked: `lut[0x38]==1.0, 0x7E==448.0, 0x7F==NaN`),
`bf16(lut[w_u8] * weight_scale)`. E4M3 is exactly bf16-representable, so one rounding only.

## Stage 2 — weight_scale_2 (`ts`) carry + fold

MLX nvfp4 has no per-tensor scale slot and `e4m3_block_scale · weight_scale_2` is not
E4M3-representable → folding ts into the stored scales would be **lossy (rejected)**. So carry
each expert's `weight_scale_2` as f32 sidecars and re-apply at runtime. Convert-time
**asserts `min(ts) > 0`** per MoE layer (`oqnvfp4_nemotron_convert.py:_assert_positive`) — the
fold's homogeneity pull-through needs positive scales; fails LOUD rather than emit a
silently-wrong checkpoint. **The fold path depends on the activation:**

| MoE type | example | ts fold — where it goes | why |
|---|---|---|---|
| **relu², gateless** | Nemotron-H | ONE per-expert scalar `(fc1_ts²·fc2_ts)` folds into **router scores** | `nn.ReLU2` is degree-2 homogeneous: `fc2·relu²(fc1·x)` pulls both scales out to a scalar; MoE sum is linear → exact (`nemotron_h_nvfp4_ts.py:_moe_call`; proof `oqnvfp4_nemotron.md §4`) |
| **SwiGLU, gated** | MiniMax-M3, **GLM-5.2** | `down_ts`→**scores** (exact); `gate_up_ts`→**element-wise on the gate_up matmul output (fused out-axis `[gate; up]`, split axis=-1), fp32, BEFORE the activation** | `silu(gate)·up` is NOT homogeneous; the up/gate scale can't move through silu, so it stays output-side (`oqnvfp4_build.md` runtime patch §; `oqnvfp4_convert.py:convert_routed_experts` emits `gate_up_ts[E,2]`+`down_ts[E]`) |

Runtime fold is a pre-load patch that self-gates on `*_ts` key presence and does a
**per-instance** `mixer.__class__` swap (never a module-level rebind — siblings in the same
process stay stock). Wired in `omlx/utils/model_loading.py:176` gated on
`config["omlx_moe_nvfp4_ts"]`. Kill-switch `OMLX_NEMO_DISABLE_NVFP4_TS=1` (Nemotron) /
`OMLX_M3_DISABLE_NVFP4_TS=1` (M3): swap+registration still happen (strict load needs the params)
but the multiply is skipped → output mis-scaled = DEBUG/attribution only, NOT a correctness mode.
Empirically (bf16): norm-rel error 6.06e-3 < bf16 eps 7.8e-3; `ts=1` identity control
max|abs|=0.0 (`oqnvfp4_nemotron.md §4`). Fold is **mandatory** — without it, ~100× error.

## Stage 3 — DQ8 dense-shell bake (affine8-gs64)

**Why:** NVIDIA shipped the ~33B non-expert shell as FP8/bf16; the Stage-1 repack upcast it to
bf16, giving 78GB/token reads (66 dense bf16 + 12.5 nvfp4 experts) → a ~10.4 tok/s bandwidth
ceiling. DQ8 = reversible affine 8-bit, group-size 64 on the shell → ~47GB/token → ceiling ~17 tok/s (measured 13)
(`omlx-ultra-550b.md`; M0 measured 47.2ms/token weighted, `todo.md:1078`). affine8 is lossless
enough that quality held (gsm8k 96.7% n=30 vs 91.7% n=60 baseline, `todo.md:1069`).

Single source of truth = `omlx/patches/nemotron_h_dq8_map.py:DQ8_STAGES`, imported by BOTH the
load-time patch and the converter (map-parity + is-identity **tested**, below), so the two can
never target different modules. `DQ8_MODE="affine", DQ8_BITS=8, DQ8_GROUP_SIZE=64`.

| stage | env var | block_type | targets | Ultra count |
|---|---|---|---|---|
| A mamba | `OMLX_ULTRA_DQ8_MAMBA` | M | `in_proj, out_proj` | 48×2 = 96 |
| B moedense | `OMLX_ULTRA_DQ8_MOEDENSE` | E | `shared_experts.{up,down}_proj, fc1_latent_proj, fc2_latent_proj` | 48×4 = 192 |
| C attn | `OMLX_ULTRA_DQ8_ATTN` | * | `q_proj, o_proj` | 12×2 = 24 |
| D lmhead | `OMLX_ULTRA_DQ8_LMHEAD` | None (global) | `lm_head` | 1 |

**Total 313.** NEVER touched: `switch_mlp` (native NVFP4 experts), `gate`/router, all norms,
embeddings, mamba conv1d/A_log/D/dt_bias, and **`k_proj`/`v_proj`** — M0 measured quantizing
k/v a net LOSS at Ultra's skinny attn-kv shape (`nemotron_h_dq8_map.py:69`; `oqnvfp4_nemotron_convert.py:convert_attn`).

**Two equivalent forms** (bit-identical — `mx.quantize` is deterministic):
- **Load-time** (env quartet all `=1`): probe `omlx/patches/nemotron_h_dq8.py:apply_ultra_dq8`
  quantizes one Linear at a time, `mx.clear_cache()` every 4 (16 pooled ~7GB freed bf16 → too
  far from the <1GB transient budget). Adds a ~3min load-time quantize pass.
- **Convert-time bake** (`--dq8`): no load-time pass; ~305GB resident direct. The productized path.

## Converter anatomy (`oqnvfp4_nemotron_convert.py` — read before running)

- **CLI:** `.venv/bin/python omlx/tools/oqnvfp4_nemotron_convert.py --src <NVFP4_dir> --out <dir>
  --shard-size-gb 5 [--dq8] [--allow-partial] [--limit-layers N]`. `--allow-partial` skips
  not-yet-downloaded shards (smoke-test mid-download); `--limit-layers` for debug.
- **Config-dialect fixes** (`build_output_config`) — these bit Ultra and WILL bite GLM:
  - `cfg.setdefault("num_hidden_layers", len(pattern))` — Ultra ships only `layers_block_type`;
    mlx_lm ModelArgs needs `num_hidden_layers` positionally (its derive runs too late → TypeError).
  - `time_step_limit` tagged-infinity `[0.0,{"__float__":"Infinity"}]` → decode to float; **drop
    the key** when non-finite (== mlx_lm default (0, inf)) or `mx.clip` ValueError at load.
  - drops ModelOpt per-shard `model-*-of-*.json` fragments (they name source shards we re-sharded).
  - `resolve_pattern`: accepts `hybrid_override_pattern` (str, Super) OR `layers_block_type` list
    (Ultra) via `{mamba:M, attention:*, moe:E, mlp:-}`.
- **`--dq8` path:** `dq8_targets_for_block_type` (reads DQ8_STAGES) → `quantize_dq8` (=`mx.quantize`
  gs64/8/affine) → `apply_dq8_to_block` replaces each bf16 `.weight` with `.weight/.scales/.biases`;
  `lm_head` special-cased (global, not under a layer); per-module config entries added;
  sets `cfg["omlx_ultra_dq8_baked"]=True` (informational marker only).
- **Load-side baked detection** (`nemotron_h_dq8.py:_classify`, 3-way): each target is `linear`
  (plain bf16 → quantize), `baked` (already QuantizedLinear at our exact bits/gs/mode → skip), or
  `other` (any other quant state). **All-baked → log INFO + skip** the stage; **any mix, or any
  `other` → raise** (never guess a corrupted/partial checkpoint). This makes the env quartet
  **inert** on a baked checkpoint — the four "baked checkpoint detected" INFO lines are CORRECT.

## GATE BATTERY (the heart — run every one before serving)

**G1 — tensor census** (offline, no model load): every DQ8 target emitted as a q8 triple,
experts + ts sidecars untouched, no stray-quantized excluded module.
```
.venv/bin/python $OMLX_RESEARCH_TMP/ultra_speed/baked_gate_offline.py
# edit OLD=/old oQNVFP4 dir, NEW=/baked dir first
# expect: gate1: targets=313 missing=0 stray=0 expert_mats=96 ts_sidecars=96
```
Arithmetic to reproduce for any model: `targets = ΣA+B+C+D`; baked tensor count =
`base_tensors + targets×2` (each bf16 `.weight` → 3 tensors, +2). Ultra: **1745 = 1119 + 313×2**
(`todo.md:1096`). FAIL = converter emitted a bf16 target, quantized an excluded module, or
mangled expert/ts tensors.

**G2 — bit-parity vs `mx.quantize`** (same script, gate2): `mx.quantize` is **deterministic**, so
the baked triple must `mx.array_equal` (NOT `allclose`) the result of quantizing the OLD
checkpoint's bf16 tensor with `(gs64, 8, affine)`, sampled across stages/depths.
```
# expect: gate2 PASS <name> ... ; final line: OFFLINE GATES: ALL PASS
```
Proves load-time-DQ8 ≡ convert-time-bake at the tensor level. Unit-tested:
`tests/test_nemotron_ultra_dq8_convert.py::TestBitParity` (converter vs `QuantizedLinear.from_linear`,
f32 AND bf16); map-parity + `DQ8_STAGES is` identity: `::TestMapParity`. FAIL = a rounding
divergence between the two quantize call paths.

**G3 — real-path load ("D4")**: load through the REAL omlx engine path **offline** (server pool
must be empty — a 465GB standalone load can't coexist with a served model). This is where
config-dialect crashes surface (num_hidden_layers TypeError, time_step_limit ValueError).
Pattern: `maybe_apply_pre_load_patches(path)` → `mlx_lm.load(path)`, then a few greedy prompts,
assert no NaN/Inf and coherent+correct gen (`oqnvfp4_nemotron.md §7 D4`, `§10`; `todo.md:1044`).

**G4 — live ladder** (serve, then verify engagement BEFORE trusting any number):
- serve with the DQ8 env quartet in the launch line; grep the log:
  `[ULTRA-DQ8] <stage> expected==actual` per stage (96/192/24/1) on a **load-time** build; on a
  **baked** build expect `baked checkpoint detected ×4` INFO instead (both are correct).
  `[ULTRA-DECODE] sorted_routes=48/48` confirms the decode fast path (`moe_fastpath.py:_RouteCensus`).
- T1/T256 stream-free probes (short == 5k at every rung); quality spot-check gsm8k **n≥30**.
- Live-path law: a gated fast path that never logs engagement is presumed OFF (profiling.md
  cross-ref). Cross-ref ops-runbook.md for the serve/restart procedure, profiling.md for benching.

## Worked example — Ultra 550B end-to-end (all cited)

| step | value | source |
|---|---|---|
| NVFP4 master (source) | 352GB / 113 shards, archived on T7 | `oqnvfp4_nemotron.md §12` |
| → oQNVFP4 (Stage 1+2, no DQ8) | 357.737GB / 97 shards / 1119 tensors / 96 ts sidecars | `todo.md:1043` |
| → +DQ8 bake (`--dq8`) | 327.192GB disk / 1745 tensors / **305.08GB resident** | `todo.md:1096,1101` |
| conversion time | ~45 min (positivity assert green on all 48 MoE layers) | `todo.md:1097,1043` |
| load | 68s, no quantize pass | `todo.md:1102` |
| decode | 13.07–13.09 tok/s (== the 305.07GB load-time-DQ8 build) | `todo.md:1101` |
| gsm8k | 96.67% n=30 (identical to load-time-DQ8) | `todo.md:1102` |

Physics: 78GB→~47GB token reads is the whole win (`omlx-ultra-550b.md`). **`du` while converting
misleads** — early reads run ahead/behind during warmup and shard flush; trust the final
`Summary:` line and the census, not mid-run `du -sh` (grep-able trap: `gotchas.md` "du mid-convert"; preflight.py check 5).

## GLM-5.2-NVFP4 — ✅ CONVERTED 2026-07-06 (as-built deltas vs the plan below)

Executed as `omlx/tools/oqnvfp4_glm_convert.py` → `GLM-5.2-oQNVFP4` (427.75 GB / 3130 tensors; measured record in `models/glm52.md §11`, gates in `scripts/glm_oqnvfp4_gate_offline.py`). As-built corrections to the plan below — trust these over the older text:
1. **Fused-shared does NOT apply to GLM** (point 1 below is M3-only): `_pack_mlx_unpacked_moe_weights` never runs for glm_moe_dsa (text engine, not vlm.py), GLM's `DeepseekV32MoE` keeps `shared_experts` a separate module, and NO `omlx_moe_shared_expert_mode` key is needed. Shared experts went affine8-gs64 separate; `--shared-nvfp4` flag exists as the −1.4 GB lever (untested).
2. **kv_b is pre-absorbed at convert time** (fold cloned from `deepseek_v32.Model.sanitize:1053-1089`) → quantized `embed_q`/`unembed_out` on disk; no kv_b emitted; sanitize skips; native `mh_qmm_m1`/`q8_vup_flat` engage at affine8-gs64.
3. **ts-fold** = new patch `omlx/patches/glm_moe_dsa/nvfp4_ts.py` (nemotron-pattern sanitize wrap + per-instance `_TsSwitchGLU` swap), fleet key `omlx_moe_nvfp4_ts`, `down_ts` folded on **output rows** (not scores) so ONE hook covers the native `glm_moe_weighted_sum` + decode-opt gemv + plain-sum paths. `cast_predicate` wrapped to keep ts f32.
4. Experts emitted **pre-fused** `switch_mlp.gate_up_proj` ⇒ converter writes the nvfp4 quant-dict entries itself (the 603f47f spec-carry only fires for pre-fusion layouts).
5. MTP drop = omit `layers.78*` AND pop `num_nextn_predict_layers` (MTP patch then never applies).
6. Indexer schedule must mirror `ModelArgs.__post_init__` exactly (`indexer_types` > `index_topk_pattern` > freq/offset); source ships indexer weights only on "full" layers (21 of 78). Converter refuses a non-empty `--out` (stale-shard mixing).

## Puzzle-75B oQ48 — THIRD converter lineage (bake-from-bf16: no repack, no ts-fold)

`omlx/tools/oq_puzzle_convert.py`: NemotronH-Puzzle BF16 master → oQ48 by SINGLE quantization
(`mx.quantize` deterministic; no byte-repack, no ts sidecars, no load-time patch — pure bake).
Distinct from both oqnvfp4 lineages because Puzzle's MoE is heterogeneous PER LAYER
(`block_configs`: own intermediate + top-k per layer; 512 experts global).
Recipe: shell affine8-gs64 (257 targets via the `SHELL_STAGES` single-source dict — converter +
offline gate both import it, can't drift); experts affine4-gs64 stacked per layer into the stock
post-sanitize `switch_mlp.fc1/fc2` names (strict load, zero sanitize surgery); `mtp.*` tensors AND
config keys dropped. NEVER quantized: k/v_proj, gate/router, norms, conv1d, A_log/D/dt_bias, embeddings.
**Traps:** BF16 master ships `model.`-prefixed names (vendored sanitize remaps → `backbone.`); the
converter refuses a non-empty `--out` (stale-shard guard, GLM-converter pattern). Gate battery:
`omlx/tools/oq_puzzle_gate_offline.py` = census (257 + per-layer expert shapes vs block_configs, zero
stray/mtp) + bit-parity with FORCED per-stage coverage + config lint.
Result (EXP-091): 46.9GB / 1437 tensors; NVIDIA torch-ref logits 1.2e-6; tiny-e2e cosine 0.99990;
ALL PASS. Dossier: `models/nemotron-puzzle.md`.

## Next conversion — GLM-5.2-NVFP4 (465GB) deltas

GLM-5.2 = MLA + DSA lightning-indexer + SwiGLU MoE (256 routed + 1 shared, first 3 layers dense)
+ MTP layer 78. NVFP4 release quantizes **only the routed experts**; everything else is BF16
(`tasks/glm_quant_matrix.md` [2][3]). So GLM oQNVFP4 = 28.55B BF16 shell + 45.30B E4M3 block
scales + 362.39B U8 packed FP4 experts.

1. **Fork the M3 converter** (`oqnvfp4_convert.py`), not Nemotron's — GLM experts are **SwiGLU**
   (gate/up/down), so use the M3 ts mechanism: `down_ts`→scores, `gate_up_ts`→pre-activation
   output-side fold (§Stage 2 table; `glm_quant_matrix.md`). Nemotron's single-scalar router
   fold does NOT transfer. The fused `gate_up_proj.weight` out-axis is `[gate; up]` (split axis=-1,
   first half gate) — reversing it silently blends them (`oqnvfp4_build.md:58-60`; gotchas.md).
   **Pick a shared-expert mode** via the converter's `omlx_moe_shared_expert_mode` config key
   (`oqnvfp4_build.md:95`): `"separate"` = `switch[256]` routed-only + a separate affine8 shared MLP
   (default); `"fused"` = `switch[257]` with the shared grafted as expert 256 (`--fuse-shared-nvfp4`,
   ts sidecars `[257,2]/[257]`). A **separate-shared checkpoint MUST set the key** or the engine's
   `_pack_mlx_unpacked_moe_weights` force-fuses `shared_experts.down_proj` into the routed stack and
   **crashes on a concat mismatch** (`oqnvfp4_build.md:150-158`; `omlx/engine/vlm.py:759,765`;
   gotchas.md — needs a server restart). (M3 uses 128/129; GLM is 256/257 = 256 routed + 1 shared.)
2. **Byte-repack only the routed experts**; NVIDIA's `ignore` list (lm_head, embed, layers 0–2,
   all `self_attn*` incl. DSA indexer, all `shared_experts*`, layer 78/MTP) is already BF16 →
   passthrough. Drop `input_scale`; `kv_cache_quant_algo=FP8` scales are KV-deploy, drop.
3. **DQ8-bake the 28.55B BF16 shell** to cut token reads (same affine8-gs64 doctrine). Build a
   GLM `DQ8_STAGES` map + its own fingerprint; keep the router/gate, norms, and the
   precision-sensitive **DSA indexer** BF16 (NVIDIA & FP8 both keep the indexer high-precision —
   `glm_quant_matrix.md [4]`). MTP layer 78: drop for KV headroom, or graft per MTP strategy.
4. Run G1→G4 unchanged. Expect the same config-dialect traps (num_hidden_layers, tagged floats).
   Cross-ref models/*.md for the GLM decode-kernel state; future-campaigns.md for the 256k-KV goal.

## Post-swap rules

- **New weights → NEW MODEL NAME** (e.g. `-dq8` suffix). Reusing a name poisons the SSD/mmap
  cache with stale bytes (`todo.md:1103`; kv-cache.md cross-ref).
- **Bounce the server BEFORE `rm` of the old model** — APFS/mmap holds the file; deleting under a
  live mmap corrupts or hangs. Stop pool → rm → restart/discovery (ops-runbook.md cross-ref).
