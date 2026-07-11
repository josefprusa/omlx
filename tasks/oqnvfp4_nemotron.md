# oQNVFP4 Converter — Nemotron-3-Super-120B-A12B (NemotronH latent-MoE hybrid)

**Deliverable 1 — MAPPING SPEC (review gate before writing the converter).**
Pathfinder for the 550B Ultra. Source: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
(80.3 GB, 17 shards). Target loader: **stock** `mlx_lm/models/nemotron_h.py` (verified in
`.venv`), naming/structure ground-truthed against the user's existing
`~/.omlx/models/NVIDIA-Nemotron-3-Super-120B-A12B-oQ4e` (same arch, affine build).

All facts below are verified against the scout configs, the oQ4e checkpoint headers,
the installed MLX 0.31.2, and the mlx_lm loader — not re-derived assumptions.

---

## 0. TL;DR design (one sentence)
**Keep only the 40,960 routed-expert tensors as NVFP4 (bit-exact byte-repack);
dequantize everything else to bf16; carry per-expert `weight_scale_2` as two fp32
sidecars per MoE layer and fold `fc1_ts²·fc2_ts` into the router scores at runtime
(exact for relu²).** Est. output **~79.3 GB**.

---

## 1. Source architecture & quant inventory (verified)

Config: `hidden=4096`, `moe_latent_size=1024`, `moe_intermediate_size=2688`,
`moe_shared_expert_intermediate_size=5376`, `n_routed_experts=512`,
`num_experts_per_tok=22`, `vocab=131072`, `mlp_hidden_act=relu2`, `dtype=bfloat16`.

`hybrid_override_pattern` (len 88) → **40 `M` mamba2, 40 `E` MoE, 8 `*` attention.**

`quantization_config` = ModelOpt **MIXED_PRECISION**, two config groups:
- **group_1 NVFP4** (bits=4, group_size=16): 40,961 tensors = **all routed experts**
  `up_proj`+`down_proj` (512×40 each) + exactly 2 stray `shared_experts` tensors.
- **group_0 FP8** (bits=8, type=float, per-tensor E4M3): 139 tensors = a *subset* of
  mamba `in_proj`/`out_proj`, latent projs, `o_proj`, shared experts.
- Everything else (embeddings, lm_head, gate, norms, q/k/v, most latent projs, ~half
  the mamba projections) is **bf16** in the source (no `weight_scale`).
- `kv_cache_quant_algo=FP8` → `k_proj.k_scale`, `v_proj.v_scale` are KV-cache deploy
  scales, **not** weight scales.

Source tensor-name patterns (64 total; `mtp.*` shown but **DROPPED** — see §5):

| source suffix (per `backbone.layers.N.mixer`) | source dtype(s) | count |
|---|---|---|
| `experts.K.up_proj.{weight, weight_scale, weight_scale_2, input_scale}` | U8 / E4M3 / f32 / f32 | 20480 |
| `experts.K.down_proj.{weight, weight_scale, weight_scale_2, input_scale}` | U8 / E4M3 / f32 / f32 | 20480 |
| `gate.weight` / `gate.e_score_correction_bias` | bf16 / f32 | 40 / 40 |
| `fc1_latent_proj.weight` (+`weight_scale`,`input_scale` on 3) | bf16 or F8_E4M3 | 40 |
| `fc2_latent_proj.weight` (+`weight_scale`,`input_scale` on 1) | bf16 or F8_E4M3 | 40 |
| `shared_experts.up_proj.weight` (+scales) | F8_E4M3(39)/NVFP4(1) | 40 |
| `shared_experts.down_proj.weight` (+scales) | F8_E4M3(37)/NVFP4(1)/bf16(2) | 40 |
| `in_proj.{weight}` (+`weight_scale`,`input_scale` on 28) | F8_E4M3(28)/bf16(12) | 40 |
| `out_proj.{weight}` (+`weight_scale`,`input_scale` on 29) | F8_E4M3(29)/bf16(11) | 40 |
| `A_log`,`D`,`dt_bias`,`conv1d.{weight,bias}`,`norm.weight` (mamba) | f32/bf16 | 40 each |
| `q_proj.weight` | bf16 | 8 |
| `k_proj.{weight, k_scale}` / `v_proj.{weight, v_scale}` | bf16 / f32 | 8 each |
| `o_proj.weight` (+`weight_scale`,`input_scale` on 2) | F8_E4M3(2)/bf16(6) | 8 |
| globals: `backbone.embeddings.weight`, `backbone.norm_f.weight`, `lm_head.weight` | bf16 | 1 each |
| `backbone.layers.N.norm.weight` (input norm, every layer) | bf16 | 88 |

---

## 2. Target layout (stock mlx_lm nemotron_h — verified against oQ4e headers)

`Model.sanitize()` in the stock loader (verified):
1. **drops every `mtp.*` weight** (line 539) → we never emit MTP.
2. `moveaxis` on `conv1d.weight` if last dim ≠ 1.
3. **stacks** `experts.{e}.up_proj.weight → switch_mlp.fc1.weight` and
   `experts.{e}.down_proj.weight → switch_mlp.fc2.weight`.

⇒ **`switch_mlp.fc1` = up_proj, `switch_mlp.fc2` = down_proj** (relu² MLP, no gate).
Because we **pre-stack** experts in the converter (emit `switch_mlp.fc1/fc2` directly),
the stock stacking loop is a no-op (`experts.0…` absent) and the pre-stacked
`{weight,scales}` bind straight into `QuantizedSwitchLinear`. Confirmed by oQ4e headers:
`switch_mlp.fc1.weight = [512, 2688, …]`, `switch_mlp.fc2.weight = [512, 1024, …]`.

Experts operate in **latent space**: `fc1_latent_proj` (4096→1024, bf16) → SwitchMLP
(1024→2688→1024) → `fc2_latent_proj` (1024→4096, bf16) → `+ shared_experts`.

Output tensor shapes (NVFP4, MLX 0.31.2 packing verified live: weight `uint32`
= 8 codes/word, scales `uint8` E4M3 = 1/group-of-16, **2 tensors only, no bias**):

| target tensor | shape | dtype |
|---|---|---|
| `…mixer.switch_mlp.fc1.weight` | `[512, 2688, 128]` (in=1024→1024/8) | U32 |
| `…mixer.switch_mlp.fc1.scales` | `[512, 2688, 64]`  (1024/16)        | U8 (E4M3) |
| `…mixer.switch_mlp.fc2.weight` | `[512, 1024, 336]` (in=2688→2688/8) | U32 |
| `…mixer.switch_mlp.fc2.scales` | `[512, 1024, 168]` (2688/16)        | U8 (E4M3) |
| `…mixer.switch_mlp.fc1_ts` | `[512]` | F32 | ← weight_scale_2 sidecar (up) |
| `…mixer.switch_mlp.fc2_ts` | `[512]` | F32 | ← weight_scale_2 sidecar (down) |

Everything else → **bf16** (`gate.weight`, `fc*_latent_proj.weight`,
`shared_experts.*.weight`, all mamba/attention projections, norms, embeddings, lm_head),
`gate.e_score_correction_bias`/`A_log` stay **f32** (loader `cast_predicate` excludes them).

---

## 3. Source → target mapping (mechanical, per suffix)

| # | source | op | target | dtype |
|---|---|---|---|---|
| 1 | `experts.K.up_proj.weight` (U8) | `view(uint32)`, stack over K | `switch_mlp.fc1.weight` | U32 |
| 2 | `experts.K.up_proj.weight_scale` (E4M3/U8) | byte-copy, stack | `switch_mlp.fc1.scales` | U8 |
| 3 | `experts.K.up_proj.weight_scale_2` (f32) | `stack → [512]` | `switch_mlp.fc1_ts` | F32 |
| 4 | `experts.K.up_proj.input_scale` | **DROP** (w4a16) | — | — |
| 5 | `experts.K.down_proj.weight` (U8) | `view(uint32)`, stack | `switch_mlp.fc2.weight` | U32 |
| 6 | `experts.K.down_proj.weight_scale` | byte-copy, stack | `switch_mlp.fc2.scales` | U8 |
| 7 | `experts.K.down_proj.weight_scale_2` | `stack → [512]` | `switch_mlp.fc2_ts` | F32 |
| 8 | `experts.K.down_proj.input_scale` | **DROP** | — | — |
| 9 | `{in,out}_proj.weight` FP8 | E4M3-LUT × `weight_scale` → bf16 | same name | BF16 |
| 10 | `{in,out}_proj.weight` bf16 | passthrough | same name | BF16 |
| 11 | `{in,out,o,fc1_latent,fc2_latent}_proj.{weight_scale,input_scale}` | **DROP** (consumed in dequant / w16) | — | — |
| 12 | `fc{1,2}_latent_proj.weight` (FP8 or bf16) | dequant→bf16 / passthrough | same name | BF16 |
| 13 | `shared_experts.{up,down}_proj.weight` (FP8/NVFP4/bf16) | dequant→bf16 (uniform, all 3 formats) | same name | BF16 |
| 14 | `q_proj.weight` | passthrough | same | BF16 |
| 15 | `k_proj.weight`,`v_proj.weight` | passthrough | same | BF16 |
| 16 | `k_proj.k_scale`,`v_proj.v_scale` | **DROP** (KV-cache deploy scale) | — | — |
| 17 | `o_proj.weight` (FP8 or bf16) | dequant→bf16 / passthrough | same | BF16 |
| 18 | `gate.weight` | passthrough | same | BF16 |
| 19 | `gate.e_score_correction_bias` | passthrough | same | F32 |
| 20 | `A_log`,`D`,`dt_bias`,`conv1d.*`,`norm.weight`,`backbone.*norm*`, `embeddings`, `lm_head` | passthrough (conv1d moveaxis handled by loader) | same | src dtype (f32/bf16) |
| 21 | `mtp.*` | **DROP** (loader drops it) | — | — |

**Dequant of a per-tensor FP8 module** (no ml_dtypes dependency; LUT validated live):
```
lut = e4m3fn_uint8_to_f32_lut()          # 256-entry, S.EEEE.MMM, bias 7, 0x7F=NaN
w_bf16 = (lut[weight_u8] * float(weight_scale)).astype(bf16)   # weight_scale is per-tensor f32
```
`shared_experts` NVFP4 tensors dequant via `mx.dequantize(view(u32), scales, group_size=16,
bits=4, mode="nvfp4") * weight_scale_2` → bf16 (uniform bf16 output regardless of source fmt).

---

## 4. THE ts-CARRY DECISION (exactness argument)

**MLX 0.31.2 nvfp4 state = exactly `(weight uint32, scales uint8-E4M3)` — no per-tensor
scale slot** (verified: `mx.quantize(...,mode="nvfp4")` returns 2 tensors;
`gather_qmm(...,mode="nvfp4")` runs on stacked experts). NVIDIA's dequant is
`code_fp4 · e4m3_block_scale · weight_scale_2`; MLX computes only `code_fp4 · e4m3_block_scale`.
The product `e4m3_block_scale · weight_scale_2` is **not** E4M3-representable in general
→ folding `weight_scale_2` into the stored E4M3 scales would be **lossy** (rejected).
`fc2_latent_proj` is shared across all experts (applied after the weighted sum) → cannot
absorb a **per-expert** scale (rejected). ⇒ `weight_scale_2` **must be carried**.

**Exact fold for relu² (this is cleaner than M3's SwiGLU case).** The expert is
`fc2( relu²( fc1(x) ) )` with **no gate**. `nn.ReLU2` is homogeneous of degree 2 and
`weight_scale_2 > 0`, so for the true (ts-included) weights:

```
expert_true(x) = (fc2_q·fc2_ts) · relu²( (fc1_q·fc1_ts)·x )
              = fc2_ts · fc1_ts² · [ fc2_q · relu²(fc1_q·x) ]
              = (fc1_ts² · fc2_ts) · expert_quant(x)          # c_e, a per-expert scalar
```
The MoE weighted sum is `y = Σ_e score_e · expert_e(x)`, so folding `c_e` into the
post-normalization router scores is **algebraically exact**:

```
scores_eff = scores * (fc1_ts[inds]²  * fc2_ts[inds])   # fp32
y = (switch_mlp(x, inds) * scores_eff[..., None]).sum(-2)
```
This mirrors the **M3-proven** mechanism (M3 `language.py:1983` folds `down_ts` into
`scores`; max|diff| = 0.0) but collapses to a **single** per-expert multiply because
relu² has no gate branch — no pre-activation ts multiply is needed. `norm_topk_prob`
normalizes router probabilities *before* `c_e` is applied, so the fold is independent of
it (correct).

**Positivity precondition (required, asserted at convert time).** The pull-through
`relu²(s·z) = s²·relu²(z)` needs `s = fc1_ts > 0` (for `s < 0`, `relu(s·z) ≠ s·relu(z)`).
ModelOpt `weight_scale_2` is an amax-derived magnitude (`amax/(448·6)`), always strictly
positive. The converter **asserts `min(fc1_ts) > 0` and `min(fc2_ts) > 0` per MoE layer**
and fails LOUD if a pathological zero/negative scale appears — so it can never silently
emit a wrong-at-inference checkpoint. (D2 observed ts ≈ 2.1–2.9e-5 > 0 across sampled
experts.)

**Storage:** two fp32 sidecars per MoE layer — `switch_mlp.fc1_ts[512]`,
`switch_mlp.fc2_ts[512]` (kept separate, not pre-combined, for transparent per-tensor
verification and to keep `fc1_ts` applied inside relu² if a future kernel wants it).
Total ts payload = 0.16 MB.

**Empirically validated (synthetic, MLX 0.31.2), before writing any converter code:**
- fp32 algebra check: fold path == ts-baked-into-weights (diff = pure fp32 reassociation).
- bf16 realistic-scale: **norm-relative error 6.06e-3 < bf16 eps 7.8e-3** (agreement to
  bf16 rounding — same class of noise as any quant kernel's accumulation order).
- `ts=1` identity control: **max|abs| = 0.0 exactly** (fold reduces to identity — no bug).
- control WITHOUT the fold: norm-rel error ~100× larger → the fold is mandatory.

**Runtime requirement (documented, NOT applied to server here).** Exact inference needs
`NemotronHMoE.__call__` to fold `c_e` into scores — a ~3-line change. This is an
integration step for the team-lead's window. The offline load test (§7) **monkeypatches**
this fold in a standalone script so the repo/server stay untouched. Without the fold the
model is mis-scaled per expert and will degrade — the fold is mandatory for correctness.

---

## 5. FP8-mixer & MTP policy

- **FP8 mixers → bf16 (EXACT, default).** `E4M3 val` is exactly representable in bf16
  (3 mantissa ≤ 7); the single `× weight_scale` product is rounded once to bf16 — the
  model's native compute precision (`dtype=bfloat16`). No `mx.dequantize(mxfp8)` — source
  FP8 is **per-tensor** E4M3, not per-32 E8M0, so the M3 mxfp8 repack does **not** apply.
- **Size:** bf16-everything-else ≈ **15.9 GB**; total ≈ **79.3 GB** (source 80.3 GB).
  *Alt if size matters:* requant non-experts to affine8 gs64 → total ≈ **72 GB**
  (reuses `requant_affine`). Default = bf16 (exact); report final at conversion.
- **MTP dropped** (`mtp.*`, incl. its own 512 bf16 experts) — stock loader discards it.
- **`input_scale`/`k_scale`/`v_scale` dropped** — w4a16 + bf16 KV is higher fidelity than
  NVIDIA's w4a4/fp8-kv deployment (matches M3).

---

## 6. Output `config.json` quantization map

Top-level default `{"group_size":64,"bits":4,"mode":"affine"}` (harmless — no module hits
it since bf16 modules carry no `.scales`). **Per-MoE-layer** entries (the only quantized
modules), driving `to_quantized(mode="nvfp4")` → `gather_qmm(mode="nvfp4")`:
```json
"backbone.layers.<E>.mixer.switch_mlp.fc1": {"group_size":16,"bits":4,"mode":"nvfp4"},
"backbone.layers.<E>.mixer.switch_mlp.fc2": {"group_size":16,"bits":4,"mode":"nvfp4"}
```
Set both `config["quantization"]` and `config["quantization_config"]` (loader reads either);
this **overwrites** the source ModelOpt `config_groups`. Add **top-level**
`config["omlx_moe_nvfp4_ts"] = true` (Nemotron config is flat — no `text_config`) as the
runtime-fold marker. Copy tokenizer, chat_template, `configuration_nemotron_h.py`,
`modeling_nemotron_h.py`, etc.; skip source `hf_quant_config.json`.

---

## 7. Verification plan (deliverables 2 & 4)

**D2 — per-class bit/near-exactness — ✅ DONE & PASSED** (validator
`omlx-quant-work/d2_validate.py`, run against real source shards, MoE layer 1, MLX 0.31.2):
- **NVFP4 experts** (up+down, experts 0/1/2/255/511): repacked `weight`/`scales` bytes
  **byte-identical** to source, AND `mx.dequantize(repack,nvfp4)·weight_scale_2` vs an
  independent ModelOpt-formula reference (`fp4 E2M1 · E4M3 · ts`) → **max|diff| = 0.000e+00**,
  **low-first nibble order**. Crux proven on the new model. (ts ≈ 2.1–2.9e-5, all > 0.)
- **FP8 mixers** (in_proj, out_proj, shared up, fc1_latent): LUT×scale→bf16 vs f64 ref →
  **max|rel| = 2.3e-3…2.9e-3 < bf16 eps 7.8e-3** (single rounding); `weight_scale` is a
  **per-tensor scalar** (size 1) — confirmed.
- **bf16 passthrough** (embeddings, gate, norm): **byte-identical**, source dtype BF16.

**D3 — full conversion — ✅ DONE.** `88 layers, 923 tensors, 79.270 GB` →
`~/.omlx/models/unigilby/Nemotron-3-Super-oQNVFP4` (21 shards, index complete, tokenizer +
`configuration/modeling_nemotron_h.py` copied, 80 ts sidecars + 40 switch_mlp layers).
Convert-time positivity assert passed on all 40 MoE layers.

**D4 — offline load test — ✅ DONE & PASSED** (`omlx-quant-work/d4_loadtest.py`, one load
per subprocess, `mx.set_wired_limit(110 GB)` first, ts-fold via the §10 per-instance swap):
- **(A) ts-fold ENABLED:** all 923 tensors bind **strict** (ts included); **no NaN/Inf**
  (checked every decode step); instance-swap correctly scoped — `type(mixer) is _TsFoldMoE`
  on MoE, mamba mixer untouched, and `NemotronHMoE.__call__` NOT globally rebound. Greedy
  gen coherent AND correct: `"capital of France is"` → `" Paris."`; `"17 + 25? A:"` →
  `" 42"`; colors prompt on-topic.
- **(B) kill-switch = BIT-STOCK:** `disable` path (no swap, ts dropped) and a fully-unpatched
  stock load both confirm `type(mixer) is NemotronHMoE` and produce **byte-identical token
  sequences across all 3 prompts** (proven, not assumed; stock output is mis-scaled — the
  requirement is only that the two stock paths MATCH).
- (Argmax-agreement vs oQ4e is a weak cross-quant check; primary gate = D2 exactness +
  coherent generation, both met.) Then STOP — no server integration.

---

## 8. Open items — resolved by D2
1. ✅ Source `up_proj.weight` `[2688,512]`U8 → `view→[2688,128]`U32; `down_proj`
   `[1024,1344]`U8 → `[1024,336]`U32. Byte-identity confirmed.
2. ✅ `weight_scale` byte-copies as E4M3 (value-diff 0.0 confirms interpretation).
3. ✅ FP8 `weight_scale` is a per-tensor scalar (size 1).
4. ⏳ `mtp.*` drop — the converter never *reads* mtp tensors (it only iterates
   `backbone.layers.*` by the hybrid pattern + 3 named globals), so nothing mtp is emitted
   regardless of exact names. Belt-and-suspenders confirmed by construction.

## 9. Implementation notes
- New sibling `omlx/tools/oqnvfp4_nemotron_convert.py` (leave M3 `oqnvfp4_convert.py`
  untouched). Reuses `TensorIndex`, `ShardWriter`, `repack_nvfp4`, `scalar_f32`,
  `cast_to_bf16/f32`, `copy_aux_files` from the M3 module.
- Adds: `e4m3fn_lut()` (+anchor asserts), `dequant_fp8_pertensor()`, `to_bf16()` dispatch,
  `convert_{mamba,attn,moe}`, `convert_routed_experts` (stack fc1/fc2 + ts + positivity
  assert), hybrid-pattern driver. `--allow-partial`/`--limit-layers` for smoke tests.
- OPS: output → `~/.omlx/models/unigilby/Nemotron-3-Super-oQNVFP4`; scratch only under
  `$OMLX_COLD_STORAGE/omlx-quant-work/`; no git; server/tmux untouched.

## 10. Runtime ts-fold patch — IMPLEMENTED & VERIFIED via the real omlx load path

Exact inference re-applies each expert's `weight_scale_2` by folding the per-expert scalar
`c_e = fc1_ts²·fc2_ts` into the router scores. Landed as:

- **`omlx/patches/nemotron_h_nvfp4_ts.py`** — `apply_nemotron_h_nvfp4_ts_patch()`. Wraps
  `mlx_lm.models.nemotron_h.Model.sanitize` **once** (idempotent, no import-time side
  effects). The wrapper **self-gates on the presence of `*.fc1_ts` keys** in the checkpoint
  weights: a ts checkpoint gets a **per-instance** `mixer.__class__ = _TsFoldMoE` swap +
  `fc1_ts`/`fc2_ts` registration on THIS model's MoE layers; any other nemotron_h model
  (affine oQ4e, Nano — no ts keys) falls through to stock sanitize untouched. `sanitize`
  runs before `nn.quantize`/`load_weights` in mlx_lm `load_model`, exactly when the
  instances exist and the ts params must be in the tree for strict load.
- **`omlx/utils/model_loading.py`** — dispatched in `maybe_apply_pre_load_patches`, gated on
  `config["omlx_moe_nvfp4_ts"] and model_type == "nemotron_h"` (mirrors the other pre-load
  patches).
- `_moe_call` body is copied verbatim from **mlx_lm 0.31.3 `NemotronHMoE.__call__`
  (nemotron_h.py lines 408–424)** + 2 fold lines (`# +fold`) — re-diff on any mlx_lm upgrade.

**No `cast_predicate` change needed:** mlx_lm 0.31.3 `load_model` binds weights **as-stored**
(no `set_dtype`/cast pass), and the converter writes `fc1_ts`/`fc2_ts` as **F32** — they load
fp32 automatically. (Left the fp32-exclusion note here in case a future mlx_lm adds a load
cast.)

**Kill-switch `OMLX_NEMO_DISABLE_NVFP4_TS=1` (final reconciled semantics):** registration +
per-instance swap STILL happen (so strict `load_weights` binds the 80 ts params), but the
fold multiply is skipped inside `_moe_call`. Output is then mis-scaled — a DEBUG/attribution
switch, **not** a correctness mode. (With the fold skipped, `_moe_call` == stock lines
408–424, so the math equals stock.)

**Verified via the REAL path** (`omlx-quant-work/d4_real.py`:
`maybe_apply_pre_load_patches(path)` → `mlx_lm.load(path)`, no monkeypatch):
- NVFP4 model: strict load, all **40 MoE layers = `_TsFoldMoE`**, `NemotronHMoE.__call__`
  NOT rebound, no NaN, coherent+correct gen (`" Paris."`, `" 42"`).
- **Non-interference:** affine oQ4e loaded in the SAME process → all 40 MoE layers stay
  **stock `NemotronHMoE`**, generates fine. The MTP sanitize patch (Nemotron declares
  `num_nextn_predict_layers=1`) and this patch compose without conflict.

## 11. Live server bench (D5) — oQNVFP4 vs oQ4e (`nemotron-3-super`)
Both served on the restarted production server (M3 Ultra); ts-fold engaged live (log line 82,
fired once on oQNVFP4 load; absent on oQ4e load → non-interference confirmed live).

| metric | oQ4e (affine) | oQNVFP4 (nvfp4) |
|---|---|---|
| disk | 68.6 GB | 77.5 GB |
| decode short / 5k(warm) | 49.4 / 50.4 tok/s | 29.8 / 29.4 tok/s (**0.58×**) |
| TTFT ~5k | 3.9 s (SpecPrefill) | 7.2 s (no spec cfg) |
| gsm8k SERIAL | 92% (11/12) | 92% (11/12) |
| mmlu n=100 / arc n=100 | 81% / 95% | 80% / 94% |

- **Quality PRESERVED** — statistically tied on all three benches. Lossless repack + exact
  relu² ts-fold is correct in live serving.
- **Speed** — oQNVFP4 decode ~0.58× oQ4e: the nvfp4 `gather_qmm` (gs16 = 4× scale reads of
  affine gs64, + E4M3 + fp4 decode) vs the mature affine-int4 kernel. Kernel-opt opportunity,
  not correctness. TTFT gap is just oQ4e's SpecPrefill (draft=Nano); oQNVFP4 has none configured.
- **Artifact flagged (NOT quantization):** acc_bench's *concurrent* (6-worker) gsm8k scored
  oQ4e 15% — a SpecPrefill-under-concurrency degradation; single-shot + serial both = 92%.
  Reported the clean serial number.

## 12. Nemotron-3-Ultra-550B deltas (pathfinder → Ultra)
Source `Nemotron-3-Ultra-NVFP4` (352 GB, 113 shards). Same NemotronH arch/tensor layout as
Super — the converter is **dim-agnostic** (all dims read from config/tensors), needing only a
pattern-source generalization (**done + verified**):

| | Super-120B | Ultra-550B |
|---|---|---|
| hidden / latent / moe_int / shared_int | 4096 / 1024 / 2688 / 5376 | **8192 / 2048 / 5120 / 10240** |
| layers (M / * / E) | 88 (40/8/40) | **108 (48/12/48)** |
| n_routed_experts / topk | 512 / 22 | 512 / 22 |
| mamba_num_heads / attn heads | 128 / 32 | **256 / 64** |
| pattern key | `hybrid_override_pattern` (str) | **`layers_block_type` (list)**, `num_hidden_layers=None` |

**Converter change (Super-safe, verified on both scouts):** `resolve_pattern(cfg)` accepts the
string OR the `layers_block_type` name-list (`{mamba:M, attention:*, moe:E, mlp:-}`); the
`num_hidden_layers` assert is skipped when None. No other change — `to_bf16` dispatch
(weight_scale_2→nvfp4, weight_scale→fp8, else bf16) auto-handles Ultra's slightly different
per-tensor quant mix (latent projs all bf16, in/out_proj all FP8, o_proj all bf16, **no**
nvfp4 shared experts). `NVFP4_GROUP_SIZE=16` universal. MTP (`mtp_layers_block_type`
`[attention, moe]`) dropped as before.

**Stacked expert shapes (nvfp4):** fc1(up) weight `[512, 5120, 256]` U32 + scales
`[512, 5120, 128]` U8; fc2(down) weight `[512, 2048, 640]` U32 + scales `[512, 2048, 320]` U8.

**Output size:** nvfp4 experts **289.9 GB** + bf16 rest **67.8 GB** + ts 0.2 MB = **~357.7 GB**
(source 352 GB). Alt affine8-non-experts ≈ 322 GB (headroom only, not built).

**Disk fit:** internal free **656 GB**, output ~358 GB → **~298 GB margin** (source on the T7
external, not internal). Fits.

**Runtime:** Super ≈ 79 GB in ~10 min → Ultra ~4.5× data ≈ **45–60 min**.

**Streaming memory:** `TensorIndex.load` reads per-tensor (never a whole shard resident);
peak = one MoE layer's stacked experts **~6 GB** + ShardWriter pending (≤5 GB) ≈ 10–15 GB.
ts-positivity assert runs per-MoE-layer on the `[512]` stack; LUT self-check once — both scale
trivially to 113 shards. Use `--allow-partial` to smoke-test on landed shards mid-download.

**GATE:** do NOT start conversion until the lead confirms the disk-fit numbers AND opens the
offline window (352 GB standalone load cannot coexist with a loaded server model — server pool
must be empty/stopped for the Ultra D4 load test).
