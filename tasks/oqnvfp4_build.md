# oQ-NVFP4 repack pipeline — build log

Goal: convert `nvidia/MiniMax-M3-NVFP4` → omlx-servable MLX checkpoint at
`~/.omlx/models/unigilby/MiniMax-M3-oQNVFP4`, preserving NVIDIA's calibrated
quantized values EXACTLY via byte-repack. Branch `glm5.2-native-kernels-v0.4.5`,
do NOT commit.

## Status: RESEARCH DONE · BUILDING

Download: 36/88 shards present (non-contiguous: 1-34,36,38), ~47GB. Build + unit
tests run on present shards; full run deferred until download completes.

---

## PROVEN (this session)

- **Byte-repack bit-exact** (unit test #1, `/tmp` proof reproduced on present shards):
  - nvfp4 experts: codes U8[out,in/2].view(uint32) → weight; weight_scale
    F8_E4M3 bytes as uint8 → scales; biases=None; g16/b4/mode=nvfp4.
    `max|mx.dequantize − manual two-level| = 0.0` (3 tensors, lo-nibble-first).
  - mxfp8: fp8 E4M3[out,in].view(uint32) → weight; weight_scale_inv E8M0 bytes
    as uint8 → scales; g32/b8/mode=mxfp8. `max|diff| = 0.0` (3 tensors).
  - weight_scale_2 (f32 scalar per expert per proj) is the second-level scale
    NOT representable in MLX single-level nvfp4 → carried at runtime (ts).

## Load path (verified in .venv mlx 0.31.2 + patched mlx_vlm)

- omlx serves VLM via `mlx_vlm.utils.load_model` (installed copy is omlx-patched;
  has nvfp4/mxfp8 support + `_transform_compressed_tensors_*`).
- Order: build model → (`_transform_compressed_tensors_weights`: no-op for us) →
  sanitize (no-op on pre-fused tensors) → `nn.quantize(model, global, class_predicate)`
  → `load_weights(strict)`.
- `get_class_predicate(p,m)`: `if p in config["quantization"]: return config["quantization"][p]`
  (per-module dict {bits,group_size,mode} or False); elif no `to_quantized`→False;
  elif `m.weight.size%64!=0`→False; else `f"{p}.scales" in weights`.
  ⇒ modules with **no config entry AND no `.scales`** stay float (this is how
  oQ4's `gate.weight` stays bf16; norms/vision too).
- MLX quantize output shapes (verified): nvfp4 g16b4 → weight uint32[out,in/8],
  scales uint8[out,in/16], NO biases. mxfp8 g32b8 → weight uint32[out,in/4],
  scales uint8[out,in/32], NO biases. affine g64b8 → uint32[out,in/4] + bf16
  scales/biases[out,in/64].

## Source inventory (from range-probed index, 60 layers = 3 dense + 57 MoE)

- routed experts `…block_sparse_moe.experts.{0..127}.{w1,w2,w3}` (w1=gate[3072,6144],
  w3=up[3072,6144], w2=down[6144,3072]): NVFP4 (.weight U8, .weight_scale E4M3,
  .weight_scale_2 F32, .input_scale F32 [unused on MLX]).
- shared `…shared_experts.{gate,up,down}_proj`: MXFP8 (.weight F8_E4M3, .weight_scale_inv E8M0).
- attn `self_attn.{q,k,v,o}_proj` + `index_{q,k}_proj`: MXFP8.
- dense `layers.{0,1,2}.mlp.{gate,up,down}_proj`: MXFP8.
- `block_sparse_moe.gate.weight`: F32 (unquantized). norms: bf16/f32.
- embed_tokens / lm_head: bf16. vision_tower.* + projectors: bf16 (+biases).
- `e_score_correction_bias`: f32.

## oQ4 = mirror target (ground truth from its safetensors)

- embed/lm_head/attn/index/dense-mlp: **affine8 g64** (uint32 + bf16 scales+biases).
- switch FUSED: `switch_mlp.gate_up_proj.weight` U32[129,6144,768] (+scales/biases
  bf16[129,6144,96]), `down_proj` U32[129,6144,384]. 129 = 128 routed + shared@128.
  gate_up out-axis = [gate(3072); up(3072)] (split axis=-1 → first half gate).
- gate.weight bf16[128,6144] NO scales (unquantized). norms bf16. vision bf16.
- Fusion done by `_sanitize_moe_weights` (only when experts.M.* present); oQ4 ships
  pre-fused so sanitize is a no-op. n_shared==1 & shared_int==int ⇒ pack_shared.

## DESIGN

### Converter `omlx/tools/oqnvfp4_convert.py` (standalone; imports oq helpers)
Per-layer streaming (bounded mem), tensor→shard map from present shard headers.
Emits FUSED tensors matching oQ4 layout + ts sidecars.

Policy (flags):
| module | default | mode |
|--------|---------|------|
| routed experts | byte-repack nvfp4, FUSE→switch_mlp.gate_up_proj[128]/down_proj[128] | nvfp4 g16b4 + ts sidecar |
| shared experts | dequant mxfp8→bf16 → requant | **separate** module, affine8 g64 |
| attn q/k/v/o + index | dequant mxfp8→bf16 → requant | affine8 g64 |
| dense mlp | dequant mxfp8→bf16 → requant | affine8 g64 |
| gate.weight | f32→bf16 keep | float (no scales) |
| norms | keep bf16 | float |
| embed/lm_head | bf16 → requant | affine8 g64 |
| vision + projectors | pass-through bf16 | float |

Flags:
- `--attn-native-mxfp8`: attn/index byte-repack mxfp8 (disables M3 packed-proj; warn).
- `--fuse-shared-nvfp4`: shared→nvfp4 requant, fuse as expert 128 (switch[129]);
  ts sidecars [129,2]/[129]. Reuses existing pack_shared structure.
- default: separate shared (switch[128] + separate affine8 shared).

ts sidecars (per MoE layer): `…switch_mlp.gate_up_ts` f32[E,2]=[gate_ts,up_ts],
`…switch_mlp.down_ts` f32[E]. (E=128 separate / 129 fused; shared slot ts=1.)

config.json: mirror source config; `quantization` = {global affine8 g64} + explicit
nvfp4 entries for every `switch_mlp.gate_up_proj`/`down_proj`; affine8 modules fall
through via `.scales` presence; gate/norms/vision left out (stay float).
New keys (text_config): `omlx_moe_shared_expert_mode` ("separate"|"fused"),
`omlx_moe_nvfp4_ts` (bool). Absent ⇒ legacy behavior (oQ4 unaffected).

### Runtime patch (vendored minimax_m3_vl)
- `config.py TextConfig`: add `omlx_moe_shared_expert_mode: Optional[str]=None`,
  `omlx_moe_nvfp4_ts: bool=False`.
- `language.py MiniMaxSparseMoeBlock`:
  - separate mode: switch_mlp = MiniMaxPackedSwitchGLU(num_experts=128) [fused
    gate_up, routed only] + separate shared_experts MiniMaxMLP; DON'T append shared.
  - ts (when omlx_moe_nvfp4_ts): register `switch_mlp.gate_up_ts`[E,2],
    `switch_mlp.down_ts`[E]; fold down_ts into scores (scores*=down_ts[inds], exact);
    pass gate_up_ts into MiniMaxPackedSwitchGLU (multiply gate_up output in fp32
    before activation, expand [E,2]→[E,2I] gather by idx).
  - env kill: `OMLX_M3_DISABLE_NVFP4_TS=1` → skip ts (model WRONG, warn).
- Backward compat: new keys absent ⇒ no ts buffers registered ⇒ oQ4 loads strict.

## VERIFICATION results
- [x] #1 byte-equality + dequant-parity (nvfp4×3, mxfp8×3 on real shards): PASS max|diff|=0.0
- [x] #2 ts-carry algebra (synthetic): score-fold + gate_up-mul == two-level dequant MoE, rel=1.4e-7 (fp32)
- [x] #2b ts-carry via REAL patched code path (separate mode, quantized switch): rel=3.9e-3 (bf16-activation level; algebra exact)
- [x] STRUCTURAL: legacy(no keys)=pack_shared 129 no-ts UNCHANGED; separate=128+sep-shared+ts; fused=129+ts
- [x] KILL SWITCH OMLX_M3_DISABLE_NVFP4_TS: bypass active (63% diff) + warns once
- [x] CONVERTER OUTPUT (real, 5-layer partial): nvfp4 dequant==source max|diff|=0.0;
      ts sidecars==source weight_scale_2; true-weight(dequant*ts) parity max|diff|=0.0;
      attn affine8+biases; separate shared present; gate bf16 no-scales; config correct
- [x] #3a LOAD via mlx_vlm.load_model (5-layer truncated): model builds; switch=MiniMaxPackedSwitchGLU
      nvfp4 g16 b4; gate_up_ts[128,2]/down_ts[128] bound; separate shared; attn affine; forward logits FINITE
- [x] LIVE ts-engagement (loaded model, ts ON vs OFF): rel=1.58 -> ENGAGED LIVE (addresses live-path lesson)
- [x] #4a py_compile language.py + config.py: OK
- [x] #3b FULL-MODEL coherent generation: converted 248GB oQNVFP4 loaded (lazy) + generated
      "...the capital of France is Paris." (correct + coherent). Census: pack_qkv=3, pack_full=57
      -> M3 packed-projection fast path ENGAGES with affine8 attention (validates default policy).
      NOTE: requires env MLX_MAX_OPS_PER_BUFFER=500 (without it: Metal GPU command-buffer TIMEOUT).
- [x] #4b oQ4 decode probe (patched code): layer17 pack_shared=True 129-expert NO-ts UNCHANGED;
      generated "...Paris." coherent -> backward-compat CONFIRMED.
- [x] #3c SERVER chat completion: restarted server w/ OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=500;
      loaded MiniMax-M3-oQNVFP4 (260.4GB resident); chat -> "The capital of France is Paris, home to the
      iconic Eiffel Tower". Long-form coherent; ~17 tok/s aggregate (incl prefill). Model left LOADED for
      lead's benchmark gauntlet.

## STATUS: COMPLETE — all verification PASS. Server running w/ patch, oQNVFP4 loaded, ready for handoff.

## FULL CONVERSION RESULT
Done: 60 layers, 2942 tensors, 248.0 GB, 58 shards at ~/.omlx/models/unigilby/MiniMax-M3-oQNVFP4.
Structural: dense=3 moe=57 ts_layers=57 vision=515; flags separate/ts=True; global affine8 g64.

## FULL CONVERSION (default: separate-shared + nvfp4 ts)
Running: PID logged in /tmp/oqnvfp4_fullrun.log. Command:
```
.venv/bin/python omlx/tools/oqnvfp4_convert.py \
  --src "$OMLX_COLD_STORAGE/omlx-quant-work/MiniMax-M3-NVFP4" \
  --out ~/.omlx/models/unigilby/MiniMax-M3-oQNVFP4 --shard-size-gb 8
```
Variants: add `--fuse-shared-nvfp4` (shared→nvfp4 @128, 129-switch) or `--attn-native-mxfp8`.

## Extra fix: omlx engine MoE sanitize (the spec's "config-driven skip")
`omlx/engine/vlm.py` `_pack_mlx_unpacked_moe_weights` (the "MiniMax M3 MLX-format MoE
sanitize patch") force-fuses `shared_experts.down_proj` into `switch_mlp.down_proj`
whenever both keys exist. My separate-shared checkpoint legitimately has BOTH (routed
nvfp4 down [128,·,384] + separate affine8 shared down [·,768]) -> concat shape crash
`(128,6144,384) vs (1,6144,768)`. FIX: skip packing when
`args.omlx_moe_shared_expert_mode is not None` (checkpoint already in final layout).
Harmless for legacy oQ4 / old unpacked 4bit (key absent -> packing runs as before).
Requires server restart to pick up (server imports omlx.engine.vlm at startup).

## Fix applied to codex draft
- Vision/global tensors were force-flushed one-file-per-tensor (536 files, 327 tiny).
  Removed force_flush so ShardWriter packs to --shard-size-gb. Correctness unaffected.

## Deviations from spec
- dense mlp (layers 0-2): treated like shared → affine8 g64 (spec didn't name them;
  matches oQ4 + MXFP8 source; not experts, no fast-path needs nvfp4).
- Converter emits FUSED switch tensors directly (mirrors oQ4) rather than per-expert +
  load-time sanitize fusion — bounded memory per layer, matches proven oQ4 load profile.
