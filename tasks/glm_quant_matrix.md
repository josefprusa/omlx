# GLM-5.2 per-component quantization matrix

`glm_moe_dsa` — MLA attention + DeepSeek-style sparse "lightning indexer" (DSA) + MoE (first 3 layers dense, 256 routed experts + 1 shared expert, 8 active) + 1 MTP/nextn layer. 78 hidden layers (0–77) + MTP as layer 78. 753.3B total params, ~40B activated. Native training precision **BF16**.

| Component | NATIVE `zai-org/GLM-5.2` | NATIVE-FP8 `zai-org/GLM-5.2-FP8` | NVIDIA `nvidia/GLM-5.2-NVFP4` | LOCAL `avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw` |
|---|---|---|---|---|
| **Embeddings** (`model.embed_tokens`) | BF16 | **BF16** (in `modules_to_not_convert`) | **BF16** (in `ignore`) | **MLX affine 6-bit, gs64** |
| **MLA attention** (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`) | BF16 | **FP8 E4M3, block 128×128**, dynamic act (`weight_scale_inv`) | **BF16** — all `self_attn*` excluded; attention is *not* FP4 | **MLX affine 4-bit, gs64** (stored as `q_a_proj,q_b_proj,kv_a_proj_with_mqa,embed_q,unembed_out,o_proj`; `kv_b` folded into `embed_q`+`unembed_out` via MLA absorption) |
| **DSA / lightning indexer** (`indexers_proj`, `indexer.wk/wq_b/weights_proj`, `indexer.k_norm`) | BF16 | **BF16** — `indexers_proj` + `indexer.k_norm` explicitly in `not_convert` (kept high-precision while rest of attention is FP8) | **BF16** — folded under blanket `self_attn*` exclusion | **BF16 on main stack** (not in quant dict). **MTP-layer indexer = 4-bit** (see MTP row) |
| **Dense MLP** (layers 0–2, `first_k_dense_replace=3`, `intermediate=12288`) | BF16 | **FP8 E4M3, block 128×128** (dense MLP weights not excluded) | **BF16** — layers `0*/1.*/2.*` fully excluded | **MLX affine 4-bit, gs64** (`gate_proj/up_proj/down_proj`) |
| **Routed experts** (256 × `gate/up/down`, `moe_intermediate=2048`, layers 3–77) | BF16 | **FP8 E4M3, block 128×128**, dynamic act (this is ~751B of the params) | **NVFP4** — FP4 (E2M1) values, gs16, static; E4M3 per-16 block-scales + FP32 `weight_scale_2` + static FP32 `input_scale`. **The only quantized component.** | **MLX affine 3-bit, gs64** (`switch_mlp gate/up/down`) — the lowest-bit component |
| **Shared expert** (1 × `gate/up/down`) | BF16 | **FP8 E4M3, block 128×128** (shared expert *is* quantized) | **BF16** — `mlp.shared_experts*` explicitly excluded (card: "shared expert is not quantized") | **MLX affine 4-bit, gs64** |
| **Router / gate** (`mlp.gate.weight` + `e_score_correction_bias`) | BF16 weight; **F32** correction bias | **BF16** — every `mlp.gate` + `e_score_correction_bias` in `not_convert` | **BF16** — lands in the 28.55B BF16 remainder (not FP4); card: only experts quantized | **F32/BF16, unquantized** (kept full precision) |
| **Norms** (all RMSNorm: `input/post_attention/q_a/kv_a` layernorms, `model.norm`) | BF16 | **BF16** (all in `not_convert`) | **BF16** (not Linear → not targeted) | **BF16** (not in quant dict) |
| **lm_head** | BF16 | **BF16** (in `not_convert`) | **BF16** (in `ignore`) | **MLX affine 6-bit, gs64** |
| **MTP / nextn** (layer 78: `enorm,hnorm,eh_proj,shared_head`, + a full decoder block) | BF16 | attn+experts **FP8 E4M3 128×128**; norms/`eh_proj`/`mlp.gate`/indexer **BF16** | **BF16** — entire `model.layers.78*` excluded (shipped but unquantized) | **MLX affine 4-bit, gs64** — grafted from `inferencerlabs/GLM-5.2-MTP-MLX-Q4`; attn/`switch_mlp`/shared/`indexer(wk,wq_b,weights_proj)`/`eh_proj` all 4-bit U32; norms + router BF16. *Not in base config's quant dict — loaded via omlx MTP patch.* |

## Footnotes

**[1] On-disk size / params (from HF `?expand=safetensors&expand=usedStorage`, and local `du`):**
- **Native BF16** — 1,506,687,604,850 B ≈ **1.507 TB (1.37 TiB)**; 753,329,940,480 params. Dtype: BF16 everything **except 19,456 F32** = the 76 router `e_score_correction_bias` tensors (256 × 76 gates).
- **FP8** — 761,025,363,709 B ≈ **761 GB (709 GiB)**; 753.38B params = **751.23B F8_E4M3** + 2.10B BF16 + 45.9M F32.
- **NVFP4** — 464,865,454,030 B ≈ **465 GB (433 GiB)**. Dtype: **362.39B U8** (= 724.78B packed FP4 expert weights) + **45.30B F8_E4M3** (block scales) + **28.55B BF16** (all non-expert weights) + 19,456 F32.
- **LOCAL MLX 3.5bpw** — **≈311 GiB on disk** (`du -sh`): 76 base shards (~305 GiB) + one 5.6 GiB grafted MTP shard. MLX affine, `group_size=64` throughout. Bit tally over 929 quantized tensors: **3-bit ×225** (routed experts), **4-bit ×702** (shared experts, all MLA, dense MLP, MTP), **6-bit ×2** (embed + lm_head); routers/norms/main-stack-indexer left BF16. "3.5 bpw" is the routed-expert-dominated weighted average.

**[2] NVFP4 closure (the headline result):** BF16 remainder = 28.55B = 753.33B total − 724.78B routed-expert params, *exactly*. NVFP4 quantizes **only the routed experts** and leaves **everything else BF16** — including all MLA attention, the **DSA lightning indexer**, and the **entire MTP layer**. That is why a "4-bit" export is still 465 GB. This matters for our conversion planning: NVFP4 gives us **no** FP4 reference for attention, indexer, or MTP.

**[3] NVFP4 `hf_quant_config.json` `ignore` list (gold):** `lm_head`, `model.embed_tokens`, `model.layers.0*`, `model.layers.1.*`, `model.layers.2.*`, then for every layer 3–77 both `…self_attn*` and `…mlp.shared_experts*`, and finally `model.layers.78*`. `quant_algo=NVFP4`, `group_size=16`, `kv_cache_quant_algo=FP8` (num_bits 8, E4M3 — runtime KV, `--kv-cache-dtype fp8_e4m3`, not a stored-weight quant). Producer: `modelopt 0.46.0`. The router `mlp.gate` is *not* explicitly listed but is tiny (~118M params) and stays BF16.

**[4] FP8 (`zai-org`) is DeepSeek-style block FP8:** `fmt=e4m3`, `weight_block_size=[128,128]`, `activation_scheme=dynamic`. Its ~600-entry `modules_to_not_convert` preserves BF16 for: all norms, all routers (`mlp.gate` + `e_score_correction_bias`), the **DSA indexer projections** (`self_attn.indexers_proj`) and indexer `k_norm`, embeddings, `lm_head`, and the MTP layer's norms/`eh_proj`/`gate`/indexer. Note it keeps the **indexer BF16 while quantizing the rest of attention** — the indexer is treated as precision-sensitive.

**[5] Two surprises worth flagging:**
- **Shared expert**: FP8 release *quantizes* it (FP8); NVFP4 release *keeps it BF16*. Opposite calls.
- **Attention**: NVFP4 leaves *all* attention BF16 (nothing in `self_attn` is FP4), whereas FP8 quantizes the MLA projections. So NVFP4 ≈ "FP4 experts + BF16 shell"; FP8 ≈ "FP8 nearly-everything, BF16 norms/router/indexer".

## Sources
- **Native**: `https://huggingface.co/zai-org/GLM-5.2/resolve/main/config.json`; API `…/api/models/zai-org/GLM-5.2?expand=safetensors&expand=usedStorage`.
- **FP8**: `https://huggingface.co/zai-org/GLM-5.2-FP8/resolve/main/config.json` (`quantization_config` + `modules_to_not_convert`); API `…/api/models/zai-org/GLM-5.2-FP8`.
- **NVFP4**: `…/nvidia/GLM-5.2-NVFP4/resolve/main/hf_quant_config.json`, `…/config.json`, `…/README.md`; API `…/api/models/nvidia/GLM-5.2-NVFP4`.
- **Local**: `$HOME/.omlx/models/avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw/config.json` (`quantization` dict, 929 per-module entries) + safetensors shard headers (shards 00001/00038/00076 + `model-mtp-00001`); MTP source `…/inferencerlabs/GLM-5.2-MTP-MLX-Q4/config.json`.
