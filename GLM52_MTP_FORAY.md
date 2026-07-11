# GLM-5.2 MTP Speculative Decoding on omlx/MLX (M3 Ultra) — Investigation & Open Problem

**Status:** native MTP self-speculative decoding *works and is lossless*, but on a **single-user, batch-1
M3 Ultra (MLX)** it lands at **~0.85× baseline** (slower), blocked by a **diffuse per-token MLX execution
overhead** in the multi-token "verify" forward. We disproved the "fundamental MoE wall" hypothesis and
fixed the two big component overheads (attention + indexer), but the residual is not attributable to any
single component. This doc is a complete briefing to find the **maximum-gain fix** that crosses ≥1.3×.

---

## 0. The Ask (for GPT-5 Pro)

A 2-token "verify" forward through the GLM-5.2 trunk costs **~1.9× a 1-token decode** on M3 Ultra/MLX
(batch 1), even though:
- the extra token's **weight read** is ~+10 ms (it activates ~8 more of 256 experts), and
- the extra token's **FLOPs** are trivial (~1 GFLOP, <0.1 ms at 27 TFLOP/s), and
- **identical** and **diverse** verify tokens cost the **same** (so it's *not* expert-activation/MoE-weight),
  and the attention + indexer paths are already specialized for small L.

The literature says single-user MoE spec decode *should* reach ~2× (see §6). On GPUs decode is
weight-bandwidth-bound so the verify's extra tokens are nearly free; on M3 Ultra/MLX at batch 1 the decode
is **overhead/compute-bound** and the 2-token verify pays ~2×. **We need to eliminate the ~37 ms/token
"work" that is far above the actual FLOPs/bandwidth** — see the cost decomposition in §7.

**Concrete questions:**
1. What is the most likely source of the ~37 ms/token MLX overhead in an L=2 forward of a 78-layer MoE, and
   how do we remove it (mx.compile of the decode step despite stateful KV cache? a fused multi-token decode
   kernel? a known MLX `gather_qmm` / `scaled_dot_product_attention` small-shape inefficiency)?
2. Why is an isolated eager `model(x, cache)` call ~**3100 ms** while the *same* call in omlx's
   executor-thread/generation loop is ~**48 ms** (65×)? (codex says it's *not* `mx.compile`.) If we
   understood that 65× optimization, could we apply it to the L=2 verify?
3. Is there a fundamentally better MTP/verify formulation for this regime (e.g., run the verify as the
   optimized L==1 path applied to each token but with shared weight loads; a tree/medusa scheme; EAGLE-style;
   or restructure so verify ≈ 1 decode)?
4. Anything model-/quant-specific (3-bit routed experts, MLA absorbed form, DSA top-2048 indexer) that opens
   a cheaper verify?

Target: **≥1.3× decode** (≈27 tok/s) at single-user batch-1, lossless under greedy. Hardware fixed
(M3 Ultra 512 GB, MLX). Everything below is measured on this machine.

---

## 1. Setup

- **Model:** `avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw`, `model_type = glm_moe_dsa` (a DeepSeek-V3.2 derivative).
  - hidden 6144, 78 layers (first 3 dense, 75 MoE), **256 routed experts, top-8, 1 shared expert**,
    moe_intermediate 2048, vocab 154880.
  - **MLA** attention: `kv_lora_rank=512`, `qk_rope_head_dim=64`, 64 heads. **DSA** sparse attention with an
    **indexer** selecting `index_topk=2048` keys (32 index heads, head_dim 128).
  - Mixed quant: routed experts **3-bit g64**, MLA/shared/dense **4-bit g64**, lm_head/embed 6-bit.
- **MTP head** (grafted from HF `inferencerlabs/GLM-5.2-MTP-MLX-Q4`): `eh_proj` (Linear 2·6144→6144, Q4) +
  `enorm`/`hnorm`/`norm` (RMSNorm) + **one full GLM-DSA decoder layer**; reuses the trunk's `embed_tokens`
  + `lm_head`. Grafted under top-level `mtp.0.*` keys.
- **Hardware/runtime:** Apple **M3 Ultra, 512 GB, ~819 GB/s**, MLX (mlx-metal 0.31.2), native GLM Metal
  kernels active (sparse-MLA, exact-block, indexer, 4-bit V-up, weighted-sum). **Single-user, batch 1.**
- **Server:** omlx (FastAPI, continuous batching). Its MTP path (`omlx/patches/mlx_lm_mtp/`) does
  **1 draft + 2-token verify** per cycle and only batches *aligned* sequences (so the literature's
  batch-4-16 MoE sweet spot is not readily reachable).

### Baseline (MTP off), measured

| context | decode tok/s |
|---|---|
| short (~300) | **21.0** |
| 16k | **19.6** |
| 64k | **18.2** |

---

## 2. How MTP self-speculation works here

Per cycle (single sequence):
1. **Draft:** `mtp_logits = model.mtp_forward(trunk_hidden_t, next_token, mtp_cache)` — runs the 1-layer MTP
   head → 1 draft token. Cheap: measured **~2.4 ms** (the head is one Q4 layer).
2. **Verify:** `logits = model(concat([next_main, draft]), cache=prompt_cache)` — a **2-token (L=2)** forward
   through the full 78-layer trunk. `verify_logits = logits[:,0]`, `bonus_logits = logits[:,1]`.
3. **Accept/reject (greedy):** accept iff `argmax(verify_logits) == draft`. On accept emit 2 tokens
   (draft + bonus); on reject emit 1 (the corrected verify token) and `cache.trim(1)`.

Emitted/cycle = `1 + α` (α = accept rate). Speedup ≈ `(1+α)·t_decode / (t_verify + t_mtp_head)`.

**This is lossless under greedy** (verified: per-position `argmax` of the verify forward == the trunk's true
greedy, `argmax_agree = 1.000`). The only question is whether `t_verify` is cheap enough.

---

## 3. Acceptance is NOT the problem (measured live, per prompt)

| prompt | live α (greedy) |
|---|---|
| factual list | **80.9%** |
| code (LRU cache) | **77.6%** |
| prose (story) | **65.6%** |
| merge_sort | 41.1% (outlier) |

Mean ~66%; matches DeepSeek's reported 85-90% ballpark for in-distribution prompts. `mtp_head` cost ~2.4 ms
(negligible). **So the entire problem is `t_verify`.**

---

## 4. The core finding: verify cost vs token-width

Width sweep in the **real executor-thread regime** (w1 == a real decode; warm/steady medians):

| context | w1 (decode) | w2 (verify, 2 tok) | w3 | w4 |
|---|---|---|---|---|
| 512 | 50 ms | **131 ms (2.6×)** | 147 | 195 |
| 4096 | 55 ms | **102 ms (1.84×)** | — | 157 |
| 16k (live) | 51 ms | **~106 ms (2.08×)** | | |
| 64k (live) | 55 ms | **~131 ms (2.4×)** | | |

The jump is **w1→w2** (+50-80 ms); **w2→w3 is only ~+16 ms**. So there's a large *fixed L>1 penalty* plus a
modest per-token marginal.

### The decisive isolation: identical vs diverse verify tokens

If the cost were the MoE expert-doubling, *diverse* tokens (which hit ~2× distinct experts) would cost more
than *identical* tokens (same 8 experts). They cost the **same**:

| context | w1 | w2 **identical** | w2 **diverse** |
|---|---|---|---|
| 512 | 50 ms | **131 ms (2.5×)** | 131 ms (2.5×) |
| 4096 | 55 ms | **96 ms (1.74×)** | 98 ms (1.79×) |

**⇒ The verify cost is NOT expert-activation / MoE-weight.** (At 4096 there's a tiny ~+2 ms diverse effect;
negligible.) This is the key result that disproves the "MoE wall."

---

## 5. What we fixed (all lossless, `argmax_agree=1.000`), flag-gated `OMLX_MTP_VERIFY_FAST`

### 5.1 The L==1 vs L>1 attention asymmetry (the original catastrophe)

GLM-DSA decode (L==1) is hand-optimized: it projects **only the query** into latent space (`embed_q`) and
attends to the **raw** gathered top-2048 `kv_latent` (absorbed MLA). The L>1 path instead projects the
**entire** `kv_latent` (`embed_q(kv_latent)`, `unembed_out(kv_latent)`) → **O(context)** → verify exploded to
**414 ms @16k, 558 ms @4096-probe**.

The optimized L==1 path (`omlx/patches/glm_moe_dsa/glm_moe_dsa_model.py`, `GlmMoeDsaAttention.__call__`):

```python
if L == 1:
    topk_indices, _ = _parse_topk_state(topk_state)
    if topk_indices is not None:
        idx = topk_indices[:, :, 0, :, None]
        kv_latent = mx.take_along_axis(kv_latent, mx.broadcast_to(idx, ...), axis=2)  # gather top-2048
        k_pe      = mx.take_along_axis(k_pe,      mx.broadcast_to(idx, ...), axis=2)
    pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
    q_nope = self.embed_q(q_nope)                       # project ONLY the 1 query → latent
    output = scaled_dot_product_attention(q_nope, kv_latent, kv_latent, cache=cache,
                                          scale=self.scale, mask=pe_scores)
    output = self.unembed_out(output)                   # project ONLY the 1 output
    return self.o_proj(...), topk_state
```

**FIX — extend the absorbed form to `1 < L <= 8` (per-query topk gather, project only the L queries):**

```python
# after _parse_topk_state(...) and the indexer-cache mx.depends(...) edge:
if _small_l_verify_enabled() and topk_indices is not None and 1 < L <= 8:
    Kctx = kv_latent.shape[2]
    # (compact-prefix expansion omitted for brevity; prefix_rows==0 at K>index_topk)
    idx = topk_indices[..., None]                                  # [B,1,L,T,1]
    kv_src = mx.broadcast_to(kv_latent[:, :, None, :, :], (B, 1, L, Kctx, kv_latent.shape[-1]))
    pe_src = mx.broadcast_to(k_pe[:, :, None, :, :],      (B, 1, L, Kctx, k_pe.shape[-1]))
    kv_sel = mx.take_along_axis(kv_src, mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)), axis=3)
    kpe_sel = mx.take_along_axis(pe_src, mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)), axis=3)
    m4 = mask
    while m4.ndim < 4: m4 = m4[None]
    sel_mask = mx.take_along_axis(mx.broadcast_to(m4, (B, 1, L, Kctx)), topk_indices, axis=-1)  # [B,1,L,T]
    q_latent = self.embed_q(q_nope)                                # project ONLY L queries → [B,H,L,512]
    pe_scores = mx.sum((q_pe * self.scale)[:, :, :, None, :] * kpe_sel, axis=-1)                # [B,H,L,T]
    pe_scores = mx.where(sel_mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))
    # per-query attention via (B*L) batch, kv head=1 (MQA broadcast to H):
    q_flat   = q_latent.transpose(0, 2, 1, 3).reshape(B * L, self.num_heads, 1, q_latent.shape[-1])
    kv_flat  = kv_sel.transpose(0, 2, 1, 3, 4).reshape(B * L, 1, T, kv_sel.shape[-1])
    mask_flat = pe_scores.transpose(0, 2, 1, 3).reshape(B * L, self.num_heads, 1, T)
    out = scaled_dot_product_attention(q_flat, kv_flat, kv_flat, cache=None, scale=self.scale, mask=mask_flat)
    out = out.reshape(B, L, self.num_heads, kv_sel.shape[-1]).transpose(0, 2, 1, 3)
    out = self.unembed_out(out)                                    # project ONLY L outputs
    return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1)), topk_state
```

Mathematically equal to the unabsorbed reference within Q4 rounding (it's the same absorbed identity the
L==1 path uses). **Result: verify 414→106 ms @16k.** Context-independent attention.

### 5.2 The DSA indexer pads query rows to multiples of 64

For L>1 the indexer uses `fused_indexer_scores`, which **pads query rows to a multiple of 64** — so a 2-token
verify computes the indexer over **64 padded rows** (~32× waste), per "full" indexer layer. The L==1 path has
a cheap `s==1` branch that avoids this.

**FIX — extend the cheap `s==1` indexer math to small `s`, no padding** (`deepseek_v32.py`, `Indexer.__call__`):

```python
if _small_l_verify_enabled() and 1 < s <= 8:
    scores = q @ k.swapaxes(-1, -2)            # [b, n_heads, s, K]
    scores = mx.maximum(scores, 0)
    weights = (weights_lh * self.weight_scale).swapaxes(-1, -2)[..., None]   # [b, n_heads, s, 1]
    scores = (scores * weights).sum(axis=1, keepdims=True)                   # [b, 1, s, K]
    if mask is not None:
        scores = mx.where(mask, scores, -float("inf"))
    return select_topk(scores)                 # native dsa_topk_indices or argpartition over K
```

(Only fires at `K > index_topk` where every verify row has a full causal prefix → no prefix-row handling.)
**Result: verify 131→103 ms/cycle @64k.**

### 5.3 MoE sort threshold (no measurable effect)

`SwitchGLU` only sorts/uses the native weighted-sum at `indices.size >= 64`; a 2-token top-8 verify has 16
routes → unsorted `gather_qmm`. We lowered the threshold to 16 under the flag:

```python
do_sort = indices.size >= 64 or (_fast_verify and indices.size >= 16)
```

**Result: no measurable change** (consistent with §4's identical==diverse: the MoE is not the variable cost).

---

## 6. Literature reconciliation (why DeepSeek gets 1.8× and we don't)

- **DeepSeek-V3** ships MTP: **1.8× TPS, 85-90% acceptance** for the 2nd token
  ([report](https://arxiv.org/pdf/2412.19437), [SGLang MTP](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/mtp.html)).
- **Cohere** ("[MoE models get *more* from speculative decoding](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)"):
  non-monotonic curve, **~1.95× at BS=1**, peak ~2.2× at BS=4-16, declining at BS≥32 (compute-bound). Adjacent-token
  **expert overlap ~0.38** (vs 0.12 random) → verify activates few new experts; dense parts amortize.
- **MoESD** ([arxiv 2505.19645](https://arxiv.org/pdf/2505.19645)): MoE spec decode is beneficial at small
  batch (memory-bound) and *hurts* at large batch (compute-bound) when the verify activates more experts.

**Key:** their wins assume decode is **weight-bandwidth-bound**, so the verify's extra tokens are nearly free
(weights read once). On **M3 Ultra/MLX at batch 1**, a decode is **NOT** weight-bound — see §7. The 2-token
verify pays ~2× regardless of expert overlap (identical==diverse). This is a **runtime/regime** difference,
not the model.

---

## 7. THE OPEN PROBLEM — the residual diffuse overhead

After §5's fixes, with identical tokens (zero extra expert/weight work), at ctx 4096:

```
w1 (decode, L=1)      = 55 ms
w2 (verify, L=2)      = 96 ms   (1.74×)
```

Decompose `t(L) = F + W·L`:  `F + W = 55`,  `F + 2W = 96`  ⇒  **W ≈ 41 ms/token, F ≈ 14 ms fixed.**

Now the physics for **one extra (identical) token** through the 78-layer trunk:
- weight read: **~0 extra** (same experts, same dense weights) — confirmed by identical==diverse.
- FLOPs: ~1 GFLOP ⇒ **<0.1 ms** at 27 TFLOP/s.
- kernel-launch count: **unchanged** (same ops, bigger tensors) ⇒ fixed overhead should NOT scale with L.

Yet the marginal is **~41 ms/token**. None of weight / FLOPs / launch-count explains it. **This is the
mystery to crack.** Hypotheses (unverified):
- MLX kernels run far from peak for these tiny shapes, and the L=2 shape hits a *different, less-optimized*
  code path than the hyper-tuned L==1 decode (every op processes 2× elements at low efficiency).
- The MoE `gather_qmm` (3-bit, group_size 64) and/or MLA `scaled_dot_product_attention` have a small-batch
  inefficiency where 2 rows cost ~2× one row (e.g. per-row dequant of the 3-bit experts not amortized within
  a tile).
- The 65× standalone-vs-server gap (next bullet) is the same phenomenon and is the real lever.

**Unsolved sub-mystery (likely the key):** an isolated eager call
`model(mx.array([[tok]]), cache=cache)` on the main thread measures **~3100 ms**, but the *same* forward in
omlx's executor-thread generation loop is **~48 ms** (65×). codex investigated and says it's **not**
`mx.compile` (the decode step isn't compiled; only the router top-k is). What makes the in-loop decode 65×
faster, and can that optimization be made to apply to the L=2 verify? Relevant code:
`omlx/engine_core.py` (executor thread + thread-local `generation_stream`, the `@mx.compile` references),
`omlx/engine/batched.py` (decode step), `mlx_lm/generate.py` (`generate_step`, `async_eval`).

If the L=2 verify could be made as per-token-efficient as L=1 (W → ~10 ms, the true marginal),
`t_verify(2) ≈ 14 + 2·10 = 34 ms ≈ 1.2× decode` ⇒ at α=0.66, speedup `1.66/1.2 = 1.4×` ✅.

---

## 8. Current numbers, all 3 fixes (still < baseline)

| context | catastrophe | +all fixes | baseline | ratio |
|---|---|---|---|---|
| short | — | ~19 tok/s | 21.0 | 0.90× |
| 16k | 5 (0.27×) | **15.9** | 19.6 | 0.81× |
| 64k | 5 | **~15.5** | 18.2 | 0.85× |

Verify is still ~1.9× a decode (the §7 residual). All fixes lossless (`argmax_agree=1.000`).

---

## 9. Files & repro

**Branch:** `glm5.2-native-kernels-v0.4.5` (omlx). Changes (all flag-gated `OMLX_MTP_VERIFY_FAST`, default off;
MTP attach gated by `model_settings.mtp_enabled`):
- `omlx/patches/glm_moe_dsa/glm_moe_dsa_model.py` — small-L absorbed attention (§5.1) + `_small_l_verify_enabled`.
- `omlx/patches/glm_moe_dsa/deepseek_v32.py` — small-s non-padded indexer (§5.2) + helper.
- `omlx/patches/glm_moe_dsa/switch_layers.py` — MoE sort threshold (§5.3).
- `omlx/patches/mlx_lm_mtp/glm_moe_dsa_model.py` (NEW) — the glm_moe_dsa MTP head module (eh_proj/enorm/hnorm/
  norm + 1 GLM-DSA decoder layer; `mtp_forward`, `make_mtp_cache`, sanitize).
- `omlx/patches/mlx_lm_mtp/__init__.py`, `omlx/utils/model_loading.py` — register glm_moe_dsa MTP.
- `omlx/patches/mlx_lm_mtp/batch_generator.py` — diagnostics (width/ISO probes, env-gated).

**Measurement (executor-thread regime is essential — standalone eager is 65× slower, §7):**
- Per-prompt acceptance + verify cost are logged by the verify cycle:
  `MTP[uid] ... accept=X/Y ... timing[backbone=...ms mtp=...ms ...]`.
- Enable: set `mtp_enabled=true` for the model in `~/.omlx/model_settings.json`; restart
  `OMLX_MTP_VERIFY_FAST=1 uv run omlx serve`; send greedy requests; read the `MTP[...]` log lines.
- The width/identical-vs-diverse ISO probe is in `batch_generator._mtp_verify_iso_probe` (env `OMLX_MTP_ISO=1`).
- Native kernel build (already done; nanobind pinned 2.12.0 for ABI v19):
  `rm -rf build; CMAKE_ARGS="-DPython_EXECUTABLE=<repo>/.venv/bin/python" OMLX_WITH_CUSTOM_KERNEL=1 .venv/bin/python setup.py build_ext --inplace --with-custom-kernel`

**Verify correctness harness:** prefill >2048 tokens, run an L=2/3/4 trunk forward with the flag off
(reference L>1 path) vs on (small-L), compare per-position `argmax` (must be identical) + max|Δlogit|
(~0.1 rel, Q4 rounding).

---

## 10b. Probe results (ran GPT-5 Pro's plan) — diagnosis REFINED

Ran the synthetic microbenches (real GLM shapes, 3-bit/4-bit/6-bit quant, warm medians).

**Probe A — `gather_qmm` (routed experts, 3-bit g64), one MoE layer's gate_up+down:**
```
M1(8 routes)=0.43ms  M2same=0.61(1.40x)  M2div=0.62(1.42x)  M4=0.93(2.14x)   (sorted vs unsorted identical)
```
⇒ Row-wise scaling **confirmed** (M2same ≈ M2div ⇒ duplicate experts NOT coalesced), but the marginal is
only **~+13 ms** (×75 layers: 33→46 ms). Real, but **not** the dominant residual.

**Probe B — dense `quantized_matmul`, M=1 vs M=2, weighted by layer count:**
```
linear        M2/M1
lm_head        1.76x   (x1   → +1.0 ms total)
q_a_proj       1.00x   (x78  → ~0)
q_b_proj       1.05x   (x78  → ~0)
kv_a_proj      1.00x   (x78  → ~0)
o_proj         0.97x   (x78  → ~0)
shared g/u/d   ~1.00x  (x75  → ~0)
NON-MoE quantized-linear delta(2nd token) = +0.6 ms
```
⇒ **The dense quantized linears amortize weight-dequant across the 2 rows (~1.0×).** `lm_head` is the only
one that scales (1.76×) but it's a single op (~+1 ms). **lm_head and the per-layer linears are NOT the
residual** — this *refutes* the "lm_head is a big suspect / add M=2 argmax lm_head" hypothesis.

**Reconciliation of the ~46 ms verify marginal (measured w2−w1 @ ctx 4096):**
| component | M2 marginal |
|---|---|
| MoE `gather_qmm` (row-wise, 1.4×) | ~+13 ms |
| non-MoE quantized linears (amortized) | ~0 ms |
| `lm_head` | ~+1 ms |
| **diffuse residual (attention/indexer/norms/RoPE/cache + small-L ops × 78 layers, MLX dispatch)** | **~+32 ms** |

**⇒ The dominant blocker is diffuse MLX op-execution overhead for the L=2 graph, NOT any quantized matmul.**
It is not FLOPs/bandwidth (each op is tiny); it is MLX running ~the same op count but on the L=2 shape with
the extra small-L ops, eagerly, ×78 layers. This is **`mx.compile` / kernel-fusion territory**, possibly
larger than the 5-15 ms first estimated. Secondary lever: an M-coalescing routed-expert kernel (~13 ms).
`lm_head`/linear M=2 kernels are NOT worth it. To cross ≥1.3× we must shave ~28 ms ⇒ need most of the
diffuse 32 ms (compile/fusion of the verify step) + optionally the MoE 13 ms.

**Compile feasibility gate (ran it): `mx.compile(full L=2 forward)` = 1.06× (97.8→92.7 ms).** Only ~6%
fusible ⇒ the diffuse ~32 ms is in the **opaque big kernels** (`gather_qmm` / SDPA / `quantized_matmul` /
native Metal), not fusible glue — `mx.compile` can't fuse single kernels. The static-cache+compile rewrite
would buy ~6%, not the ~30% needed. **Compile is dead as a lever.**

**Remaining open (for a future GPT-5 Pro round, if revisited):** the only untried lever is a custom
**M-coalescing routed-expert kernel** (~13 ms) and/or a **fully-fused GLM-DSA decoder-layer Metal kernel**
(one kernel/layer, to eliminate the inter-kernel dispatch/dataflow that is the diffuse ~32 ms). The latter
is the only thing that could plausibly cross ≥1.3×, but it's a months-class rewrite of the model's forward
as fused Metal. The MoE kernel alone caps at ~1.1-1.18× @ long ctx (>1.0× but <1.3× bar). Conclusion:
**single-user batch-1 M3 Ultra/MLX is the wrong regime** — the literature's 1.8-2× is GPU (weight-bound) /
serving-batch 4-16; here the L=2 forward is ~2× the L=1 forward as an emergent MLX-execution cost that no
cheap lever (compile, component kernels) removes.

## 10. Summary for the helper

- Acceptance is fine (~66% avg). The head is cheap (~2.4 ms). **The sole blocker is `t_verify` ≈ 1.9× a
  decode.**
- It is **provably not** expert-activation/MoE-weight (identical==diverse), attention (fixed), or indexer
  (fixed). It is **diffuse MLX per-token execution overhead** (~41 ms/token marginal vs <0.1 ms FLOPs / ~0
  extra weight).
- The literature's 1.3-2× single-user MoE-spec wins assume a weight-bandwidth-bound decode; M3 Ultra/MLX
  batch-1 isn't. **The win requires making the L=2 verify forward as per-token-efficient as the L==1 decode**
  (or otherwise structuring the verify to cost ≈1 decode). The 65× standalone-vs-in-loop gap (§7) is the
  most promising lever to understand.
- If `t_verify(2)` drops to ~1.2× a decode, this is a **~1.4× decode win** at the measured acceptance.

---

## 11. Cross-campaign addenda (2026-07-05, MiniMax-M3 overlap/EAGLE campaigns)

**The §7 "65× standalone-vs-server mystery" is (very likely) SOLVED:** the M3 EAGLE Gate-1 harness hit the
identical symptom (~2.3 s/token standalone vs ~40 ms in-server) and the cause was the standalone process
missing `mx.set_wired_limit(...)` — unwired weights → GPU page-fault storm on every weight read. The omlx
server raises the wired limit at startup (process_memory_enforcer); an isolated `model(x, cache)` on a fresh
main thread does not. Fix for any standalone repro: `mx.set_wired_limit(506*1024**3)` before the forward.
This removes the mystery but does NOT change §10b's verdict (the in-loop 48ms path is the already-optimized
one; there is no hidden 65× to harvest).

**Independent confirmation of §10b from MiniMax-M3 (2026-07-04/05):** M3's EAGLE-3 wrapper measured verify
L=2 GPU-wait = 1.83× base @5k / 2.12× @16k (86.9%/82.6% live acceptance — drafter not the problem), i.e.
the same ~2× L=2 emergent cost on a different MoE (fewer, larger experts). M3's leg-C attributed it to
expert divergence; GLM's §10b identical-vs-diverse control shows the diffuse per-op overhead dominates
instead (M3's identical-token control was never run — its expert/diffuse split is unmeasured). Both models,
both wrappers, both quant schemes → batch-1 L=2 verify ≈ 2× on MLX is structural. Also proven on M3 within
the same campaigns: raising MLX MAX_ACTIVE_TASKS (cap 10→64) = flat; mx.compile of the full L=1 decode =
flat at batch-1 despite 86× less host eval work (mechanism: host already overlapped; wall = GPU +
per-op dispatch). The only levers left standing for BOTH models: fused decoder-layer/mega-kernel work
(months-class) and batch≥2 (expert reads + per-op costs amortize).

**Base-decode residual @ golden env (buffer env OPS=4000/MB=4000, shipped 2026-07-04, GLM 20.8→23.1):**
GLM L=1 decode = 43.3 ms/token: 53.6% GPU-wait, 17.5% Metal encode, ~12.5% python — ~10-12 ms/token
non-GPU on the critical path, same diffuse family as §7's W. No cheap lever applies (see above).
