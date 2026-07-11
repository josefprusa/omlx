> Verified 2026-07-05 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Experiments registry — pre-Ultra (EXP-001…049)

Chronological ledger of every performance experiment BEFORE the Nemotron-3-Ultra-550B day
(GLM-5.2, Kimi, MiniMax-M3, EAGLE/MTP speculative work, Nemotron-Super bring-up, infrastructure).
The Ultra-550B day (EXP-050+) and the dead-levers digest live in sibling files
(`experiments/ultra-day.md`, `experiments/index.md`, `dead-levers.md`). Verdicts: WIN (shipped or
kept), DEAD (measured negative, do not re-litigate without new data), PARKED (correct but shelved),
OPEN-BUG (unresolved production risk).

**Source paths** (cite key): repo-relative files under `$OMLX/` —
`tasks/todo.md` (master chronology), `GLM52_MTP_FORAY.md` (repo root), `tasks/eagle3_build.md`,
`tasks/eagle_temp1.md`, `tasks/overlap_levers.md`, `tasks/p0_routing.md`, `tasks/lessons.md`,
`tasks/compile_spikes/PHASE_A_SUMMARY.md`, `tasks/oqnvfp4_nemotron.md`. `mem/<name>` =
`$CLAUDE_PROJECT_MEMORY/<name>`. Probe scripts:
`~/.claude/jobs/62f9cfe9/tmp/` and the session `scratchpad/`.

**Terms:** MLA = Multi-head Latent Attention; DSA = DeepSeek Sparse Attention (indexer selects top-2048
keys); MoE = Mixture of Experts; MTP = Multi-Token Prediction (self-speculative head); EAGLE-3 =
draft-head speculative decoding; MSA = MiniMax Sparse Attention; SDPA = Scaled Dot-Product Attention;
GQA = Grouped-Query Attention; qmv/qmm = quantized matrix-vector/matrix-matrix; gather_qmm = grouped
(per-expert) qmm; gs = group size; TG = threadgroup; SLC = System-Level Cache; KV = key/value cache;
NIAH = Needle-In-A-Haystack; TTFT = Time-To-First-Token; ABI = Application Binary Interface;
NVFP4 = NVIDIA FP4; MXFP8 = microscaling FP8; oQ4/oQNVFP4 = omlx affine-4bit / NVFP4 quant recipes;
fs5 = MiniMax-M3 variant (fused-shared expert + 5-bit attention layers 17-44); α = draft accept rate.
"Golden env" = launch with `MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000` (EXP-044).
Batch-1 decode on this box is GPU-bandwidth-bound (proven repeatedly); host work rarely on the critical path.

### GLM-5.2 native kernels & KV cache (2026-06-24…29) — model avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw
## EXP-001: GLM native Metal kernels revival (nanobind ABI fix)
- Hypothesis: the #1984 GLM MoE-DSA native Metal kernels are disabled only by a nanobind ABI mismatch, not a real defect.
- Method: pin `nanobind==2.12.0` (== NB_INTERNALS_VERSION v19, matching mlx-metal 0.31.2; ext had been built vs 2.13.0=v20), rebuild ext with `OMLX_WITH_CUSTOM_KERNEL=1`; pre-load gate greps `strings _ext*.so` for `v19_system_libcpp_abi1`. Kill switch `OMLX_GLM_DISABLE_NATIVE=1`.
- Numbers: identical 19,854-tok prompt, seed0/temp0, 64 out — dense 184.7s vs native 117.7s = **1.57×**. Long ctx: native completes ~53k (~323s) where dense OOMs ~44k (>417GB prefill cap). 5 of 6 native ops bit-agnostic; DSA sparse-attention engages >11,264 tokens. 11/11 patch tests pass (now 12/12 — a test was added since; see models/glm52.md §5).
- Verdict: WIN — foundational; native kernels both speed up AND enable long context.
- Revival: n/a (shipped, committed). Re-check ABI tag after any MLX/nanobind bump.
- Source: `mem/omlx-glm52-native-kernels.md`.
## EXP-002: GLM 4-bit MLA V-up kernel (bits-threaded, window widened)
- Hypothesis: the 8-bit-only, 32-64k-windowed `glm_dsa_q8_vup_flat` kernel could fire for this 4-bit-MLA model if `bits` were threaded through and the window widened.
- Method: thread `bits` through `csrc/fused_moe.{cpp,metal}`, instantiate the 4-bit Metal kernel, bits-aware Python guard, widen window to ≥11,264 (no ceiling).
- Numbers: microbench native 4-bit V-up = 1.06-1.14× faster than `mx.quantized_matmul` across prefill chunks (M=1024..6144). But V-up is a small compute slice → **sub-1% end-to-end** (lost in full-model noise).
- Verdict: WIN (marginal) — real but tiny; committed as capability.
- Revival: only meaningful if V-up ever becomes a larger share of the graph.
- Source: `mem/omlx-glm52-native-kernels.md`.
## EXP-003: GLM native 3-bit MoE gather_qmm kernel (Stage-0 gate)
- Hypothesis: a specialized 3-bit sorted gather_qmm could beat stock `mx.gather_qmm` for the routed experts (>10% on both gate_up AND down = GO).
- Method: microbench on GLM shapes (8192-tok chunk, ~256 rows/expert); note `affine_gather_qmm_rhs` is verbatim MLX's own kernel.
- Numbers: `mx.gather_qmm` 3-bit gs64 = gate_up 19.9 TF/s, down 20.0 TF/s vs dense-fp16 ceiling ~25 TF/s (~90% of M3 peak) → gather_qmm is ~80% of fp16 peak, 1.25× off dense; prefill compute-bound (arithmetic intensity ~1365 FLOP/B ≫ ridge ~34). Building the "native" kernel = parity by construction.
- Verdict: DEAD — no >10% lever; STOP (no code written; avoided the int8-style sink).
- Revival: only if a genuinely different tiling out-tunes MLX's shape heuristic (low odds).
- Source: `mem/omlx-glm52-native-kernels.md` (Stage-0 result).
## EXP-004: GLM int8 MLA-KV cache + native q8 sparse-MLA kernel
- Hypothesis: quantizing the MLA latent cache to int8 halves KV bytes and (via a native q8 kernel) speeds bandwidth-bound sparse-MLA prefill.
- Method: `Int8MLALatentCache` + `BatchInt8MLALatentCache` (dequant-on-read), native `sparse_mla_attention_q8` kernel (commit c0610a2); token-identical validation; live GLM prefills.
- Numbers: token-IDENTICAL (max_abs=0.0). MLA-KV **42% smaller** (500k 44.9→26.2GB, 1M 89.9→52.4GB). BUT ~2× slower prefill (16k: 185s int8 vs 91s fp16) from a dequant-on-read transient that death-spirals the prefill throttle through a 5-clamp stack. Kernel-level int8 ≈ dense parity (16k 31 vs 29.6ms) — Apple has no int8 MMA.
- Verdict: PARKED — built + validated, then ABANDONED 2026-06-25 (reverted speed-chasing to clean c0610a2; q8 kernel + cache kept as committed capability, setting OFF). fp16 already fits 1M ctx on 512GB.
- Revival: only for >1M ctx or a smaller machine where fp16 KV doesn't fit; then also needs block-slicing handler + throttle intercept + GLM adaptive-step fix.
- Source: `mem/omlx-glm52-int8-mla-kv.md`.

### GLM-5.2 MTP self-speculative decoding (2026-06-29, re-run 07-03) — the "verify wall"
## EXP-005: GLM MTP head graft + teacher-forced acceptance gate
- Hypothesis: a grafted MTP head (our 3.5bpw checkpoint dropped MTP weights) can predict the next trunk-greedy token well enough (α>0.5) to be worth a verify path.
- Method: graft `inferencerlabs/GLM-5.2-MTP-MLX-Q4` (eh_proj + 1 GLM-DSA decoder layer) under `mtp.0.*`; teacher-forced gate α = P(mtp_pred[t]==trunk_greedy[t+1]).
- Numbers: PASS, overall α=**0.603** (code 0.674, prose 0.750, 12-tok factual list 0.09 hard). eh_proj concat order [embed,hidden] confirmed; RMSNorm no-shift confirmed.
- Verdict: WIN — graft is correct; acceptance is fine for typical prompts.
- Revival: n/a (kept inert; the blocker is verify cost, not acceptance — see EXP-006/009).
- Source: `mem/omlx-glm52-mtp.md` (GATE RESULT); `GLM52_MTP_FORAY.md §3`.
## EXP-006: GLM MTP live self-speculative decode
- Hypothesis: 1-draft + 2-token verify per cycle nets ≥1.3× decode at good acceptance.
- Method: live server MTP stat lines, clean A/B same prompt, 256 tok greedy, identical output.
- Numbers: mtp-off **21.0** vs mtp-on **15.5 tok/s = 0.74×**. Live α by prompt 80.9/77.6/65.6% (merge_sort 41% outlier). backbone = 97% of decode; verify(2-token) ≈ 1.8× a 1-token decode.
- Verdict: DEAD — slower at every context; the 2-token verify defeats the gain.
- Revival: batch≥2 (expert-weight reads amortize across the batch) — the recurring escape hatch.
- Source: `mem/omlx-glm52-mtp.md` (LIVE RESULT); `GLM52_MTP_FORAY.md §8`.
## EXP-007: GLM MTP small-L absorbed-attention + small-s indexer verify fixes
- Hypothesis: the L>1 verify explodes because the general path projects the WHOLE kv_latent (O(context)); an absorbed small-L branch that projects only the L queries collapses it.
- Method: `OMLX_MTP_VERIFY_FAST`-gated branches — small-L (1<L≤8) absorbed MLA in `glm_moe_dsa_model.py` (per-query top-2048 gather, project only L queries), small-s (1<s≤8) non-padded indexer in `deepseek_v32.py` (skips the 64-row pad).
- Numbers: argmax_agree=**1.000** (greedy-lossless). Verify **414→106 ms @16k**, 441→131 ms @64k; indexer fix 131→103 ms/cycle @64k. Context-independent attention restored.
- Verdict: PARKED — real, lossless, but net still <baseline (Wall B remains). Kept flag-gated OFF as groundwork.
- Revival: batch≥2 or a denser target; reuse verbatim if an expert-batching verify kernel ever lands.
- Source: `GLM52_MTP_FORAY.md §5.1/§5.2`; `mem/omlx-glm52-mtp.md` (M2 STEP-1).
## EXP-008: GLM MTP MoE sort-threshold lowering (64→16 routes)
- Hypothesis: the SwitchGLU `do_sort` threshold of 64 leaves a 16-route verify on the slow unsorted path; lowering it helps.
- Method: `do_sort = indices.size>=64 or (fast_verify and indices.size>=16)` (`switch_layers.py`).
- Numbers: **no measurable change** — consistent with the identical==diverse finding (EXP-009); the MoE is not the variable cost.
- Verdict: DEAD — no effect.
- Revival: none.
- Source: `GLM52_MTP_FORAY.md §5.3`.
## EXP-009: GLM MTP verify-cost isolation (identical-vs-diverse + probes A/B)
- Hypothesis (falsified): the verify cost is the MoE expert-doubling (2 verify tokens hit ~2× distinct experts).
- Method: feed IDENTICAL vs DIVERSE verify tokens (same vs ~2× experts) in the real executor-thread regime; then GPT-5-Pro synthetic probes A (gather_qmm 3-bit shapes) and B (dense quantized_matmul M=1 vs M=2).
- Numbers: identical w2=131ms == diverse 131ms @512 (96 vs 98 @4096) → **NOT expert-activation**. Decompose t(L)=F+W·L → F≈14ms, W≈**41 ms/token**. Probe A: M2same 1.40×≈M2div 1.42× (MoE marginal only ~+13ms ×75 layers); Probe B: dense linears ~1.00× (weight-dequant amortized across 2 rows), lm_head 1.76× but +1ms only. Residual = **~32ms diffuse MLX per-op dispatch** ×78 layers.
- Verdict: DEAD — hypothesis falsified and the component-kernel lever is dead: the wall is diffuse eager-execution overhead of the L=2 graph, not any single matmul.
- Revival: only a fully-fused decoder-layer Metal kernel (months-class) touches the diffuse cost.
- Source: `GLM52_MTP_FORAY.md §4/§7/§10b`.
## EXP-010: GLM MTP mx.compile(L=2 verify) feasibility gate
- Hypothesis: compiling the whole L=2 forward fuses the diffuse glue away.
- Method: `mx.compile(full L=2 forward)`, warm medians.
- Numbers: **1.06×** (97.8→92.7ms) — only ~6% fusible; the diffuse cost lives inside opaque big kernels (gather_qmm/SDPA/quantized_matmul/native), which compile can't fuse.
- Verdict: DEAD — compile is not the lever; a static-cache+compile rewrite buys ~6%, not the ~30% needed.
- Revival: none for batch-1 (independently re-confirmed for L=1 base decode in EXP-046).
- Source: `GLM52_MTP_FORAY.md §10b`; `mem/omlx-glm52-mtp.md` (GPT-5 Pro round).
## EXP-011: GLM MTP chained-K (K=2 / K=5) with cache-correctness fixes
- Hypothesis: chaining K self-drafts amortizes the fixed L>1 penalty enough to break even.
- Method: `mlx_lm_mtp` patches — fixed gap-cache bug (only every other position got an MTP entry) and trim bug (hardcoded trim-1 corrupted K>1 output); per-position acceptance logging.
- Numbers: dense-cache fix works → α1=**73%** live, but α2|1 collapses to **15%** (self-chained hidden drifts on the quantized stack). verify(3)=124ms. K=2: 14.3 tok/s (0.66×); K=5: 9.5 (0.44×) vs 21.7 baseline. Break-even at K=2 needs 2.84 tok/cycle → α~0.92 everywhere.
- Verdict: DEAD — verify-cost wall confirmed end-to-end with working acceptance; `mtp_enabled=false` restored.
- Revival: batch≥2; or a fix for the α2 chain-hidden collapse (Q4-head/3-bit-trunk drift).
- Source: `tasks/todo.md` ("MTP chained K built + tested live").

### GLM-5.2 decode/prefill optimization (2026-07-03) — the "core-MLX" campaign
## EXP-012: GLM decode-kernels shipped (fused indexer-scores + flash-decode + embed_q swap)
- Hypothesis: a handful of decode-only fusions recover a few percent losslessly.
- Method: `decode_kernels.py` — fused s==1 indexer-scores kernel, split-K flash-decode sparse-MLA gated `K≥98304` (`OMLX_GLM_FLASH_DECODE_MIN_K`), `mh_qmm_m1` (embed_q via gather_qmm, bit-identical), MoE weighted-sum as one gemv. Kill switch `OMLX_GLM_DISABLE_DECODE_OPT=1`.
- Numbers: live A/B greedy-identical — 20.8→**21.7** short (+4.3%), 19.5→19.9 @14.5k (+2.1%), 18.2→**18.5** @57.7k (+1.6%). Synthetic layer max_abs 2e-3, topk overlap 99.95%.
- Verdict: WIN — shipped +2-4% decode.
- Revival: n/a. (Caveat: much of the win came from non-gated parts; the fused indexer-scores kernel was later found dtype-gated dead until EXP-020.)
- Source: `tasks/todo.md` ("Shipped"); `mem/omlx-glm52-decode-opts.md`.
## EXP-013: GLM dual-stream shared-expert overlap
- Hypothesis: running the shared expert on a 2nd `mx.stream` overlaps it with routed experts.
- Method: shared expert dispatched on a second stream; probe `scratchpad/dualstream_probe.py`.
- Numbers: 53.8→**69.9 ms/token (23% SLOWER)** — ~200µs/token cross-stream fence ×75 layers — AND numerically wrong (max_abs_diff=**18**; MLX buffer reuse is stream-unsafe here).
- Verdict: DEAD on this MLX version (both slower and incorrect).
- Revival: only if a future MLX makes cross-stream fences cheap and buffer reuse stream-safe.
- Source: `tasks/todo.md` ("two remaining levers").
## EXP-014: GLM read-once flash-decode sparse-MLA kernel (K-A, five architectures)
- Hypothesis: a single fused gather+pe+SDPA decode kernel beats the MLX gather+SDPA chain.
- Method: five kernel architectures, allclose vs MLX chain then live; probes `flashdecode_v4/v5.py`.
- Numbers: v1 114µs, v2 108, v3 78 (shipped for K≥98k), v4 180-430 (read-once all-64-heads-per-TG), v5 92-94 (faithful clone of MLX sdpa_vector + in-kernel gather). All lose to MLX gather+SDPA 67-70µs at ≤32k. Root cause: materialize-once-then-sequential-SDPA (compact 2.4MB buffer stays SLC-hot, re-read at ~3TB/s) beats any in-kernel random-gather variant on M3 Ultra.
- Verdict: DEAD — MLX's chain is near-optimal below ~64k; only v3 survives at ≥98k.
- Revival: none on M3 Ultra memory hierarchy; would need different HW.
- Source: `tasks/todo.md` ("K-A", "remaining levers").
## EXP-015: GLM K-B fused rms_norm+qmv prologue kernel
- Hypothesis: fusing the RMSNorm into the following qmv captures ~40µs of glue in the projection chain ("70µs vs 31 floor").
- Method: `mx.fast.metal_kernel` rms prologue staging normed x into TG mem + qmv reading from TG mem; probe `scratchpad/rms_qmv_probe.py`.
- Numbers: correct (rel 1e-3) but parity-to-loss — q_a 31.0 vs 30.6, kv_a 29.5 vs 21.4, q_b 30.1 vs 27.1 µs. Premise was floor-math optimism: RMSNorm already pipelines (chain minus qmv-only = 2-4µs); the 3 standalone qmvs sum to 69µs = the "70".
- Verdict: DEAD — no capturable glue.
- Revival: none.
- Source: `tasks/todo.md` ("K-B CLOSED NEGATIVE").
## EXP-016: GLM K-C q8_vup at decode L=1
- Hypothesis: the existing `glm_dsa_q8_vup_flat` kernel, if its Python key_length gate is relaxed at L=1, adds ~1% decode.
- Method: relax gate + shape test at L=1; probe `scratchpad/q8vup_probe.py`.
- Numbers: works (max_abs 4.4e-3) but **34.3µs vs MLX chain 23.1µs** — a prefill-shaped kernel, underutilized at M=1.
- Verdict: DEAD — slower at decode; no integration.
- Revival: none at M=1.
- Source: `tasks/todo.md` ("K-C CLOSED NEGATIVE").
## EXP-017: GLM K-D int8 MLA-KV decode (Phase 1a gather / 1b quantized-SDPA)
- Hypothesis: an int8 latent cache halves decode attention reads (gather + SDPA) for a decode speedup.
- Method: Phase-1a int8-gathered decode attn; Phase-1b `mlx quantized_scaled_dot_product_attention` on gathered tuples; probes `int8kv_probe.py`, `int8kv_1b.py`.
- Numbers: 1a = 76.7µs vs fp16 66.9 @32k (LOSES — dequant + 3-array gather beats the read-halving while SLC absorbs the fp16 read), 79.7 vs 97.0 @131k but identical to shipped flash-v3 (not additive). 1b = 130µs vs fused fp16 SDPA 66µs @32k (2× slower, unfused qmm+softmax+qmm).
- Verdict: DEAD — no ≥2% decode boost at any phase; surviving value is memory-only (KV halves → ~2× context, quality tax 1.4e-3/layer).
- Revival: as a capacity feature only (needs full convert-post-prefill/dequant-on-extend/SSD-save integration).
- Source: `tasks/todo.md` ("K-D VERDICT"); `mem/omlx-glm52-decode-opts.md`.
## EXP-018: GLM block-union prefill (lead #5)
- Hypothesis: unioning per-query top-2048 blocks into dense tiles could turn sparse-MLA prefill into an efficient dense matmul.
- Method: env-gated topk dump (`OMLX_GLM_TOPK_DUMP`, layers 6/30/62) on a real 20k code prompt; `union_analyze.py` over `topkdump/`.
- Numbers: union ratios L6 1.41-2.19×, L30 1.70-3.31×, L62 2.00-4.55×; overlap decays with depth. At dense-viable tile sizes (Bq≥32) inflation 1.8-4.5× ≥ any efficiency gain (~1.6-1.8×).
- Verdict: DEAD — the last unexplored prefill idea, closed with real data; prefill is at ceiling on all fronts.
- Revival: none for this model/quant on this HW.
- Source: `tasks/todo.md` ("Lead sweep … block-union").
## EXP-019: GLM omlx serving-layer tax
- Hypothesis: the omlx scheduler/sampler/stream path adds overhead vs bare mlx_lm.
- Method: bare `mlx_lm.stream_generate` with identical patched model code vs omlx server; `standalone_bench.py`.
- Numbers: bare mlx_lm **21.38** vs omlx server **21.7 tok/s** (server slightly faster — better pipelining). Model-card 21.29 = bare mlx_lm confirmed.
- Verdict: WIN (null result) — serving tax is ZERO; no optimization available in the omlx path.
- Revival: none. Ops note: `sudo sysctl iogpu.wired_limit_mb=518144` raises the Metal cap 464→506GB (+42GB KV headroom).
- Source: `tasks/todo.md` ("omlx serving-layer tax MEASURED ZERO").
## EXP-020: GLM fp16-gate kernel "disease" fix (bf16 template-ize + live census)
- Hypothesis: the fused decode-indexer kernel never engages live because it is gated on fp16 while the live model runs bf16 (torch_dtype/safetensors BF16).
- Method: template-ize the kernel (`T ∈ {half, bfloat}`, output/weights in model dtype), widen the gate, add one-shot `[GLM-DKO] ENGAGED/BAIL` census logs.
- Numbers: standalone bf16 top2048_overlap **2048/2048** vs exact MLX chain (K=2049 & 58368). Confirmed `fused_decode_indexer_scores` NEVER engaged live before; `flash_decode_sparse_mla` had NO caller (dead code). The shipped +2-4% (EXP-012) came from the non-gated parts.
- Verdict: WIN — fixes a silent fp16/bf16 fallback; but note the fused indexer path is still gated on `mask is None` (deepseek_v32.py:271), so restored/batch decode at long ctx can still skip it.
- Revival: extend the fused path to bool-mask decode (mirror M3's original_mask take_along_axis validity trick).
- Source: `tasks/todo.md` ("GLM SAME-DISEASE FIX"); `mem/omlx-live-path-verification.md`.

### Kimi-K2.7-Code VLM (2026-07-03) — avlp12 Kimi 1T MoE, DeepSeek-V3 arch (kimi_k25), no DSA/MTP
## EXP-021: Kimi dsv3_decode_opts patch
- Hypothesis: the GLM/Kimi decode playbook (M=1 gather_qmm swap + MoE weighted-sum gemv) ports to the DeepSeek-V3 family.
- Method: new `omlx/patches/dsv3_decode_opts.py` (QuantizedMultiLinear M=1→gather_qmm class-wide + dv3 MoE wsum gemv), wired for deepseek_v3/kimi_k2/kimi_k25. Kill `OMLX_DISABLE_DSV3_DECODE_OPT=1`.
- Numbers: live A/B same greedy prompt — OFF 24.6 → ON **25.4 tok/s (+3.3%)** (card ~23; we serve ~10% above card). Output coherent; ≤1-ulp embed_q swap flips rare near-ties.
- Verdict: WIN — shipped +3.3%.
- Revival: n/a.
- Source: `tasks/todo.md` (Kimi RESULTS); `mem/omlx-glm52-decode-opts.md`.
## EXP-022: Kimi memory_monitor MLA-KV estimate bug
- Hypothesis: Kimi prompts >~4k tokens are wrongly rejected by the prefill guard.
- Method: fix `estimate_mla_kv_bytes_per_token` — descend into `text_config` for VLM configs, count plain per-layer KVCache as the main MLA cache (was falling back to the expanded-MHA formula → ~23× KV overestimate).
- Numbers: now 70,272 B/token (61×576×2B ✓), 128k = 9.2GB (was ~23× that). Long context on Kimi was impossible before this.
- Verdict: WIN — bug fix; unblocks Kimi long prompts entirely.
- Revival: n/a.
- Source: `tasks/todo.md` (Kimi BUG FIX).
## EXP-023: Kimi long-context decode
- Hypothesis: Kimi holds decode speed at depth (it has no DSA → reads the full latent cache).
- Method: live long-ctx decode bench.
- Numbers: 16k = **21.2 tok/s** (vs 25.4 short; on the no-DSA bandwidth curve). 64k/128k legs CANCELLED — fresh prefill in the throttled regime crawls (~23 tok/s @24k depth vs GLM's ~180).
- Verdict: PARKED — decode itself fine at 16k; blocked by a prefill-throttle phantom-transient over-reservation (estimator reserves for materialized attention scores MLX never materializes).
- Revival: fix `estimate_chunk_transient_bytes` for fused-SDPA models (tighten the phantom reservation), then re-bench a cache-busted 64k prefill.
- Source: `tasks/todo.md` (Kimi long-ctx bench).

### MiniMax-M3 sparse-attention (2026-07-03…04) — unigilby MiniMax-M3-oQ4 428B MoE VLM (minimax_m3_vl)
## EXP-024: M3 bring-up + weighted-sum gemv edit
- Hypothesis: the vendored model is modern; only the MoE weighted-sum needs the gemv edit.
- Method: vendored `language.py:1444` wsum mul+sum → one gemv; audit (router already `@mx.compile`, gate_up already fused). Live decode/prefill bench.
- Numbers: **26.5 tok/s short** decode (+22% vs card's 21.7), prefill 182 tok/s JIT-warm (card 214; first request pays ~50s Metal JIT). Native `minimax_msa_topk` healthy.
- Verdict: WIN — day-1 baseline well above card.
- Revival: n/a (superseded by the full M3 campaign below).
- Source: `tasks/todo.md` ("M3 RUNNING").
## EXP-025: M3 context-sag root cause = fp16-gated fused-index kernel never engaged live
- Hypothesis (first-principles pass): the anomalous decode sag (26.5→~19.6 by 3.6k, physics says −6%) is a real per-token O(K) leak, not thermal/contamination.
- Method: multi-shot census `OMLX_M3_DEBUG_PATH=N` with branch counters; also caught two red herrings (bench contamination — `m3_16k.py` requested GLM; a layer-0 debug artifact faking a "plain KVCache" theory).
- Numbers: census showed `fused_none=57 + scores_fallback=57` EVERY layer EVERY step @16k — the fused index kernel NEVER engaged. Cause: dtype gate required fp16; live model runs bf16. Fallback = full-cache fp32 astype ×2 + matmul + pad + reshape + blockmax on 57 layers/token = O(K) traffic (~1.9GB/tok @16k, ~7.7GB @64k) = the anomalous slope.
- Verdict: WIN (diagnosis) — root-caused the sag to a silent gated-kernel fallback (the house "fp16-gate vs bf16-live" law); unlocked the +7% fix in EXP-026.
- Revival: n/a — fixed in EXP-026.
- Source: `tasks/todo.md` ("M3 SLOPE ROOT CAUSE FOUND"); `mem/omlx-live-path-verification.md`.
## EXP-026: M3 fused-index kernel dtype-generic fix + census instrument
- Hypothesis: a dtype-generic (fp16+bf16) fused index kernel + a permanent engagement census recovers the O(K) slope.
- Method: dtype-generic scalar loads (MLX auto-instantiates per dtype), widen gate, bit-exact vs fallback; permanent census `OMLX_M3_DEBUG_PATH`. Kill `OMLX_M3_DISABLE_FUSED_INDEX`.
- Numbers: bit-exact (max|diff|=0.0, top16 identical, K=3712/16389/65536, both dtypes). LIVE @16.8k: **19.55→20.94 tok/s (+7%)**, `fused_hit=57/57`.
- Verdict: WIN — the big M3 slope fix; also the origin of the permanent census discipline.
- Revival: n/a.
- Source: `tasks/todo.md` ("M3 SLOPE ROOT CAUSE FOUND & FIXED").
## EXP-027: M3 fused topk+sort+positions kernel (m3_fused_index_topk)
- Hypothesis: fusing NaN-clean + init/local forcing + argpartition + ones + sort (+ per-query positions) into one dispatch trims the fixed sparse tax.
- Method: one metal kernel; taken-bitmap guard for degenerate <16-valid; positions as a 2nd output (Lever B). Verified 30/30 vs exact mx chain. Kill `OMLX_M3_DISABLE_FUSED_TOPK` / `_FUSED_POSITIONS`.
- Numbers: census `fused_topk=57/57`; live 21.52 @16k (+0.42 = −0.9 ms/tok), 21.88 @9.5k. Below the +2-3ms estimate (several replaced ops were metadata-only; ~5 real dispatches saved). Fixed sparse tax now ~5ms.
- Verdict: WIN — shipped.
- Revival: n/a.
- Source: `tasks/todo.md` ("FUSED TOPK KERNEL SHIPPED"); `tasks/eagle3_build.md` (Lever B).
## EXP-028: M3 packed 5-way projections (one quantized_matmul)
- Hypothesis: q/k/v/index_q/index_k can be one `quantized_matmul` + reshape + head-slice views (no copies) since all five outputs are 128-dim multiples.
- Method: lazy per-layer packed build with spec checks (bits/group/mode equal, no bias, exact QuantizedLinear); tiers full/qkv/none. Kill `OMLX_M3_DISABLE_PACKED_PROJ`.
- Numbers: bit-exact at 8-bit AND 5-bit (max|diff|=0); census pack_full=56/qkv=3/none=1 (matches config). LIVE **27.53 short (+2%)** / 22.50 @9.5k / 22.02 @16k. Cost +3-4GB resident (originals kept for fallback).
- Verdict: WIN — shipped; cumulative @16k 19.6→22.0 (+12%).
- Revival: n/a.
- Source: `tasks/todo.md` ("PACKED PROJECTIONS SHIPPED").
## EXP-029: M3 prefill trivial-mask/position neutralization
- Hypothesis: unpadded singleton chunked prefill carries a trivial causal-array mask + positions that block the native MSA-prefill path.
- Method: neutralize trivial positions + causal-array mask under an `isinstance(offset,int)` guard.
- Numbers: 16k fresh prefill **273→342 tok/s (+25%)** (warm-to-warm TTFT 60.1→48.0s).
- Verdict: WIN — shipped.
- Revival: n/a.
- Source: `tasks/todo.md` ("M3 PUSHED FURTHER … PREFILL +25%").
## EXP-030: M3 SPARSE_MIN_K default 2048→4096 (kill the masked-dense band)
- Hypothesis: a masked-dense band between 2k-4k is slower than either dense (<4k) or compact (≥4k); moving the floor to 4096 removes it.
- Method: `OMLX_M3_SPARSE_MIN_K` default 2048→4096 (dense <4k, compact ≥4k).
- Numbers: 3.8k decode **21.6→24.3 tok/s (+12.5%)**; 16k compact unchanged 19.6.
- Verdict: WIN — shipped as default.
- Revival: n/a.
- Source: `tasks/todo.md` ("DECODE +12.5% at mid-context").
## EXP-031: M3 dense-vs-sparse crossover + MLX no-GQA-sharing mechanism
- Hypothesis: is sparse actually faster than dense at depth, and why is dense so slow?
- Method: dense-forced legs (`SPARSE_MIN_K=1e9`) vs sparse at 16k/69k; Codex source read of MLX sdpa_vector.
- Numbers: dense 19.88 @16k (sparse 20.94, +5%), **13.10 @69k (sparse 18.5, +41%)**. Dense grows ~0.72ms/1k (3.5× naive KV physics) because MLX `sdpa_vector` has NO GQA K/V head sharing — every q head streams its kv head independently (16× unique bytes at GQA-16; Apple cache absorbs to ~3.5×; 2-pass engages K≥1024, no block knob in 0.31.2).
- Verdict: WIN (decision validated) — `SPARSE_MIN_K=4096` vindicated as the shipped default; MSA sparse was starved by the fp16 gate, not slow by design.
- Revival: upstream MLX PR #3455 adds `MLX_SDPA_BLOCKS`; a native GQA-sharing sdpa_vector would help dense.
- Source: `tasks/todo.md` ("M3 DENSE-vs-SPARSE CROSSOVER", "MLX INTERNALS").
## EXP-032: M3 flash sparse-SDPA kernel (v1 + Lever-C split-K v2)
- Hypothesis: an online-softmax kernel over the 2048 selected keys (no compact K/V materialization) beats the gather+SDPA chain.
- Method: v1 single-pass online-softmax; Lever-C v2 2-pass split-K (`fused_flash_v2.py`, 512 TGs pass1 + 64-TG merge). Parity-verified; kill `OMLX_M3_ENABLE_FLASH_SPARSE(_V2)` (opt-in OFF).
- Numbers: v1 0.33-0.38ms vs mx chain 0.29ms @16k (64 TGs = 1/q-head under-occupy the 80-core M3 Ultra; per-key simd_sum latency-bound). v2 285.7µs vs mx gather+SDPA 264.8µs @16k (~8% slower — now-SORTED contiguous gather is cheap, MLX native SDPA wins).
- Verdict: DEAD — both slower; left opt-in-off. Same graveyard as the GLM flash-decode (EXP-014).
- Revival: would need a fundamentally better occupancy strategy than 64 TGs; diminishing vs the ~1ms attention ceiling.
- Source: `tasks/todo.md` ("FLASH SPARSE SDPA", "LEVER C").
## EXP-033: M3 fused SwiGLU-ts kernel (fs5 Lever A)
- Hypothesis: fuse the OpenAI-style SwiGLU (a=1.702, L=7.0) with the NVFP4 tensor-scale multiply into one kernel.
- Method: `m3_swiglu_oai_ts` kernel spliced into the fs5 MoE path. Parity rel=0 on the real code path. Kill `OMLX_M3_DISABLE_FUSED_SWIGLU_TS`.
- Numbers: census `swiglu_ts_fused=57/step`, standalone fallback=0. Combined with Lever B, delta +0.12/+0.26/+0.20 tok/s (short/9.5k/16k). oQ4 correctly shows no ts engagement.
- Verdict: WIN — lossless, shipped (small, positive, kill-switched).
- Revival: n/a.
- Source: `tasks/todo.md` ("LEVER A"); `mem/omlx-glm52-decode-opts.md`.
## EXP-034: M3 NIAH ladder + long-context reasoning smoke (to 103k)
- Hypothesis: the post-kernel-fix M3 does clean retrieval AND multi-hop reasoning at long context.
- Method: multi-needle NIAH (3 needles @25/50/75% depth, temp 0) at 13/26/51/103k; 6 reasoning families (object-chain, aggregation+arith, variable-chain, latest-state, transitive, temporal) × {26k,51k,103k}; harnesses `niah_bench.py`, `quick_reason.py`.
- Numbers: NIAH **12/12 perfect** at each depth; decode at depth 22.7/21.9/19.8/18.7 (sub-linear: 18.7 @102.7k vs 17.2 linear-fit). Reasoning **18/18 semantic** across depths. Prefill ~310-331 tok/s flat to 100k+.
- Verdict: WIN (quality) — clean retrieval + reasoning to 100k+ confirmed on this box.
- Revival: 256k/512k NIAH stages unrun (fused_topk verified nb≤4096 = 512k-capable); BABILong/LongBench/RULER full suites on offer (~5.5min/128k-item, prefill-bound).
- Source: `tasks/todo.md` ("NIAH LADDER", "REASONING @103k VERIFIED").

### MiniMax-M3 oQNVFP4 conversion (2026-07-04) — NVIDIA NVFP4 → MLX byte-repack
## EXP-035: oQNVFP4 byte-repack pre-flights (M=1 kernel parity + byte-layout proof)
- Hypothesis: NVIDIA's calibrated NVFP4/MXFP8 checkpoints repack into MLX as a pure byte-copy, and the MLX qmv/gather_qmm kernels are competitive at M=1.
- Method: PF1 microbench nvfp4/mxfp8 vs affine at M=1; PF2 range-fetch a real expert triplet + index_k_proj and view the bytes directly as MLX state, compare to spec-math dequant.
- Numbers: PF1 — qmv nvfp4 261µs vs affine4 269; mxfp8 273 vs affine8 301 (**mxfp8 FASTER**); gather_qmm nvfp4 232 vs affine4 227. PF2 — NVFP4→MLX is a PURE byte-repack (max|diff|=**0.0** vs spec math, lo-nibble-first; input_scale proves activation-aware calibration, MLX ignores act scales); MXFP8 too (index_k_proj max|diff|=0.0, E8M0 scales byte-copy).
- Verdict: WIN — both formats de-risked on speed AND byte-exactness; unblocks the whole oQNVFP4 track.
- Revival: n/a — this is the reusable repack template (later generalized to Nemotron, EXP-049).
- Source: `tasks/todo.md` ("PRE-FLIGHT 1/2 PASSED", "MXFP8 BYTE-REPACK PROVEN").
## EXP-036: MiniMax-M3-oQNVFP4-fs5 production variant
- Hypothesis: an NVFP4-experts variant with a runtime tensor-scale carry matches oQ4 speed and improves quality.
- Method: `omlx/tools/oqnvfp4_convert.py` — fuse-shared-nvfp4 + attn5-layers 17-44 + per-expert `weight_scale_2` runtime carry (routing-weight fold + gate_up post-mul, fp32). Requires `MLX_MAX_OPS_PER_BUFFER=500` (Metal cmd-buffer timeout otherwise).
- Numbers: speed ~same (27.09/21.66/21.64 vs oQ4 27.53/22.50/22.02; 18.0 vs 18.7 @103k). Quality better: gsm8k **95.3 vs 92.7 (+2.7pp)**, mmlu 81.3 vs 79.3, arc 96.7 vs 94.3 (+2.3pp). 6/6 reasoning + 3/3 NIAH @103k. Variant ladder: default 23.7 → fs 25.75 → fs5 27.09 short.
- Verdict: WIN — quality up at equal speed; shipped as the production M3 variant.
- Revival: n/a. (oQNVFP4 248G + fs 246G intermediate variants deletable ~494GB once blessed.)
- Source: `tasks/todo.md` ("oQ-NVFP4 GOAL MET"); `mem/omlx-glm52-decode-opts.md`.

### MiniMax-M3 EAGLE-3 speculative decoding (2026-07-04) — drafter Inferact/MiniMax-M3-EAGLE3
## EXP-037: M3 EAGLE-3 Phase-1 multi-query flash verify kernel
- Hypothesis: a small-L (2≤L≤8) verify path in the fused kernels can hit ≤14ms/extra-token.
- Method: first tried union-of-selected-blocks + one masked SDPA (lead's design); it regressed (bool-masked union SDPA 0.26→1.40ms/layer, 5.4× L2→L4, U=64 4× redundancy). PIVOTED to a multi-query FLASH kernel (`flash_sparse_sdpa_multi`) — each query flashes its OWN 16 blocks, causal, online-softmax fp32, streams K/V, no union/gather/materialization.
- Numbers: flash ON marginal **18.77 vs OFF (MSA-prefill) 21.79 ms/extra (~14% faster)** at ALL L; argmax=1.000 vs MSA oracle (L≤5; benign near-tie wobble L≥6). Floor (attention=zeros) = 13-14.6ms → **≤14ms is MoE-bound, unreachable via attention** (bar retired).
- Verdict: WIN — correct, beats MSA-prefill at every L; deployed as the Phase-1 verify path.
- Revival: n/a for attention; the residual is the MoE per-token weight-read wall (EXP-039).
- Source: `tasks/eagle3_build.md` (PHASE 1).
## EXP-038: M3 EAGLE-3 drafter integration + acceptance
- Hypothesis: reuse mlx-vlm's shipped Eagle3 path (don't build a drafter) with an omlx adapter; acceptance ≈ vendor.
- Method: `omlx/patches/mlx_vlm_mtp/eagle3_minimax.py` — taps [1,29,56], POST-final-norm recurrence (norm_output=True), fc_norm triple, prefix-KV seeded via capture-during-prefill; 3 vlm_mtp glue changes.
- Numbers: prefix-KV seeding confirmed live (`draft_next_position==prompt_len`). Acceptance greedy: math a1/a2/a3 = **97.1/93.1/85.1%** (mean_len 3.72, BEATS vendor 92/84/76); code 86.6/65.3/53.2% (mean 3.05). Three runtime bugs only a live run caught (bare-logits return, mx.array truthiness, out-param hidden_sink).
- Verdict: WIN (acceptance) — drafter is vendor-grade; the drafter is NOT the problem.
- Revival: n/a — acceptance is fine; speed is the wall (EXP-039).
- Source: `tasks/eagle3_build.md` (accept2.log; Stage 2a).
## EXP-039: M3 EAGLE-3 live speed + K-sweep {1,2,3}
- Hypothesis: good acceptance (~3 tok/round) yields a live speedup.
- Method: live fs5 greedy A/B at short/16k; round profiler; K-sweep K=1/2/3.
- Numbers: short 1120 **26.23 vs 26.74 baseline (−2%)**; 16k **18.98 vs 21.64 (−12%)**. Profile: verify L=4 = 113-126ms = ~2.75× L1 = **85% of round**; draft only 15%. K-sweep flat within ±2% (K=1 26.65/26.59, K=2 26.63/26.40) → exchange rate ~1:1 (verify-a-token ≈ generate-a-token on memory-bound MoE).
- Verdict: DEAD — latency-neutral, never a speedup; MoE verify wall (distinct experts/token, no amortization at batch-1). Ships per-request opt-in with a temp==0 engagement guard.
- Revival: batch≥2 (MoESD: even 2-4 concurrent requests flips economics positive as-is); or MoE verify-batching; or a smaller-expert / dense target.
- Source: `tasks/eagle3_build.md` (FIRST LIVE NUMBERS, 2d PROFILE); `tasks/todo.md` (K-SWEEP, K=1 DIRECT).
## EXP-040: M3 EAGLE-3 mxfp8 drafter variant
- Hypothesis: quantizing the draft layer+fc to MXFP8 g32 (faster at M=1 than affine8) costs <1pp acceptance.
- Method: `OMLX_EAGLE3_MXFP8` env; offline accept A/B bf16 vs mxfp8 on the same run.
- Numbers: accept delta **<1pp** vs bf16 (math even +0.03 mean_len); cheaper matmuls (273 vs 301µs qmv). Halves ~1.9GB/draft-step reads.
- Verdict: WIN — ships as the default drafter precision (opt-in with the EAGLE path).
- Revival: n/a.
- Source: `tasks/todo.md` (EAGLE-3 CLOSE-OUT); `tasks/eagle3_build.md` (mxfp8).
## EXP-041: M3 MoE spec-verify routing measurement (P0 → P1/P2)
- Hypothesis: verification-time expert grouping (P1) or a top-B expert budget (P2, MoE-Spec paper) beats break-even.
- Method: `OMLX_M3_ROUTE_TRACE` recorder inside the router during L=4 verify; 630 windows (210/domain: code/math/prose); union U, mass concentration, gather_qmm sorted vs unsorted; harness `tasks/p0_routing.md`, raw npz `p0_route/`.
- Numbers: P1 — `gather_qmm` ALREADY auto-dedups duplicate experts (~177µs fixed + ~36µs × U-distinct; sorted_indices=True is **1.00×**) → the natural adjacent-token overlap (code f=0.36, math 0.43) is already realized live. P2 — M3's sigmoid router is DIFFUSE: routing-weight mass at B=8-12 only .75-.84 code (needs ~.90); pick-recall code B16=0.86.
- Verdict: DEAD — both NO-GO; spec-decode at batch-1 on M3 is a measured dead end at every level (K-sweep, drafter quality, verify kernels, routing dedup, expert budget).
- Revival: batch≥2. New lead surfaced: ~177µs fixed/gather_qmm-call ×114 calls/token ≈ 20ms/token in BASE decode → gate_up+down call fusion (later attributed to host serialization, EXP-043).
- Source: `tasks/p0_routing.md` (RESULTS); `tasks/todo.md` (P0 ROUTING).
## EXP-042: ThunderMittens sidequest (Metal tile kernels)
- Hypothesis: QuixiAI/ThunderMittens' NVFP4 qgemv (claims "beats fp16 GEMV") helps our MoE/quant hot path.
- Method: port their nvfp4 qgemv verbatim to `mx.fast.metal_kernel` (their native build is a vendored MLX-0.21 fork, incompatible) and bench at our shapes.
- Numbers: **1.4-2.5× SLOWER** than MLX qmv (322 vs 789 GB/s at 8192×6144; MLX runs 96% of ceiling). Their "beats fp16" claim = vs a DEQUANTIZED mx.matmul; their fork predates MLX NVFP4.
- Verdict: DEAD — IGNORE as a dependency. Reference value only: complete fp8 MLA-KV reference (de-risks the deferred int8-MLA-KV port), GQA K/V staging structure, device-resident spec-verify pipeline.
- Revival trigger: upstream ships an indexed quant GEMV at >65% BW, or a modern-MLX rebase.
- Source: `tasks/todo.md` (THUNDERMITTENS CLOSED, TM CAPABILITY MAP).

### Infrastructure / overlap levers (2026-07-04) — host serialization, buffers, task-cap, compile
## EXP-043: Host-serialization profiling (the 17ms/token finding)
- Hypothesis (from "did we profile omlx for stupid inefficiencies?"): the decode critical path has host overhead never before measured (model path only).
- Method: `/usr/bin/sample` (no sudo, symbolizes mlx C++) on the live server during M3 and GLM decode; cross-check MLX v0.31.2 transforms.cpp.
- Numbers: M3 decode thread = 52% cond_wait inside `mx.async_eval` + 18% Metal encode + ~16% python forward + ~9% graph-node ctor → host ~17ms + GPU ~20ms **serialized** = 37ms/token; GPU idles ~45%. GLM = 48ms/token (69% GPU-wait + 16.5% encode + ~15% python). Mechanism: `MAX_ACTIVE_TASKS=10` is a compile-time constant; eval_impl blocks even async_eval callers → BatchGenerator's one-ahead pipelining never overlaps.
- Verdict: WIN (diagnosis) — first host/GPU split measured on the serving stack; framed the ~50 tok/s prize (unrealized — all 3 spawned levers below either shipped only the env win or died). Ranked directions: (1) mx.compile the decode step, (2) gate_up+down call fusion, (3) patch MAX_ACTIVE_TASKS — all tested (EXP-044/045/046).
- Revival: n/a — this framed the whole overlap campaign.
- Source: `tasks/todo.md` (HOST-SERIALIZATION PROFILING); `tasks/overlap_levers.md`.
## EXP-044: Golden-env buffer sweep (OPS/MB_PER_BUFFER) + the memory breach
- Hypothesis: MLX command-buffer commit boundaries add dispatch overhead tunable by `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER`.
- Method: grid OPS {1000,4000,16000} × MB {1000,50000,200000}; watch the memory enforcer; pick a production combo (the golden env). Stock MLX, no rebuild.
- Numbers: **OPS=4000/MB=4000** → M3 26.3→**28.4 (+8%)**, GLM 20.8→**23.0 (+10.6%)**, Nemotron 137 tok/s. The 40MB default MB cap forced early commits (why OPS-alone was flat). MB=4000 ≈ one M3 layer's weights. MEMORY BREACH: **MB=50000 drove GLM 16k prefill peak to 479GB** (failed the ≥60GB-headroom rule); MB=8000 left only 49GB; MB=4000 peaked 365GB (~80GB under soft, plateaus ~390GB at 37k-59k as chunked prefill bounds the working set).
- Verdict: WIN — the only batch-1 win of the overlap campaign; shipped as the standard launch line, works on stock MLX.
- Revival: n/a. Do NOT raise MB toward 50000 (breach).
- Source: `tasks/overlap_levers.md` (GOLDEN ENV, lines 90-95); `mem/omlx-glm52-decode-opts.md`.
## EXP-045: MLX MAX_ACTIVE_TASKS cap raise (10→64) via patched wheel
- Hypothesis: the backpressure wait past 10 active tasks is the wall; raising the cap unblocks overlap.
- Method: cloned mlx v0.31.2, made `MAX_ACTIVE_TASKS` read env `MLX_MAX_ACTIVE_TASKS`, built the stage-2 wheel (ABI-safe, mlx.core untouched); A/B cap {10,64} at golden env; `/usr/bin/sample`.
- Numbers: cap 10→64 killed the backpressure wait ENTIRELY (**8550→0 samples**) yet tok/s **FLAT** (28.27→28.05 short; 16k slightly worse); GLM cached output byte-identical across stock/cap10/cap64. The wait was overlapped / off the critical path; the real wall is main-thread python graph-build (~95% busy) + encode.
- Verdict: DEAD — not a throughput lever at batch-1. Wheel SHELVED at `~/mlx-src` + SHELF.md (insurance for batch-pileup regimes).
- Revival: batch≥2 (per-request host stacks); or as a component of dispatch-gap work.
- Source: `tasks/overlap_levers.md` (CAP A/B DEFINITIVE).
## EXP-046: mx.compile of the decode step (Phase A feasibility → Phase B live)
- Hypothesis: compiling the L=1 decode step kills the ~29% host cost (python + graph-ctor) for a big batch-1 win.
- Method: Phase A = 5 offline spikes (quant ops, custom metal kernels, stateful KV cache via inputs=/outputs=, retrace cost, op-count). Phase B = integrated `compiled_decode.py` behind `OMLX_M3_COMPILE=1` (default OFF); real fs5 parity + live 3-leg A/B.
- Numbers: Phase A all PASS — bit-identical all modes; 4931→3007 ops (−39%); scalars-as-mx.array MANDATORY (else per-token retrace + compile-cache leak); retrace ~11ms/bucket (43µs/token amortized). Phase B — real fs5 parity 220/220 argmax, max|Δlogit|=0; LIVE **FLAT** at batch-1 (24.31→24.47 mid, 24.23→24.26 @16k) despite **86× less host eval work** (eval_impl samples 7635→89). Caught cross-request KV contamination (bucket buffers reseeded only on growth → reseed-per-request fix).
- Verdict: DEAD at batch-1 (proven twice with EXP-045: batch-1 decode is GPU-bandwidth-bound; host never on the critical path). Compiled path SHELVED default-off (compiled_decode.py sha 9959bdc0).
- Revival: batch≥2 (host stacks per request = compile's real regime); CPU/thermal headroom; or as the op-count-cut FOUNDATION for mega-kernel / fewer-larger-dispatch work targeting the residual 2× @16k (41ms vs ~20ms floor = GPU dispatch gaps, ~240 qmm dispatches/token).
- Source: `tasks/compile_spikes/PHASE_A_SUMMARY.md`; `tasks/overlap_levers.md` (LEVER #1 VERDICT, SHELF).

### EAGLE at production temp=1, GLM scout, Nemotron-Super pathfinder (2026-07-05)
## EXP-047: EAGLE at production temp=1 campaign (verify_build/verify_drain discriminator)
- Hypothesis (user): MTP/EAGLE must pay at MiniMax's production sampling (temp=1.0, top_p=0.95, force_sampling); temp0-only is insufficient.
- Method: Gate-0 alpha harness (`eagle_gate0_alpha.py`, drafter q via the REAL adapter + prefix-KV seeding — validated offline, never run live, auth-blocked). Workstream-B leg-C profiler (`OMLX_VLM_MTP_PROFILE`) splitting the round into verify_build (host graph-ctor, timed BEFORE async_eval) vs verify_drain (async_eval drain + walk .tolist, GPU).
- Numbers: verify_build = ~12ms **CONSTANT** 5k→16k (host-build is small, NOT the wall); GPU-wait = ~85% of round, GROWS with ctx. L=2 verify GPU-wait **75ms @5k = 1.83× base single-decode** (commits 1.83 tok → GPU-level break-even), **87ms @16k = 2.12×** (loss). Wrapper host overhead only ~14%. Mechanism: verify L=(K+1) reads ~(K+1)× the dominant expert weights (distinct experts/token) → spec CAPPED at break-even at batch-1 EVEN AT α=1, sampling-INDEPENDENT (so temp1/rejection sampling is strictly worse).
- Verdict: DEAD — NO-GO at batch-1 with mechanism; two hypotheses retracted (sched-roundtrip 0.25ms; the "44ms verify_submit is host-build" was a mode-1 lumping artifact). Confirms + quantifies the standing "spec doesn't pay on M3 MoE" lesson.
- Revival: batch≥2 (expert reads amortize) — the assets (compiled decode L1, cap wheel, vendor-grade drafter, validated alpha harness) are banked for that campaign.
- Source: `tasks/eagle_temp1.md`; `tasks/lessons.md` (SPEC DECODE lesson, amended).
## EXP-048: The 65× wired-limit storm (standalone-vs-server mystery)
- Hypothesis: an isolated eager `model(x, cache)` is inexplicably ~65× slower than the same forward in the server loop (~3100ms vs ~48ms), suggesting a hidden in-loop optimization to harvest.
- Method: EAGLE Gate-1 harness hit the identical symptom (~2.3s/token standalone); isolate the difference vs the omlx server startup.
- Numbers: root cause = the standalone process is missing `mx.set_wired_limit(...)` → unwired weights → GPU page-fault storm on every weight read. The omlx server raises the wired limit at startup. Fix for any standalone repro: `mx.set_wired_limit(506*1024**3)` before the forward.
- Verdict: WIN (methodology) — resolved the mystery and set a standing rule, but does NOT change any perf verdict (the in-loop 48ms path is already the optimized one; there is no hidden 65× to harvest).
- Revival: n/a — this is now a standing rule: any standalone MLX repro must set the wired limit AND copy live dtype/config.
- Source: `GLM52_MTP_FORAY.md §11`; `tasks/todo.md` (GLM-5.2 OPTIMIZATION SCOUT).
## EXP-049: Nemotron-3-Super-120B oQNVFP4 pathfinder (+ SpecPrefill concurrency OPEN-BUG)
- Hypothesis: the NVFP4 byte-repack recipe generalizes to the NemotronH latent-MoE hybrid (pathfinder for the 550B Ultra); quality is preserved.
- Method: `omlx/tools/oqnvfp4_nemotron_convert.py` + `omlx/patches/nemotron_h_nvfp4_ts.py` — keep routed experts NVFP4 (byte-exact), dequant everything else to bf16, carry per-expert `weight_scale_2` folded into router scores (relu² is degree-2 homogeneous → a SINGLE per-expert scalar `fc1_ts²·fc2_ts`, exact; positivity asserted at convert). Per-instance `mixer.__class__` swap (never rebind the class — a sibling model shares the process). Verified through the REAL engine load path.
- Numbers: repack bit-exact (max|diff|=**0.0**); output 79.3GB (77.5 disk). Quality TIED vs oQ4e: gsm8k 92/92, mmlu 81/80, arc 95/94. Speed **0.58×** (29.4 vs 50.4 tok/s @5k warm) = NVFP4 gather_qmm gs16 kernel overhead (4× scale reads vs the mature affine gs64 int4 kernel) — a kernel-opt opportunity, not a correctness issue, and does NOT gate the bandwidth-bound Ultra.
- Verdict: WIN (quality/conversion) — lossless repack + exact ts-fold correct in live serving; the validated template carried to Ultra.
- Verdict (SpecPrefill sub-finding): OPEN-BUG (separate, production risk) — oQ4e's SpecPrefill under 6-way CONCURRENCY collapses gsm8k **92%→15%** (serial + single-shot both fine). The user's default model gives wrong answers under concurrent load with SpecPrefill enabled. NEEDS OWN INVESTIGATION (unresolved as of 2026-07-05). Short-answer benches (mmlu/arc 8-token) hid it; only the 512-token gsm8k exposed it. Discovered as a confound while quality-benching this conversion.
- Revival: NVFP4 gs16 gather_qmm kernel optimization; SpecPrefill concurrency bug is an unclaimed defect.
- Source: `tasks/oqnvfp4_nemotron.md §11`; `tasks/todo.md` (oQNVFP4 NEMOTRON PATHFINDER); `tasks/lessons.md` (Nemotron converter).
