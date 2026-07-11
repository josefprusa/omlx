# GLM-5.2 custom-kernel easy wins (decode + prefill) — DONE 2026-07-03

## Shipped (all in `omlx/patches/glm_moe_dsa/`, kill switch `OMLX_GLM_DISABLE_DECODE_OPT=1`)
- [x] `decode_kernels.py` (new): fused s==1 indexer-scores metal kernel,
      flash-decode sparse-MLA (2-kernel split-K, gated `K >= 98304`,
      tune via `OMLX_GLM_FLASH_DECODE_MIN_K`), `mh_qmm_m1` (embed_q via
      gather_qmm, bit-identical).
- [x] `deepseek_v32.py`: Indexer s==1 uses fused scores kernel (mask-free
      decode only); MoE weighted-sum as one gemv instead of mul+sum+astype.
- [x] `glm_moe_dsa_model.py`: L==1 flash-decode branch + embed_q swap.

## Live A/B (server, greedy, same prompt, OFF = kill switch = old behavior)
| context | OFF | ON | delta |
|---|---|---|---|
| ~1k    | 20.8 | 21.7 tok/s | +4.3% (-2.0 ms/tok) |
| 14.5k  | 19.5 | 19.9 tok/s | +2.1% |
| 57.7k  | 18.2 | 18.5 tok/s | +1.6% |
Greedy output identical on the bench prompt. Synthetic layer out max_abs 2e-3,
topk overlap 99.95% (fp16 tie noise, same class as the #1984 native kernels).

## Measured and REJECTED (don't re-litigate without new data)
- Fused router kernel: mx.compile'd `group_expert_select` already wins (36us);
  my kernel was 3x slower (uncoalesced weight reads).
- Custom multi-head 4-bit qmv for embed_q/unembed: barely beats the free
  gather_qmm swap; unembed at parity. Dropped.
- Beating MLX MoE qmm: gather_qmm standalone is 83-84% of bandwidth
  (my earlier "60%" missed 13MB scales/biases + in-stream overlap).
- split+swiglu fusion: no-op (mlx_lm swiglu already compiled, split is a view).
- Flash decode below ~64k: MLX gather+SDPA chain wins (compact 2.4MB buffer
  stays SLC-hot); crossover ~64k, kernel engaged at >=98k only.
- PREFILL: at practical ceiling everywhere. MoE 89% peak, projections 89%,
  fused_indexer_scores 24.7 TF/s (~88%) flat in K, sparse_mla topk-bound
  (flat 208ms vs K). No prefill kernel win short of block-union research.

## The two "remaining levers" — BUILT AND MEASURED NEGATIVE (2026-07-03)
- Dual-stream overlap (shared expert on 2nd mx.stream): 53.8 -> 69.9 ms/token
  (23% SLOWER; ~200us/token cross-stream fence cost over 75 layers) AND
  numerically wrong (max_abs_diff=18 — MLX buffer reuse appears stream-unsafe
  here). Dead on this MLX version. Probe: scratchpad/dualstream_probe.py.
- Read-once flash-decode v4 (all 64 heads per 1024-thread TG, 2-phase
  score/PV through TG mem): correct (2e-4) but 180-430us vs MLX 69us — barrier
  +TG-mem coordination cost exceeds the SLC re-read it eliminates. Fourth and
  final kernel architecture tried (v1 114, v2 108, v3 78 — v3 shipped for
  K>=98k, v4 worst). CONCLUSION: MLX's gather+SDPA (compact 2.4MB buffer,
  SLC-hot 64-head re-read at ~3TB/s) is near-optimal below ~64k on M3 Ultra.
  Probe: scratchpad/flashdecode_v4.py.
- Consequence for MTP: verify(2) stays ~85-88ms even with opts extended to
  L=2; at alpha 0.6 S=0.84, at 0.75 S=0.92 -> MTP remains a loss without a
  dedicated fix for the M=2 kernel cliff (~28ms). Parked.
- MTP literature check (2026-07-03): GLM-5 paper Table 2: accept length 2.76
  at 4 speculative steps (DeepSeek-V3.2: 2.55); GLM-5.2 head +20% accept
  length + IndexShare (our config's index_share_for_mtp_iteration). DeepSeek
  official: 80-90% second-token acceptance, 1.8x on GPU serving. => our
  measured alpha=0.41 is OUR cold-cache graft, real alpha1 ~0.8 achievable.
  BUT on our stack verify(L)/t1 ~= 1.9-2.8 (GPU stacks ~1.05-1.15), so even
  paper-grade acceptance gives only ~1.0-1.05x short, ~1.10-1.17x @58k
  (K=3-4). Binding constraint = verify cost (M>1 kernel cliff), NOT alpha.

## MTP chained K built + tested live (2026-07-03) — WORKS, still a net loss
Implemented (mlx_lm_mtp patches, all inert while mtp_enabled=false):
- FIXED legacy bug: MTP cache skipped every other committed position (gap
  cache). Now dense: every cycle refreshes with TRUE trunk hiddens from the
  verify forward (one L=m+1 mtp_forward that doubles as draft 1).
- FIXED trim bug: _restore_or_trim_caches hard-trimmed 1 (legacy L=2);
  chained K left K-m-1 garbage KV entries -> corrupted output ("(, (, (,").
  Now trims block_size-(accepted+1).
- K-draft self-chaining (OMLX_MTP_DRAFT_K, greedy+no-procs only),
  per-position acceptance logging, prompt-warm hook (attaches in
  GenerationBatch.__init__ — NEVER FIRES: omlx prefills outside it via the
  prompt processor; warm rows=0 always. Not binding, see below).
Live results (fresh 3.7k prompt, correct caches, clean coherent output):
  alpha1=73% (!! dense-cache fix works; near teacher-forced ceiling),
  alpha2|1=15% (self-chained hidden collapses on this quantized stack),
  verify(3)=124ms live. K=2: 14.3 tok/s (0.66x). K=5: 9.5 tok/s (0.44x).
  Baseline 21.7. Break-even at K=2 needs tokens/cycle 2.84 (max 3.0) =>
  alpha~0.92 everywhere. CONFIRMED DEAD on M3 Ultra batch-1: the verify-cost
  wall, with working acceptance, measured end-to-end. mtp_enabled=false
  restored. Open threads if ever revisited: warm hook needs the omlx prompt
  processor (not GenerationBatch.__init__); alpha2 collapse cause (chain
  hidden quality vs Q4-head/3-bit-trunk drift); L=6 verify not byte-lossless
  vs L=1 decode (fp16 small-L path, occasional argmax flips).

## CORE-MLX CAMPAIGN (2026-07-03, in progress) — native csrc primitives, >=2% counts
Toolchain proven: omlx/custom_kernels/glm_moe_dsa/csrc + nanobind 2.12.0 pin,
build: rm -rf build; CMAKE_ARGS="-DPython_EXECUTABLE=$PWD/.venv/bin/python" \
  OMLX_WITH_CUSTOM_KERNEL=1 .venv/bin/python setup.py build_ext --inplace --with-custom-kernel
Targets (measured basis in this file above):
1. [x] K-A "sparse_mla_decode" — MEASURED NEGATIVE, CLOSED. v5 = faithful
   clone of MLX v0.31.2 sdpa_vector (32 sg x 32 lanes, 32 keys in flight,
   transpose-combine) + in-kernel topk gather + fused pe: 92-94us vs MLX
   chain 67-70us at <=32k, 0.92x at 131k (v3 still better there). FIFTH and
   FINAL architecture (v1 114, v2 108, v3 78 shipped >=98k, v4 430, v5 93).
   Root cause now proven: materialize-once-then-sequential-SDPA beats ANY
   in-kernel random-gather variant on M3 Ultra — the gather op amortizes the
   random access once; everything downstream streams sequentially. Probe:
   scratchpad/flashdecode_v5.py. (original plan below for reference)
   [orig] clone MLX sdpa_vector.metal (match installed
   mlx version) into csrc; adapt: keys/values read via topk indices (in-kernel
   gather from the FULL kv_latent cache), pe-score term fused (q_pe . k_pe[idx]
   added pre-softmax), mask-free L=1, 64 heads / kv-head=1. Replaces
   gather(31us)+pe(19)+sdpa(52) = ~102us/layer with one ~55-65us dispatch
   -> ~+6% decode at all contexts. Integrate at L==1 in glm_moe_dsa_model.py
   (same guard block as flash_decode; native wins over both). Verify: allclose
   vs MLX chain, then live A/B (decode_bench, longctx_stream).
2. [ ] K-B fused rms_norm+qmv prologue primitive (proj chain 70us vs ~31
   floor): only after K-A lands; must match MLX qmv throughput (port their
   qmv kernel, add per-TG rms prologue). ~+4% decode.
3. [ ] K-C q8_vup_flat at decode L==1 (kernel EXISTS; relax the Python
   key_length gate + shape test at L=1): ~+1%, do as rider with K-A.
4. [ ] K-D int8 MLA-KV port: REAL scope = quantized latent cache class +
   int8 variants of glm_dsa_sparse_mla_attention/indexer kernels + L==1
   decode path. Prefill attention reads halve (sparse_mla 208ms/layer is
   part-memory-bound) + decode gather/sdpa halve + 2x context capacity.
   LOSSY (int8 KV) — needs user sign-off on quality tradeoff. Multi-day.
Notes: mlx sdpa_vector source: github.com/ml-explore/mlx (match version tag),
mlx/backend/metal/kernels/sdpa_vector.h (+ scaled_dot_product_attention.cpp
dispatch). Our csrc has steel_sparse_mla (prefill) to crib gather/topk
addressing from. Live baseline to beat: 21.7 tok/s short / 18.5 @58k.

### K-B design notes (2026-07-03, in progress)
- MLX qmv_fast_impl (scratchpad/quantized.h:750): 2 simdgroups/TG, 4 rows
  each (8 out-rows/TG), lane walks packs; helpers load_vector/qdot/
  get_pack_factor live in same header — paste needed 4-bit variants into
  mx.fast.metal_kernel `header` param (JIT path, no csrc rebuild).
- Fused kernel = rms prologue (cooperative ssq -> rsqrt(ssq/N+eps), stage
  xn[i]=x[i]*scale*norm_w[i] into TG mem, 12KB max) + qmv reading x from TG
  mem. RISK: MLX uses 64-thread TGs (out/8 TGs); each re-stages x
  redundantly (256 TGs x 12KB for q_a) — may eat the win. Mitigation: fatter
  TGs (8 sgs = 32 rows/TG -> 64 TGs for q_a) and measure vs MLX qmv alone.
- Sites: input_norm->{q_a[2048,6144], kv_a[576,6144]} (SAME normed x — do
  BOTH in one kernel! concat out rows 2624, saves 2 dispatches+norm),
  q_a_norm->q_b[16384,2048]. All 4-bit affine g64.
- Gate: fused_time <= qmv_time + 2us, allclose vs norm+qmv chain 1e-2.
  Integrate in GlmMoeDsaAttention.__call__ L==1 only, decode_kernels.py,
  same OMLX_GLM_DISABLE_DECODE_OPT kill switch. Then live A/B.
- K-C after: relax q8_vup_flat key_length gate (sparse_mla.py:493) at L==1
  decode + shape test; ~+1%.
- K-D after: int8 MLA-KV — user approved lossy. Scope: quantized latent
  cache (mlx_lm QuantizedKVCache? or int8 view + dequant-in-kernel),
  int8 variants of csrc sparse_mla/indexer kernels, L==1 decode path,
  prefill chunked path. Multi-day; plan first, then execute.
- K-B extraction note: scratchpad/quantized.h has the pieces — load_vector
  variants at lines ~28-190 (4-bit variant PRE-SCALES x per nibble position:
  x[4i+1]/=16 etc so qdot's masked-nibble mults work), qdot at ~292-435
  (bits==4 branch: uint16 masks 0x000f/0x00f0/0x0f00/0xf000), qmv_fast_impl
  at 750-814. My helpers_extract.txt got mangled (bad sed ranges) — re-extract
  with: sed -n '28,190p;292,436p;750,814p'. For fused kernel: adapt
  load_vector to read from threadgroup half* (normed x staged by prologue)
  instead of device — one signature change.
- K-B CLOSED NEGATIVE (2026-07-03): fused rms+qmv correct (rel 1e-3) but
  parity-to-loss vs chain (q_a 31.0 vs 30.6, kv_a 29.5 vs 21.4, q_b 30.1 vs
  27.1 us). Premise was wrong: chain minus qmv-only = 2-4us (rms_norm
  pipelines); proj_chain "70us vs 31 floor" gap was floor-math optimism —
  the 3 qmvs standalone sum to 69us. No capturable glue. Probe:
  scratchpad/rms_qmv_probe.py.
- K-C CLOSED NEGATIVE (2026-07-03): glm_dsa_q8_vup_flat WORKS at L=1
  (max_abs 4.4e-3) but 34.3us vs MLX chain 23.1us — prefill-shaped kernel,
  underutilized at M=1. No integration. Probe: scratchpad/q8vup_probe.py.

### K-D int8 MLA-KV — PHASED PLAN (user approved lossy; in progress)
Phase 1a (decode, simple): store MLA latent(512)+k_pe(64) + indexer k(128)
  in mlx_lm QuantizedKVCache-style (8-bit g64) per layer; convert POST-
  prefill (mlx_lm kv_bits pattern: prefill fp16, quantize once, decode
  appends quantized). Decode L==1: take_along_axis gather of packed+scales
  +biases (halves gather reads) -> mx.dequantize the 2048 compact rows
  (~5us) -> existing fp16 pe+SDPA unchanged. Native prefill kernels
  untouched (fp16 during prefill). Memory: KV halves post-prefill.
Phase 1b: gathered-QUANTIZED SDPA (mlx quantized sdpa or custom) to halve
  the 151MB SLC re-read -> sdpa ~52->~30us, total attn ~-3ms/tok (+7%).
Phase 2 (prefill, csrc): int8 variants of glm_dsa_sparse_mla_attention +
  fused_indexer_scores so prefill reads int8 too (chunked prefill reads
  prior chunks from cache) -> prefill attention reads halve; also fixes
  multi-turn extension on a quantized cache (today would need fp16
  fallback/dequant on extend — Phase 1 must gate/fallback there).
CAUTION: omlx SSD/prefix cache stores fp16 blocks — quantized in-memory
  cache interplay needs checking (extend/save paths); Phase 1 may need
  per-model flag int8_mla_kv_enabled (settings keys already exist in
  ~/.omlx/model_settings.json from a dead branch — wire them for real).
- K-D Phase-1a probe (scratchpad/int8kv_probe.py, FIXED bench — dependency
  must hang on q_lat, NOT `cache + 0*xi` which copies the full cache/call):
  int8-gathered decode attn: 76.7us vs fp16 66.9 at K=32k (LOSES — dequant
  + 3-array gather beats the read halving while SLC absorbs); 79.7 vs 97.0
  at K=131k (1.22x — but SAME 79us as shipped flash-v3 there, NOT additive).
  max_abs 1.4e-3/layer (acceptable lossy). => Phase-1a speed value ~nil;
  Phase 1 justification = KV memory halving (2x context) + gateway to 1b/2.
  NEXT: (1) test Phase-1b: mlx quantized-SDPA on gathered tuples vs fp16
  SDPA at decode shapes (the 52us SLC re-read is the prize, ~-1.5ms/tok if
  it halves); (2) if 1b wins, implement QuantizedKVCache integration per the
  phased plan above (convert post-prefill, dequant-on-extend, SSD-save
  interplay, wire int8_mla_kv_enabled setting); (3) Phase-2 csrc int8
  prefill kernels for the prefill read halving.
- K-D Phase-1b CLOSED NEGATIVE: mlx quantized_scaled_dot_product_attention
  (unfused qmm+softmax+qmm) on gathered tuples: 130us vs fused fp16 SDPA 66us
  @32k, 130 vs 94 @131k. Probe: scratchpad/int8kv_1b.py.
- K-D VERDICT (2026-07-03): no >=2% decode boost exists in int8 MLA-KV on
  this stack. 1a ~nil-to-negative at <=32k, duplicates shipped flash-v3 at
  >=100k; 1b 2x slower. Surviving value is MEMORY ONLY (KV halves -> ~2x
  context capacity, quality tax 1.4e-3/layer) — a capacity feature, not a
  boost; needs the full integration scope (convert post-prefill, dequant-on-
  extend, SSD-save interplay) if ever chartered. Phase-2 prefill int8 csrc:
  revised DOWN — sparse_mla is K-flat + ~40% ALU (staged tiles, SLC-heavy);
  int8 adds dequant ALU; est ~1.0-1.15x uncertain, not a confident >=2%.
CAMPAIGN CLOSED: B, C, D all resolved by measurement. Shipped total stands
at +2-4% decode (yesterday's opts + flash-v3 >=98k). This model/quant/HW
combination is measured out.

## Lead sweep (2026-07-03 night): upstream recon + block-union measured
- Upstream: jundot IS the omlx maintainer; the custom MLX fork was RETIRED
  (PR #1984 refactored to vanilla mlx 0.31.2). Recent PRs (#2070 sharded
  sparse-MLA, infra fixes) contain nothing new for single-box GLM decode/
  int8/MTP. No found money. Remaining upstream plays: file dispatch-ledger
  issue on ml-explore/mlx (fused MLA decode + fused quantized sdpa-vector);
  ask jundot to extend V4-style fused prefill attention to glm_moe_dsa.
- Lead #5 block-union: MEASURED DEAD on real data. Env-gated topk dump hook
  added (OMLX_GLM_TOPK_DUMP, layers 6/30/62) — captured 2 chunks of a real
  20k code prompt. Union ratios: L6 1.41-2.19x (Bq8-64), L30 1.70-3.31x,
  L62 2.00-4.55x. Overlap decays with depth; at dense-viable tile sizes
  (Bq>=32) inflation 1.8-4.5x >= any efficiency gain (~1.6-1.8x). The last
  unexplored prefill idea is now closed with data; prefill-at-ceiling
  verdict stands on all fronts. Data: ~/.claude/jobs/62f9cfe9/tmp/topkdump/,
  analysis union_analyze.py.
- omlx serving-layer tax MEASURED ZERO (2026-07-03): bare mlx_lm
  stream_generate with identical patched model code = 21.38 tok/s vs omlx
  server 21.7 (server slightly faster — better pipelining). Model card's
  21.29 = bare mlx-lm confirmed. No optimization available in the omlx
  scheduler/sampler/stream path. Ops note: sudo sysctl
  iogpu.wired_limit_mb=518144 raises Metal cap 464->506GB (+42GB KV
  headroom for very long contexts; server log warns about it at startup).
  Bench: ~/.claude/jobs/62f9cfe9/tmp/standalone_bench.py.

# Kimi-K2.7-Code VLM (2026-07-03, in progress)
Model: avlp12/Kimi-K2.7-Code-Alis-MLX-Dynamic-3.6bpw-VLM — 1T MoE (32B act),
DeepSeek-V3 arch (61L, 384 experts top-8, MLA 512-latent, NO DSA, NO MTP),
MoonViT bf16 vision, 466GB, mlx_vlm kimi_k25 (text delegates to mlx-lm
deepseek_v3). Card claims ~23 tok/s on this hardware. 256k ctx.
- [x] download started (~123MB/s, ETA ~60min from 02:20) -> ~/.omlx/models/avlp12/
- [x] deps: tiktoken+blobfile installed; mlx_vlm 0.6.3 has kimi_k25 native
- [x] NEW PATCH omlx/patches/dsv3_decode_opts.py (kill switch
  OMLX_DISABLE_DSV3_DECODE_OPT=1), wired in model_loading for
  deepseek_v3/kimi_k2/kimi_k25: (1) QuantizedMultiLinear M=1 -> gather_qmm
  (<=1ulp, GLM-proven 1.6x on embed_q shapes), (2) dv3 MoE wsum as one gemv.
  Expected ~+2ms/token combined (~+4-5%) by GLM analogy: 61-62 layers.
- [ ] REQUIRED before load: sudo sysctl iogpu.wired_limit_mb=518144 (peak
  467GB > default 464GB Metal cap) — USER must run.
- [ ] after download: omlx serves it (engine pool; GLM unloads on demand);
  baseline decode bench short/16k/64k via chat endpoint; A/B kill switch;
  check VLM serving-loop tax (bare mlx_vlm generate vs omlx server — the
  LLM path was zero-tax, VLM path unmeasured).
- Kimi-specific ledger notes: no DSA -> decode attention reads FULL latent
  cache (C x 576 x 2B x 61 layers: 2.3GB/tok @32k, 9GB @128k) -> long-ctx
  decode sags; int8-KV would halve it but mlx qsdpa is unfused/2x-slower
  (measured) -> upstream fused int8 sdpa_vector is the unlock. Router
  already compiled; SwitchGLU unfused gate/up (omlx GLM fork fuses them —
  portable trick, unmeasured win, medium effort).
- [x] RESULTS (2026-07-03): download OK (434GiB, 69min). Wired limit set by
  user (518144). Live A/B, same greedy prompt, server path:
  OFF (vanilla) 24.6 tok/s | ON (dsv3_decode_opts) 25.4 tok/s = +3.3%
  (card reference: ~23 -> we serve it ~10% above the card). Output coherent
  both ways; wording diverges slightly (<=1-ulp embed_q swap flips rare
  near-ties — same numerics class as GLM opts). Patch log line confirms
  application. GLM SSD-cache blocks incompatible with Kimi (expected; Kimi
  builds its own).
- Remaining (unmeasured, next session): long-ctx decode curve (16k/64k —
  expect sag from full-latent attention reads, no DSA), VLM serving-loop
  tax (bare mlx_vlm vs server), image/video path bench (custom vision_3d.py
  overlay from the model repo not yet integrated), fused gate_up port
  (concat gate+up at sanitize -> one gather_qmm, GLM-fork trick, medium).
- [x] BUG FIX omlx/memory_monitor.py estimate_mla_kv_bytes_per_token:
  (1) descend into text_config for VLM configs (kimi_k25), (2) count plain
  per-layer KVCache (no CacheList) as main MLA cache. Was falling back to
  expanded-MHA formula -> ~23x KV overestimate -> prefill guard rejected
  Kimi prompts >~4k tokens entirely. Now 70,272 B/token (61x576x2B ✓),
  128k = 9.2GB. Long context on Kimi was impossible before this.
- Kimi long-ctx bench: 16k = 21.2 tok/s decode (vs 25.4 short; on the
  no-DSA bandwidth curve). 64k/128k legs CANCELLED by user — fresh prefill
  in the throttled regime crawls (~23 tok/s at 24k depth vs GLM's ~180).
  NEXT (highest priority for Kimi): the phantom-transient investigation —
  estimate_chunk_transient_bytes appears to reserve for MATERIALIZED
  attention scores (~4.4MB/chunk-token @14k depth, from the 400-error
  preflight line) while MLX SDPA is flash-style and never materializes
  them -> ~30-40GB phantom reservation on a box with ~50GB headroom ->
  chunks shrink -> fresh prefill ~2-8x slower than physics requires.
  Plan: measure actual transient peak (mx.get_peak_memory around a chunk
  at depth) vs the estimate; tighten estimator for fused-SDPA models;
  before/after with a cache-busted 64k prefill.

# MiniMax-M3-oQ4 (unigilby, 428B MoE VLM, 23B active, 228GB, minimax_m3_vl)
README requirements vs our tree (2026-07-03):
- [x] scheduler.py minimax_m3_vl patch: MERGED (16 refs)
- [x] mlx-vlm PR #1374: NOT NEEDED — mlx_vlm_minimax_m3_compat VENDORS the
  full removed M3 model+parser modules and registers minimax_m3/_vl types.
- [x] mm:think + token 200058: present in omlx/api/thinking.py (landed as
  thinking.py, not utils.py as README says)
- [?] tool_calling.py bare <key>value</key> parser: verify against README
  repo files after download (may live in vendored parser; test tool calls)
- [x] torch 2.12.1 + torchvision installed
- [ ] model_settings: trust_remote_code true, temp 1.0, top_p 0.95, top_k 40,
  force_sampling true. Reference perf: decode ~21.7, prefill ~214 tok/s.
- [x] download started -> ~/.omlx/models/unigilby/MiniMax-M3-oQ4 (228GB,
  ~30-50min; log ~/.claude/jobs/62f9cfe9/tmp/m3_download.log)
- [x] LOW-HANGING from GLM/Kimi playbook applied to M3: vendored
  language.py:1444 wsum mul+sum -> one gemv (direct edit, vendored file is
  ours). Audit: router already @mx.compile'd, activation compiled shapeless,
  MiniMaxPackedSwitchGLU fused gate_up already present — vendored code is
  MODERN. QuantizedMultiLinear swap N/A (no MLA). Remaining checks at first
  load: (1) minimax_msa native kernel ABI-probe engages (GLM nanobind
  lesson — silent fallback would cost the sparse-attn speedup), (2) VLM
  serving-loop tax measurement (covers Kimi too), (3) ledger only if decode
  lands well under the ~63 tok/s bandwidth ceiling (card: 21.7).
- [x] M3 RUNNING: 26.5 tok/s short decode (+22% vs card's 21.7, wsum edit
  in), prefill 182 tok/s JIT-warm (card 214, first-request TTFT pays ~50s
  Metal JIT). Native ext healthy (minimax_msa_topk only — attention is MLX
  by design). Settings entry added (trust_remote_code, card sampling).
- [!] OPEN ANOMALY: decode sags 26.5 -> 19.6 by 3.6k ctx (-26%; physics says
  -6%). NOT the sparse-attn branch: raising engage floor to 32k changed
  nothing (19.7 sparse == 19.6 dense) -> cost is in the always-on per-token
  glue: suspects = index-cache bookkeeping (update_index_and_fetch),
  _normalize_attention_mask building mask arrays per token/layer, or
  BatchKVCache dynamic_roll copies. NEXT: dispatch ledger on M3 decode at
  3.6k ctx (GLM playbook); ~5ms/token recoverable if it's mask/glue.
  OMLX_M3_SPARSE_MIN_K env knob added (default 2048 = original behavior).
- [x] M3 SAG ROOT-CAUSED via mini-ledger (scratchpad m3_ledger.py):
  57/60 layers have sparse index ENABLED ([0,0,0,1..] = first 3 off).
  Per-layer decode: idx-OFF 410->477us (K 256->3712, honest physics);
  idx-ON 558->884us. Index machinery = +148us/layer even below the 2048
  gate (cache-update glue) and +407us/layer at 3.7k => ~23ms/token across
  57 layers = the 26.5->19.6 sag. The math is ~us; it's 30-50 dispatches of
  MLX glue (mask normalize, block pool/reshape, topk, compact checks).
- [ ] FIX (GLM playbook, kernel #2 of that family): fuse the per-token M3
  index chain (idx scores over K + 128-block pooling + topk-16 blocks) into
  one mx.fast.metal_kernel; native minimax_msa_topk already exists for the
  topk part — kernel needs scores+pool fused, or extend to full chain.
  Projected: ~280-350us/layer back on 57 layers => +14-18ms/token at 3.7k
  -> decode ~19.6 -> ~24-25 tok/s at context (+25%!). Secondary: trim the
  below-gate glue (+148us/layer at ALL contexts incl short: ~8ms/token ->
  short decode 26.5 -> ~30 possible!). BIGGEST remaining win in the fleet.
- [x] M3 fused index kernel BUILT + BIT-EXACT (fused_index.py, max_abs=0.0
  at K=2049/3700/16384; kill OMLX_M3_DISABLE_FUSED_INDEX): spliced into
  _build_sparse_decode_indices (compact path). BUT ledger unchanged at 3.7k
  (892us idx-ON) — that regime takes _build_sparse_mask -> compiled mask
  path, NOT compact. AND floor-test previously showed selection≈free =>
  REVISED attribution: the +150->407us/layer (O(K)-ish) leak is NOT
  scores/topk; remaining suspects RANKED: (1) MiniMaxM3KVCache vs plain
  KVCache class overhead (ledger idx-OFF used plain KVCache — confound!),
  (2) masked-dense sdpa consuming [1,H,1,K] token mask (O(K) mask read),
  (3) index proj/norm/rope/transpose glue x57.
  NEXT PROBE (decisive, decode_sub pattern): idx-ON layer with plain
  KVCache forced + per-piece timing of idx chain; then fix what it names.
  Kernel stays (engages on compact path at deeper ctx, bit-exact, free).
- [!] M3 DEEP FINDING (2026-07-03 night): LIVE decode runs cache=plain
  KVCache, use_sparse_mask=False on ALL layers — SPARSE ATTENTION IS
  SILENTLY DISABLED IN LIVE SERVING (debug: [M3DBG] cache=KVCache
  mask_none=True pos_ids=True sparse=False). Explains why no vendored-code
  fix moved live numbers. Note: debug capture was on an SSD-cached prompt —
  check whether make_prompt_cache(self.model) (scheduler:2881/2899/3901,
  defers to Model.make_cache which DOES return MiniMaxM3KVCache) is only
  bypassed on the SSD-restore path (scheduler:2758/5494 name the M3 classes
  for serialization; the RESTORE may rebuild plain KVCache) — OR whether
  fresh-prompt requests also get plain caches (then the wrapper the
  scheduler holds isn't the Model with make_cache). ONE more debug run with
  a fresh-nonce prompt (OMLX_M3_DEBUG_PATH=1, _dbg_done reset) answers it.
- STAGED + LAYER-VERIFIED fixes awaiting the serving fix to pay off live:
  (1) fused_index.py kernel (bit-exact), (2) compact-gate density -> 1.0
  (OMLX_M3_COMPACT_MAX_DENSITY; ledger 894->741us/layer at 3.7k),
  (3) batch-singleton positions neutralization (_omlx_all_zero_padding).
  Ledger-projected once live engages: ~-9ms/token at 3.7k => 19.6 -> ~23+;
  AND sparse actually working = the designed long-context behavior back.
- Debug instrumentation left in (env OMLX_M3_DEBUG_PATH, one-shot).
- [!] Plan A+B code APPLIED (all edits in vendored language.py, syntax OK,
  pycache purged) but live STILL 19.6 at 3.6k AND M3DBG never printed with
  OMLX_M3_DEBUG_PATH=1 — the served process is NOT executing the edited
  module code. NEXT: verify which file the server actually imports
  (print(mlx_vlm.models.minimax_m3_vl.language.__file__) inside the server
  venv; check _install_vendor_namespace path precedence — an installed
  mlx_vlm minimax module or another vendor copy may shadow ours), then rerun
  verification steps 1-6 from plans/shimmying-imagining-thunder.md.
  C (SSD guard) + D (scheduler hardening) still TODO per plan.
- [x] M3 PLAN EXECUTED + BENCH CONTAMINATION SOLVED (2026-07-03 eve): the
  eternal "19.6 @3.6k sag" was GLM — m3_16k.py requested GLM-5.2 (sed no-op);
  M3's real curve was never measured. Also: server restarts raced (old
  server drains on :8000 while new scans SSD) — several "no change" results
  were served by old code. ALSO: layer-0 debug artifact (non-sparse layer)
  faked the "plain KVCache" theory — fresh requests DO get
  MiniMaxM3BatchKVCache; sparse WAS running (masked-dense).
  REAL M3 numbers (fresh nonces, verified server): short 26.3-26.5;
  3.8k masked-dense 21.6, compact 20.5 (density 0.54 — masked wins; original
  0.5 threshold restored as default); 16.4k compact 19.6 decode,
  273 tok/s prefill (card: 214). 4 concurrent padded requests: all coherent
  (padded-row correctness fix working), aggregate 11.5 tok/s incl prefills.
  SHIPPED: physical-positions correctness fix (B1), B>1 compact gates +
  per-row mask gather, zero-pad flag propagation, fused index kernel
  (engages with compact at >=4k... B==1 shape), omlx wrapper kwargs
  passthrough (minimax_m3_sparse_attention.py). Knobs:
  OMLX_M3_COMPACT_MAX_DENSITY (0.5), OMLX_M3_DISABLE_FUSED_INDEX,
  OMLX_M3_SPARSE_MIN_K, OMLX_M3_DEBUG_PATH.
  Plan items C (SSD-restore guard: theory weakened — needs a REAL repro
  before building) and D (scheduler hardening for latent crashes) remain
  open; batched fused-index kernel optional.
- [x] M3 PUSHED FURTHER (2026-07-03 night):
  (1) PREFILL +25%: neutralized trivial positions+causal-array mask for
  unpadded singleton chunked prefill (isinstance(offset,int) guard) ->
  native MSA prefill path engages: 16k fresh prefill 273 -> 342 tok/s
  (warm-to-warm 60.1s -> 48.0s TTFT).
  (2) DECODE +12.5% at mid-context: SPARSE_MIN_K default 2048 -> 4096
  eliminates the masked-dense band entirely (dense <4k, compact >=4k):
  3.8k decode 21.6 -> 24.3 tok/s. 16k compact unchanged 19.6.
  Final M3 curve: 26.5 short / 24.3 @3.8k / 19.6 @16k decode,
  342 tok/s prefill (card reference: 21.7 decode / 214 prefill).
- [~] M3 long-ctx slope attack (2026-07-04): measured slope 0.157ms/1k
  (17.0 @66k). Standalone attribution blamed unsorted compact gather (+146us
  16k->64k) + kernel scalar loads — BOTH fixed (sort-always: lossless;
  half4 loads) but LIVE 64k unchanged (16.9 @67.7k = exactly on old line).
  Lesson repeated: standalone reconstructions over-localize; the live slope
  lives elsewhere (candidates: index cache update path, wrapper layers,
  position machinery, batch-cache make_mask O(K)). NEXT: in-situ timing —
  extend the OMLX_M3_DEBUG_PATH hook to time real path segments inside the
  serving process at two depths, then fix what IT names. Both micro-fixes
  kept (harmless/lossless).
- [~] M3 FIRST-PRINCIPLES ASSUMPTION LEDGER (2026-07-03, user-mandated
  "trust no assumption" pass; Codex second opinions running in parallel).
  NEW DATA: 9.5k fresh probe = 20.0 tok/s, per-chunk rate FLAT across
  window. Curve is therefore CLIFF (+8.8ms/tok between 3.8k and 9.5k,
  brackets the SPARSE_MIN_K=4096 floor AND prefill chunk step=4096) then
  gentle ~0.15ms/1k slope. NOT a smooth O(K) decay. Assumptions:
  A1 vendored file runs live — VERIFIED (patch log line + M3DBG earlier).
  A2 >4k decode takes compact+zeropad path — UNVERIFIED! only datapoint
     (3.8k) is BELOW the 4096 floor i.e. ran DENSE. Census answers.
  A3 fused index kernel engages live — UNVERIFIED (bit-exact standalone
     only). Census fused_hit/fused_none answers.
  A4 smooth O(K) slope — INVALIDATED (cliff at 4-9k, then 0.15ms/1k).
  A5 sparse faster than dense above 4096 — SUSPECT: ledger says sparse
     fixed overhead ~+7ms/tok vs dense extra reads ~+2ms @16k. 2x2 A/B
     matrix {dense-forced,sparse}x{16k,64k} answers. Dense-at-depth is
     off-distribution risk (model trained sparse) — quality sniff needed.
  A6 GLM sdpa256 patch inert for M3 — VERIFIED (head_dim==256 && L>1).
  A7 cache append O(1) amortized — VERIFIED by code read (step prealloc
     both KV & index keys; batch cache delegates to BatchKVCache).
  A8 decode window steady-state — VERIFIED @9.5k (flat quartiles).
  A9 thermal contamination post-prefill — WEAKENED by A8 flatness.
  A10 benches measured M3 — VERIFIED (server log Chat completion lines).
  A11 chunked prefill (step=4096) preserves zeropad flag/cache class —
     UNVERIFIED. Census at 16k answers.
  A12 3 non-sparse full-attn layers (0,1,2) — VERIFIED (config freq list,
     memory-monitor "60 layers (3 KVCache)"; layer 0 attn+mlp 8-bit).
  A13 physics predicts only ~2ms/tok extra @64k — recompute post-census.
  A14 B=1 zero-pad batch cache == singleton behavior — code-read plausible,
     census cross-checks live class+flag.
  A15 probe MODEL field authoritative — NO: first SSE chunk can be
     model="keepalive" artifact; use server log cross-check.
  A16 COMPACT_MAX_DENSITY default 0.5 — WRONG, file default now 1.0 (env
     unset live). Irrelevant >4k (density<<0.5).
  A17 GLM 18.5@58k slope "understood" — UNEXAMINED; GLM ~0.14ms/1k is the
     SAME order as M3 0.157 — possible common server-side per-step cost;
     cross-model control candidate.
  A18 SSD/RAM-restored prefixes keep sparse — UNKNOWN (16:11 M3DBG line
     predates instrument gating = layer-0 artifact, inconclusive).
  A19 prefill throttle doesn't affect decode — plausible, unverified;
     phantom over-reservation hit M3 too (9.5k prompt predicted 297GB!).
  Server facts: prefill chunk=4096 (adaptive), decode KV materialization
  every 256 tokens, census instrument OMLX_M3_DEBUG_PATH=<N> now
  multi-shot with branch counters (fused/compact/masked/maskless).
- [x] M3 SLOPE ROOT CAUSE FOUND & FIXED (2026-07-03, first-principles pass):
  Census (multi-shot OMLX_M3_DEBUG_PATH) revealed fused_none=57 +
  scores_fallback=57 EVERY layer EVERY step at 16k — the fused index kernel
  NEVER engaged live. Cause: kernel dtype gate required fp16; live model
  runs bf16 (torch_dtype); standalone verification had used
  set_dtype(fp16). Fallback = full-cache fp32 astype x2 + matmul + pad +
  reshape + blockmax on 57 layers/token = O(K) traffic (~1.9GB/tok @16k,
  ~7.7GB @64k) — the anomalous slope. FIX: dtype-generic scalar loads
  (fp16+bf16, MLX auto-instantiates per dtype), gate widened, bit-exact vs
  fallback (max|diff|=0.0, top16 identical, K=3712/16389/65536 both
  dtypes). LIVE @16.8k: 19.55 -> 20.94 tok/s (+7%), fused_hit=57/57,
  branch state healthy (BatchKVCache zeropad=True maskless compact ok).
  Also VERIFIED: A2/A11/A14 (chunked prefill preserves flag+class),
  A6 (sdpa256 inert for M3), H3 step-loop has no per-token O(K) work.
  Remaining: 64k measurement (running), dense-forced crossover legs,
  residual ~6-7ms sparse-machinery fixed overhead (dispatch-bound;
  candidate: fuse selection chain + gathers).
- [x] M3 POST-FIX CURVE (census on, all branch-verified fused_hit+maskless):
  26.5 short / 24.3 @3.8k / 21.2 @9.5k / 20.9 @16.8k / 18.5 @66k steady
  (census-timestamp derived; +9% at 66k, +7% at 16.8k, +6% at 9.5k, no
  regression below 4k floor). Server-reported 16.1 @66k is depressed by
  ~5s first-64-token warmup + ~4s end-of-request prefix-store tail (block
  pool grow + 259-block store; 0.25s at 9.5k) — separate phenomena, noted.
  Residual slope 16.8k->66k = 0.127ms/1k (physics ~0.04); residual cliff
  ~+6ms fixed sparse-machinery overhead vs dense — dispatch-bound,
  candidate fix: fuse selection chain (argpartition+sort+positions) and
  K/V gathers. Dense-forced 2-leg crossover measurement RUNNING.
- [x] GLM SAME-DISEASE FIX (2026-07-03): GLM stores/runs bf16 (safetensors
  BF16, no torch_dtype) -> fp16-gated fused_decode_indexer_scores NEVER
  engaged live either (the shipped +2-4% came from non-gated parts;
  flash_decode_sparse_mla has NO caller — dead code). FIX: template-ized
  kernel (T in {half,bfloat}), output/w in model dtype, gate widened,
  [GLM-DKO] ENGAGED/BAIL one-shot logs added. Standalone: bf16
  top2048_overlap 2048/2048 vs exact MLX chain (K=2049 & 58368).
  LIVE GLM A/B pending (load GLM alongside M3, short + 58k).
  NOTE: entire GLM "measured-negative" standalone catalog now carries a
  dtype asterisk (probes ran fp16, live is bf16).
- [x] M3 DENSE-vs-SPARSE CROSSOVER MEASURED (A5 resolved): dense-forced
  (SPARSE_MIN_K=1e9) legs: 19.88 @16k (vs sparse 20.94, sparse +5%),
  13.10 @69k (vs sparse 18.5, sparse +41%). Dense decode grows ~0.72ms/1k
  live (3.5x naive KV physics — mechanism q for MLX research). Dense also
  loses MSA prefill: TTFT 365s @69k vs 213s. VERDICT: SPARSE_MIN_K=4096
  default is correct; MSA sparse vindicated post-kernel-fix (the fp16
  gate was starving the architecture, not the architecture being slow).
  Post-fix extrapolation (slope 0.127ms/1k): ~16.1 @128k, ~14.3 @192k,
  ~12.8 @256k (old painful table: 14.6/12.7/11.3).
- [x] A18 CLOSED (restored-prefix path healthy): same 16k prompt twice —
  run2 TTFT 4.0s (prefix restored), decode 21.4 vs fresh 21.7, census
  run2 = MiniMaxM3BatchKVCache zeropad=True fused_hit (sparse kept).
  Plan item C (SSD-restore guard) motivating fear REFUTED empirically;
  the 16:11 "KVCache sparse=False" line was the layer-0 instrument
  artifact. A9 (thermal) further refuted: decode after 4s prefill ==
  decode after 61s prefill.
- [~] GLM LIVE w/ FIXED KERNEL (2026-07-03): short 21.5 (ledger 21.7 — no
  regression, no gain expected at tiny K); 19.5 @14.5k / 18.1 @57.7k both
  on RESTORED prefixes (TTFT 2.9s/5.1s). [GLM-DKO] ENGAGED bf16 fired in
  the short bench, BUT the fused call is gated on `mask is None`
  (deepseek_v32.py:271) — restored/batch decode likely carries a bool
  mask -> silent skip at long ctx, no log (the BAIL log lives inside the
  kernel fn, not at the upstream gate). THIRD instance of the pattern:
  fast-path gate upstream of instrumentation. FOLLOW-UP: census GLM's
  indexer call site (mask None vs array per step at 58k) + consider
  extending the fused path to bool-mask decode (mirror M3's
  original_mask take_along_axis validity trick).
- [ ] M3 NEXT LEVERS (Codex gpt-5.5 xhigh audit, line-referenced, ranked):
  (1) fused topk+sort+positions kernel appended to fused_block_scores
      (fuse NaN clean, init/local forcing, top-16, sort, position expand;
      language.py:946-997+1235-1241) — est 2-3ms/token, medium risk;
  (2) pack index_q_proj+index_k_proj into one qmv (~0.5ms, low risk;
      full QKV+index pack ~1-1.5ms but sharding code at :2034-2054);
  (3) mx.compile the elementwise chains in _build_sparse_decode_indices
      (MLX does NOT auto-fuse without compile — docs confirmed); hard
      boundaries remain argpartition/sort/gather/rope/rms/sdpa;
  (4) gather+SDPA flash kernel (skip compact K/V materialization)
      est 1.5-2.5ms, high risk;
  (5) sort-drop toggle (maskless path is permutation-invariant) —
      possible locality regression, keep env-gated.
  Scheduler step loop confirmed clean per-token (only 256-materialize +
  1024-allocator-clear spikes). Dispatch math sanity-confirmed
  (~8.6us/dispatch incl. lazy-graph+encode).
- [x] MLX INTERNALS (Codex gpt-5.5 xhigh, source-cited vs v0.31.2):
  (1) DENSE 3.5x MULTIPLIER EXPLAINED: sdpa_vector has NO GQA K/V head
  sharing — every q head streams its kv head from global independently
  (sdpa_vector.h L60-71/L217-300): source-level reads = n_q*K*(Dq+Dv)*s
  = 16x unique bytes at GQA-16; Apple GPU cache absorbs to measured
  ~3.5x. 2-pass engages K>=1024, blocks 128/512/1024 by K (no knob in
  0.31.2; upstream PR #3455 adds MLX_SDPA_BLOCKS). PR #1597 = occupancy.
  => M3 residual slope now closes to first order: 3 full-attn layers at
  the multiplier (~0.036ms/1k) + idx-key reads + gather locality.
  (2) COMMAND BUFFERS: arch-d default commits every 50 ops / 50MB
  (device.cpp L498-522); ~1000-2000 kernels/token = 20-40 commits.
  Env MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER — zero-code
  dispatch-overhead experiment (=500 RUNNING).
  (3) mx.compile(shapeless=True) matches on count/ndim/dtype — viable
  for elementwise chains w/ stable rank; python-int offset slicing
  can retrace. Lazy eval does NOT fuse without compile (compile.cpp).
  (4) slice_update donation: checked at EVAL time; requires
  use_count==1 on desc+data — a prior step's live view forces a FULL
  buffer copy (indexing.cpp L725-759, array.h L293-296). Not currently
  biting (slope closes without it); the 256-token materialization is
  the existing mitigation.
  Issues: #735 #1597 #3455 #3026 #3340 #2711 #3794 #1325.
- [x] MLX_MAX_OPS_PER_BUFFER=500 experiment: 26.8 short / 21.6 @9.5k /
  21.1 @16k (vs 26.5/21.2/20.9) = +~1% consistent, free, lossless,
  helps all models -> KEPT in standard launch. Confirms commit-boundary
  overhead is minor; per-kernel encode dominates the +6ms sparse tax ->
  fusion kernel (topk+sort+positions) remains the lever (+2-3ms est).
  FINAL PRODUCTION STATE: server env = OMLX_M3_DEBUG_PATH=256 (census
  stays on for permanent engagement visibility) + MLX_MAX_OPS_PER_BUFFER
  =500. M3 verified curve: 26.8 short / 24.3 @3.8k / 21.6 @9.5k /
  21.1 @16.8k / 18.5 @66k (all census-verified fused_hit+maskless).
- [x] FUSED TOPK KERNEL SHIPPED (2026-07-03): m3_fused_index_topk — one
  dispatch replacing NaN-where + init/local forcing chain + argpartition
  + ones + sort (per layer). Taken-bitmap guards degenerate <16-valid
  case (NaN-heavy tiny nb re-picked extracted -inf slots -> duplicate
  indices; caught standalone, live-impossible but contract kept).
  Verified 30/30 vs exact mx chain (sorted, cur-block forced, sets
  identical, both init/local configs, NaN trials). Splice: fused_sel
  early-return in _build_sparse_decode_indices; topk_valid=None signals
  sorted+all-valid downstream (sort skipped; ones-on-demand in masked
  branch). Kill switch OMLX_M3_DISABLE_FUSED_TOPK; B>1/lse/nb<=16 fall
  back. LIVE: census fused_topk=57/57; short 26.98 (nc) / 21.88 @9.5k
  (+0.28) / 21.52 @16k (+0.42 = -0.9ms/tok). Below +2-3ms estimate:
  several replaced ops were metadata-only, ~5 real dispatches saved.
  Fixed sparse tax now ~5ms. 66k capstone RUNNING. Next: pack
  index_q/k projections (+0.5ms est), gather+SDPA fusion (risky).
- [x] M3 FINAL VERIFIED CURVE (2026-07-03, census-verified, both fused
  kernels + MLX_MAX_OPS_PER_BUFFER=500):
  27.0 short / 24.3 @3.8k / 21.9 @9.5k / 21.5 @16k / 18.8 @66k steady
  (day started: 26.5 / 24.3 / 20.0 / 19.6 / 17.0 => +10% across the
  sparse band). Prefill 342 tok/s. Card reference: 21.7 / 214.
  Slope now 0.133ms/1k -> est ~16.3 @128k, ~12.8 @256k.
  66k fused-topk delta landed exactly on the K-independent prediction
  (18.5+0.9ms => 18.8 measured) — assumption verified.
  Remaining levers (diminishing): index-proj packing ~+0.5ms,
  gather+SDPA flash fusion ~+1.5-2.5ms (high risk), GLM upstream mask
  gate census. All shipped work env-killable; census permanently on.
- [x] PACKED PROJECTIONS SHIPPED (2026-07-03): q/k/v/index_q/index_k ->
  ONE quantized_matmul + reshape[B,L,77,128] + head-slice views (no
  copies; all five outputs 128-dim multiples). Lazy per-layer build with
  spec checks (bits/group/mode equal, no bias, QuantizedLinear exact
  type); tiers full(56)/qkv(3+1... census: pack_full=56 pack_qkv=3
  pack_none=1 — matches config prediction exactly). Bit-exact verified
  standalone at 8-bit AND 5-bit (q/k/v/iq/ik all max|diff|=0).
  Kill switch OMLX_M3_DISABLE_PACKED_PROJ. Cost: ~+3-4GB resident
  (originals kept for fallback/state).
  LIVE: 27.53 short (+2%) / 22.50 @9.5k (+2.8%) / 22.02 @16k (+2.3%).
  Cumulative today @16k: 19.6 -> 22.0 (+12%).
- [x] CAPSTONE 66k ALL-OPTS: 19.12 tok/s client-side, FLAT quartiles
  (vs 17.0 pre-campaign = +12.5%). GLM legs: 21.6 short / 19.7 @16k /
  18.3 @58k (no regression; +0.2 at depth within noise). GLM-DKO census:
  mask_none=True + ENGAGED on fresh decode; 58k restored-path engagement
  inconclusive (one-shot consumed by short bench) — parked, low stakes.
- [~] FLASH SPARSE SDPA ATTEMPT (last lever, high risk): kernel consuming
  sorted block ids, in-kernel positions, online-softmax over 2048
  selected keys, no compact K/V materialization. Acceptance bar:
  standalone parity vs maskless python path, live 16k A/B > +0.3 tok/s
  else REVERT. Kill switch OMLX_M3_DISABLE_FLASH_SPARSE.
- [x] FLASH SPARSE SDPA: MEASURED-NEGATIVE, kept opt-in. Kernel built
  (online-softmax over 2048 selected keys, in-kernel positions, no
  gathers), parity-verified (rel<5e-3 bf16 = output-dtype rounding,
  K=4k/16k/66k both dtypes). Standalone: 0.33-0.38ms vs mx chain 0.29ms
  @16k — 64 TGs (1/q-head) under-occupy the 80-core M3 Ultra; per-key
  simd_sum latency-bound; 4x unroll didn't help. Matches the GLM
  flash-decode graveyard (now with dtype asterisk removed: this one WAS
  tested in bf16). Would need a 2-pass split-K rebuild (512 TGs + merge)
  to compete — diminishing vs ~1ms ceiling. Gate flipped to opt-in
  (OMLX_M3_ENABLE_FLASH_SPARSE=1), default path unchanged.
- [x] NIAH LADDER (2026-07-03, stopped at 128k per user): multi-needle
  (3 @25/50/75% depth, varied-sentence haystack, temp 0) — 12/12 PERFECT:
  3/3 @13.1k / 3/3 @25.9k / 3/3 @51.5k / 3/3 @102.7k. Perf at depth:
  prefill ~316-331 tok/s FLAT to 100k+; decode steady 22.7/21.9/19.8/
  18.7 — decay is SUB-LINEAR (18.7 measured @102.7k vs 17.2 linear-fit
  prediction; old extrapolations pessimistic). fused_topk bumped to
  nb<=4096 (512k-capable, verified) before the run. Harness:
  ~/.claude/jobs/62f9cfe9/tmp/niah_bench.py (LENGTHS list; 256k/512k
  stages unrun). NEXT candidates (user asked re reasoning benchmarks):
  BABILong subsample (qa1-5 @64k/128k), LongBench v2 medium split
  sample, RULER reduced — all prefill-bound ~5.5min/128k-item.
- [x] LONG-CTX REASONING SMOKE (2026-07-03): 6 BABILong/RULER-style task
  families (object-chain 4-hop, aggregation+arith, variable-chain 3-hop,
  latest-state recency, transitive, temporal-order) x {26k, 51k} tokens,
  facts scattered 20-85% depth, temp 0. RESULT: 12/12 semantically
  correct (auto-score 10/12 — both "fails" = scorer artifact: stale-value
  must_not check reads the model's own reasoning trace which quotes the
  values it rejects; verified correct by inspection; fix = score final
  line only). Reasoning traces show clean fact citation + explicit
  chains at both depths. Harness: ~/.claude/jobs/62f9cfe9/tmp/
  quick_reason.py. Full-suite candidates (BABILong subsample /
  LongBench v2 medium) remain on offer.
- [x] REASONING @103k VERIFIED (user challenged the "reasoning to 100k+"
  claim — it HAD only been tested to 51k; overclaim corrected, then
  tested): all 6 task families PASS at ~102.6k actual tokens (object-
  chain, aggregation, variable-chain, latest-state w/ fixed final-line
  scorer, transitive, temporal-order). 6/6 auto-scored. TTFT ~320-370s
  (~310 tok/s prefill at 103k). CLAIM NOW MEASURED: M3 does clean
  retrieval (12/12) AND multi-hop reasoning (18/18 semantic across
  26k/51k/103k) to 100k+ on this box. Also answered: TurboQuant KV =
  runtime quantized cache, M3 hard-excluded (scheduler.py:2758);
  mxfp8-KV-for-M3 = RAM play but ~0.7ms/tok SLOWER as bolt-on
  (dispatch-bound compact path); speed-positive variant = int8 index
  keys; zero-risk variant = quantize M3 SSD blocks only. oQ4 recipe
  documented: experts 4b/g64 (the whole cut), attn 8b early/5b late,
  routers+norms bf16. Adaptive thinking supported (template
  enabled/disabled/adaptive + omlx enable_thinking/budget/preserve).
- [x] M3 ADAPTIVE THINKING VERIFIED LIVE (2026-07-03): template default
  branch = thinking_mode "adaptive" (jinja line 84; variable is
  thinking_mode STRING, not enable_thinking bool). Measured modulation:
  105 reasoning chars on "2+2" vs 843 on a rate word problem (correct
  answer); thinking_mode=disabled -> 0 chars; =enabled -> thinks.
  Per-request lever: chat_template_kwargs {"thinking_mode": ...}.
  BUGS FOUND: (1) omlx ms.enable_thinking emits bool `enable_thinking`
  -> template never reads it = silent no-op for M3 (fix: translate to
  thinking_mode when template uses it); (2) parser edge: max_tokens
  truncation mid-think leaks the unterminated block into content
  instead of reasoning_content (harmless w/ sane limits, note only).
- [ ] NVFP4-repack pre-flights QUEUED for idle box: (1) nvfp4/mxfp8
  qmv+gather_qmm M=1 microbench vs affine; (2) single-tensor HTTP-range
  bit-exactness proof of ModelOpt->MLX repack (mxfp8 byte-identical;
  nvfp4 = byte codes + E4M3 scales + tensor_scale carried exactly:
  fold into routing weights for down_proj, one post-mul for gate_up).
  Fused-switch decision: converter with BOTH flags (unfuse-shared
  lossless ~-3% vs shared->nvfp4 requant keep-fusion) -> eval decides.
- [x] enable_thinking TRANSLATION PATCH (2026-07-03): server.py
  _translate_thinking_kwargs — when the template reads string
  thinking_mode (MiniMax M3) and only boolean enable_thinking was
  provided, translate True/False -> "enabled"/"disabled" in place;
  explicit thinking_mode always wins; enable_thinking-style templates
  (Qwen/Gemma) untouched. Wired at ALL THREE ct_kwargs consumption
  sites (3387/5124/5646). Unit-tested against the real M3 jinja + a
  Qwen-style template (6 cases incl. no-tokenizer safety). ARMS ON NEXT
  RESTART (not restarted — user's live agent session on the box).
- [x] reasoning_effort MAPPING (2026-07-03): added top-level
  reasoning_effort to ChatCompletionRequest (+completions parity) — was
  previously DROPPED by pydantic (Hermes agent's levels were no-ops).
  New _apply_reasoning_effort in server.py: capability-sniffs the chat
  template and maps none/minimal/low/medium/high onto the richest
  native control: GLM-5.2 native reasoning_effort (High/Max grades +
  enable_thinking off), M3 thinking_mode (disabled/adaptive/enabled),
  bool enable_thinking (Qwen/Gemma), or budget-only (DSv4 R1-style).
  Budgets: none=128/minimal=512/low=2048/medium=8192/high=uncapped via
  existing thinking_budget enforcement; explicit kwargs/budget win.
  Wired at all 3 ct_kwargs sites BEFORE _translate_thinking_kwargs
  (hoisted above the empty-dict guard — bare requests map too).
  Unit-tested vs all three real templates x 5 levels + precedence.
  ARMS ON NEXT RESTART (live agent session, no bounce).
- [x] reasoning_effort mapping REVISED per user (2026-07-03): NO implied
  thinking budgets — force-closing a think mid-trace corrupts the trace
  and nukes answer quality; budgets stay strictly explicit. Mapping is
  template-knobs only: GLM none->off, min/low/med->High, high->Max;
  M3 none->disabled, min/low/med->adaptive (model self-modulates),
  high->enabled; bool templates on/off; DSv4 (no knob) -> honest no-op.
  Verified: zero budget leakage across all levels x 3 templates +
  explicit-kwarg and explicit-budget precedence.
- [x] HERMES PROTOCOL FINAL (2026-07-03): top-level enable_thinking
  (vLLM-style toggle) + reasoning_effort levels {minimal,low,medium,
  high,max} (+none) both added to request schema and composed:
  off-toggle wins outright (no level resurrects thinking); on/absent +
  level steers via native knobs (GLM min/low/med->High, high/max->Max;
  M3 min/low/med->adaptive, high/max->enabled, bool superseded by
  thinking_mode; bool templates toggle-only; DSv4 no-op). No implied
  budgets. Toggle wired into merged_ct_kwargs at both merge blocks
  (overrides model settings, yields to explicit request ct_kwargs).
  Verified: 5 levels x 4 templates x {on,off,absent} + explicit-knob
  precedence. Arms on next restart.
- [x] RESTART + LIVE VERIFICATION (2026-07-03 night): thinking stack
  armed and proven end-to-end. M3: effort none->0 reasoning, low/max
  think, toggle-off beats effort=max (0 reasoning). GLM: none/off -> no
  think, low -> native High grade thinks (rc=340), max -> Max grade.
  PRE-FLIGHT 1 PASSED: nvfp4/mxfp8 M=1 kernel parity with affine
  (qmv: nvfp4 261us vs affine4 269; mxfp8 273 vs affine8 301 — mxfp8
  FASTER; gather_qmm: nvfp4 232 vs affine4 227). NVFP4 repack track
  unblocked on speed; remaining gate = pre-flight 2 (single-tensor
  HTTP-range byte-layout proof), then converter + 250GB download.
- [x] PRE-FLIGHT 2 PASSED (2026-07-04): NVIDIA NVFP4 -> MLX is a PURE
  BYTE-REPACK. Range-fetched real expert triplet (layers.20 experts.0.w1:
  U8 [3072,3072] codes + F8_E4M3 [3072,384] per-16 scales + f32
  weight_scale_2=1.16e-4 + input_scale): bytes viewed directly as MLX
  nvfp4 state dequantize bit-identical to spec math (max|diff|=0.0,
  lo-nibble-first, scale bytes as-is). input_scale presence proves
  activation-aware calibration (weights inherit it; MLX ignores act
  scales). Attention/index use weight_scale_inv naming = oQ's EXISTING
  _scale_inv pairing + _block_dequant_fp8 path (E8M0 already handled).
  BOTH pre-flights green => oQ mod fully de-risked:
  (1) oQ "repack-modelopt" mode: nvfp4 byte-copy + mxfp8 via existing
  fp8 path; (2) per-expert weight_scale_2 sidecar + M3 MoE runtime
  carry (routing-weight fold for w2, post-mul for w1/w3); (3) shared
  expert: unfuse flag (lossless) vs requant-to-nvfp4 (keep fusion) —
  both, eval decides. Remaining: the build (~half day) + 250GB
  download + PPL/smoke gate vs MXFP8 reference (optional 442GB).
- [x] MXFP8 BYTE-REPACK PROVEN TOO (2026-07-04): index_k_proj real bytes
  (F8_E4M3 [128,6144] + U8 E8M0 [128,192] per-32) viewed directly as MLX
  mxfp8 state: max|diff|=0.0 vs spec math. BOTH formats byte-copy.
  Shard-9 header facts: attn/index/shared = mxfp8 + weight_scale_inv
  (E8M0 u8); routers F32 unquantized; shared experts separate
  gate/up/down. NVFP4 download RUNNING -> $OMLX_COLD_STORAGE/
  omlx-quant-work/ (T7 USB 4TB, 1.7T free; BACKUP DRIVE — only our new
  folder touched). NEXT: oQ repack-modelopt mode build.
- [x] oQ-NVFP4 GOAL MET (2026-07-04): production variant
  MiniMax-M3-oQNVFP4-fs5 (fuse-shared-nvfp4 + attn5-layers 17-44).
  SPEED ~same: 27.09/21.66/21.64 vs oQ4 27.53/22.50/22.02 (-1.6/-3.7/
  -1.7%), 18.0 vs 18.7 @103k. QUALITY better: gsm8k 95.3 vs 92.7
  (+2.7pp), mmlu 81.3 vs 79.3 (+2.0pp); default variant measured arc
  96.7 vs 94.3 (+2.3pp) and mmlu 83.0. LONG-CTX: 6/6 reasoning @103k,
  NIAH 3/3 @103k. Variant ladder: default 23.7 short (attn8+separate
  shared+ts) -> fs 25.75 -> fs5 27.09. Converter grew --attn5-layers
  (oQ4 sensitivity mirror). Settings entry added for fs5.
  CLEANUP CANDIDATES: MiniMax-M3-oQNVFP4 (248G) + -fs (246G) variants
  deletable once user blesses fs5 (~494GB reclaim). Admin-API accuracy
  harness path blocked by session auth (classifier: no credential
  extraction) -> public-API acc_bench.py used instead (identical
  measurement). oQ4 leg wedge: dual 260+240GB load stalled server ->
  restart + solo legs (lesson: sequential model swaps for A/B on this
  RAM budget).

## M3 fs5 DECODE SPEEDUP FORAY (2026-07-04) — levers A/B/C
Goal: fs5 decode >= oQ4. Baselines (fs5 live): 27.09 short / 21.66 @9.5k /
21.64 @16k. oQ4 ref: 27.53 / 22.50 / 22.02. Live dtype bf16; fs5=fused mode
(129-expert switch, shared@128), nvfp4_ts=True; swiglu a=1.702 L=7.0 b=1.0.
Live module = vendored language.py (verified via compat patch). Sparse-attn
patch wraps _build_sparse_decode_indices (passthrough return) + _sparse_decode
_attention (**kwargs forward) — 3-tuple + new kwarg both survive.
- [x] LEVER A: fused m3_swiglu_oai_ts kernel LANDED + LIVE-VERIFIED.
  Parity rel=0 real code path (rel<=8.7e-4 direct large-R). Census
  swiglu_ts_fused=57/step, standalone=0 (no fallback). Kill
  OMLX_M3_DISABLE_FUSED_SWIGLU_TS verified (flips to standalone). oQ4
  correctly NO ts engagement.
- [x] LEVER B: topk kernel positions 2nd-output LANDED + LIVE-VERIFIED.
  Bit-identical vs mx ALL edge cases (nb 17/100/256/4095/4096 x tail
  pads). Census fused_positions==fused_topk==compact_ok (no fallback).
  Kill OMLX_M3_DISABLE_FUSED_POSITIONS verified (emit-flag skips kernel
  write too, per Codex finding #3). Applies to oQ4 too (harmless).
- CODEX REVIEW (read-only, gpt-5.5 xhigh): no Lever-A math issue;
  wrapper contract preserved; finding #3 (kill-switch didn't skip kernel
  write) FIXED via emit param. Findings #1/#2 = PRE-EXISTING team WIP
  (mask-before-topk in batched/left-padded compact path; prefill mask
  rewrite) — NOT introduced by A/B (levers are output-identical to
  baseline); flagged to lead, out of scope, batched-only (single-stream
  probe path uses maskless, unaffected).
- SPEED (same-session decode tok/s, single runs; prior baseline
  27.09/21.66/21.64):
  ctx    both-off  A-only(R3)  A+B(R1)   oQ4-ref
  short  27.18     27.16       27.30     27.53
  9.5k   22.00     22.14       22.26     22.50
  16k    21.71     21.55       21.91     22.02
  Combined A+B delta +0.12/+0.26/+0.20 (consistently positive). Per-lever
  split at noise floor (~+-0.15; "B" shows +0.14 at short where inactive).
  oQ4 regression: 27.89 short coherent (>=27.53 baseline) NO REGRESSION.
- [x] LEVER C (STRETCH): 2-pass split-K flash ATTEMPTED (Codex-drafted new
  file fused_flash_v2.py, 512 TGs pass1 + 64-TG merge pass2, consumes
  Lever-B positions, no gathers). Parity PASS rel 2.55-3.00e-3 (incl tail
  pos>=K mask). BUT standalone @16k: v2=285.7us vs mx gather+SDPA=264.8us
  -> v2 ~8% SLOWER. MLX native SDPA + now-SORTED (contiguous, cheap) gather
  beats the hand kernel. FAILS perf bar -> left opt-in-off
  (OMLX_M3_ENABLE_FLASH_SPARSE_V2 default OFF, NOT wired into language.py =
  inert). Measured negative, recorded. File kept for future work.
- Discipline: py_compile; standalone parity BEFORE restart; Codex seat2
  read-only adversarial review of full diff pre-deploy; restart w/
  MLX_MAX_OPS_PER_BUFFER=500 + OMLX_M3_DEBUG_PATH=256; health-gate; probe
  {4|210|360}{128|256} fresh nonce; grep M3CENSUS; A/B via kill switches;
  oQ4 regression.
- [ ] EAGLE3 (user directive): quantize the draft head to MXFP8 (mx.quantize
  mode=mxfp8 g32 — measured FASTER than affine8 at M=1: 273 vs 301us; halves
  ~1.9GB/draft-step reads => ~1ms/output-token at K=4). Gate 2 must measure
  acceptance alpha for BOTH bf16 and mxfp8 heads (offline, same run) — ship
  mxfp8 unless alpha drops >1pp. Head downloaded: $OMLX_COLD_STORAGE/
  omlx-quant-work/MiniMax-M3-EAGLE3 (6.1GB bf16, LlamaForCausalLMEagle3,
  1 layer, hidden 6144, MHA-64, vocab 200064, fc_norm+norm_output).
- [x] EAGLE GATES (2026-07-04): GATE 1 verify-ratio measured (warm 4k,
  wired, warmed-up): L=1 47.4ms, marginal +44.6 first extra token (L>1
  mode switch off compact path!) then ~17-21ms/extra -> naive EAGLE dead
  (needs a~0.85); fix identified = small-L (L<=8) compact verify path
  (extend fused score+topk kernels to L queries + union-gather + masked
  sdpa via EXISTING machinery) -> marginal ~12ms -> K=3 breakeven a~0.6.
  GATE 2 acceptance: offline harness on fs5 hiddens: taps (2,30,57) win
  grid (=vendor's), a1=0.55 w/ chain collapse = HARNESS artifacts
  (missing draft prefix-KV + "+final layer" input per README); mxfp8
  head costs only ~1pp (SHIP mxfp8). VENDOR numbers (vLLM vs MXFP8,
  K=3): code/math a=0.92/0.84/0.75, accept len ~3.5; dialogue 2.7.
  Projection on our stack w/ small-L path: ~43 tok/s code, ~34 prose.
  README: ttt_length=7, embed/lm_head/final-norm SHARED from target
  (reuse fs5's quantized head at serve = -5GB). VERDICT: build = small-L
  verify path + EAGLE3 integration (~2 days crew).

## LEDGER (deferred / post-EAGLE levers)
- [DEFER] M3 verify-path review finding #2: __call__ rewrites any L>1 mx.array mask to "causal"
  (language.py ~1622-1632), a PRE-EXISTING assumption shared by the MSA-prefill path. The new
  small-L verify path inherits it identically (oracle-consistent, no regression). Hardening it
  would touch MSA-prefill too — out of scope. Revisit only if a B==1 L>1 physical padding mask
  is ever produced. (Signed off by lead 2026-07-04.)
- [POST-EAGLE] MoE verify-batching: the L>=4 decode marginal is MoE-bandwidth-bound (~14.6ms/extra
  floor with attention=zero); the <=14ms/extra bar was retired. Batching verify tokens' MoE
  expert-weight reads is the lever to push below the floor. Separate effort.
- [ ] SIDEQUEST (user): QuixiAI/ThunderMittens (Hartford fork of Hazy's
  Metal ThunderKittens, active) — dense-fp tile kernels claiming 3.9x
  causal attn / 3x norms vs mx.fast on M4 Max. NOT our hot path (no
  quantized matmul; our qmv is 83-105% bw; decode shapes kill norm wins).
  ONE experiment worth doing: their causal attention at OUR MSA-prefill
  shapes (4096-chunk, GQA-4) vs mx SDPA — prefill attn is a minor slice
  (MoE ~89%) so ceiling is single-digit %. ALSO check QuixiAI/ds4
  ("DeepSeek 4 Flash local inference engine for Metal") for ideas.

## EAGLE-3 (Phase 2) — CLOSE-OUT LEDGER (2026-07-04)
- SHIPPED: MiniMax-M3 EAGLE-3 spec-decode path (reuses mlx-vlm Eagle3 + omlx adapter eagle3_minimax.py + vlm_mtp glue).
  Correct (prefix-KV seeded, argmax-exact greedy), accepts well: math a1/a2/a3=0.95-0.97/0.92-0.93/0.85 (BEATS vendor),
  code 0.84-0.87/0.65/0.50-0.53, mean accept len 3.0 code / 2.7-3.7 math.
- mxfp8 drafter: SHIPS as default drafter precision — accept delta <1pp vs bf16 (math even +0.03 len), cheaper matmuls. Env OMLX_EAGLE3_MXFP8.
- v1 CONSTRAINT: greedy-only (temp==0 engagement guard). Sampled requests (fs5 default force_sampling+temp=1.0) → spec OFF (no regression). Ship = per-request opt-in.
- PERF VERDICT: ~break-even on M3 (live: 26.2short/19.0@16k vs baseline 27.1/21.6). NOT a speedup. Root: verify L=4 forward = 85% of round, MoE-weight-read-bound (per-token expert reads, no amortization) — same wall as retired Phase-1 <=14ms bar. Draft only 15%; draft-side opt caps ~+5-8% short/~0% 16k.
- FOLLOW-UPS: [2b] speculative rejection sampling → unlock temp>0 (lifts the greedy-only constraint). [POST-EAGLE] MoE verify-batching (see earlier ledger) is the only lever past the wall. Skipped draft epilogue kernel (lm_head=bandwidth, fusion can't shrink bytes — profile-justified).
- OUTSTANDING robustness (before wider/non-M3 use): 128k tap-buffer bound; prefix-cache-hit persists taps (else re-prefill fallback). Both avoided in short-ctx greedy opt-in v1.
- [x] K-SWEEP CLOSES EAGLE QUESTION (2026-07-04, user-prompted): model
  predicted K=1/+10-16%, K=2/+5-20%, K=10/-33-50%. LIVE K=2: math 26.63,
  code 26.40 vs ~26.9 baseline = break-even EVEN ON MATH. Back-solved:
  true cost per speculated token ~35ms ≈ 0.94x base token -> profit
  requires alpha~1. No K pays on M3. Definitive: memory-bound MoE
  verify, not drafter quality (ours beats vendor) nor K policy.
  Settings left: block_size=3 (K=2, least-bad), opt-in + temp guard.
- [x] THUNDERMITTENS SIDEQUEST CLOSED (2026-07-04): VERDICT IGNORE.
  Scout ported their nvfp4 qgemv verbatim to mx.fast.metal_kernel
  (their native build = vendored MLX 0.21 fork, incompatible) and
  benched at our shapes: 1.4-2.5x SLOWER than MLX qmv (322 vs 789GB/s
  at 8192x6144 — MLX runs 96% of ceiling, reconfirming our ledger).
  Their "beats fp16 GEMV" claim = vs DEQUANTIZED mx.matmul; their fork
  predates MLX nvfp4 entirely. No indexed/MoE quant GEMV exists there;
  paged-decode sparse partition = DeepSeek-MLA-specific; nothing
  transfers. RE-VISIT TRIGGER: upstream ships indexed quant GEMV at
  >65% BW. Report: scratchpad/tm_scout/tm_scout_report.md.
- [x] K=1 DIRECT MEASUREMENT (final EAGLE datapoint): math 26.65 / code
  26.59 vs ~26.9 baseline = break-even. Complete measured K-curve
  {1,2,3} flat within +-2% both content types => exchange rate ~1:1
  (verify-a-token ~= generate-a-token on memory-bound MoE). EAGLE-3 on
  M3 = latency-neutral capability; becomes valuable iff the exchange
  rate changes (MoE verify-batching / upstream gather amortization /
  smaller experts). Settings left at block_size=2 (K=1), opt-in,
  temp-guarded.
- [x] TM CAPABILITY MAP (final, scratchpad/tm_scout/tm_capability_map.md):
  dependency NO (dead MLX-0.21 fork, fp16-act, no root LICENSE — check
  upstream terms before any verbatim lift). REFERENCE yes, 3 finds:
  (1) fp8 MLA-KV complete reference (insert + partitioned dequant-on-
  read decode, e4m3+UE8M0/64, 1.25x penalty) => DE-RISKS our deferred
  int8-MLA-KV port (GLM 2x-context item; Kimi long-ctx fix note);
  (2) GQA K/V staging kernel structure (flashinfer-style TG staging) —
  the MLX no-sharing gap, if we ever build native GQA decode;
  (3) device-resident spec-verify pipeline (linear+tree) — contract to
  copy if tree-MTP ever lands. VALIDATION: their best sparse kernel
  independently converged on our MSA index-gather+partition design.
  Watch triggers: indexed quant GEMV / partitioned block-sparse /
  modern-MLX rebase. VLM micro-fusions (embed lookup 8x, multimodal
  spans 12x) noted if profiles ever show that glue.
- [x] SPEC-VERIFY LIT SWEEP (2026-07-04): the wall is PUBLISHED (2506.
  20675 verify 2-3x data movement; 2505.19645 batch-1 uniquely bad;
  llama.cpp Metal + RTX3090 Qwen-MoE replications) but NOT fundamental.
  ROADMAP ADOPTED (literature-backed, projected +25% code / +50-70%
  math at K=3):
  P0 measure (1d): in-block trace of per-layer expert indices + gate
    probs during L=4 verify -> union U distribution + mass curve.
  P1 lossless grouped verify: TRY mx.gather_qmm(sorted_indices=True)
    first; else segment-GEMM kernel (dequant expert tiles once, M<=4).
  P2 MoE-Spec budget (2602.16052, +16-26% over EAGLE-3 at batch1 on
    Qwen3-30B-A3B 128-expert): top-B expert shortlist per layer from
    summed gate mass, remap draft rows within it, committed row exact;
    sweep B in {8,10,12,16}; quality-gated.
  P3 Cascade utility gate (2506.20675): EMA tokens-gained/cost, auto
    K-shrink/off; caps worst case at -5%.
  P4 optional: SelfJudge relaxed acceptance once marginal cheap
    (asymptote 1.54x -> 3.5x).
  FACTS: draft-side work capped at 1.54x on our marginals (proven);
  natural adjacent-token overlap f~0.15-0.30 (insufficient alone on
  code); code routing mass concentrates (top-8/64 = 95%). MoESD:
  even 2-4 concurrent requests flips economics positive AS-IS.
- [x] P0 ROUTING MEASUREMENT (2026-07-04): P1 AND P2 BOTH NO-GO —
  definitive, 630 windows. (1) P1 no-op: mx.gather_qmm ALREADY
  auto-dedups duplicate-expert reads (time ~ 177us fixed + 36us x U
  distinct; sorted_indices=True changes nothing, 1.00x) — the natural
  overlap (code f=.36, math .43) is already realized live. (2) P2
  fails code quality gate: M3's sigmoid router is DIFFUSE — routing-
  weight mass at B=8-12 only .75-.84 code (.81-.88 math); needs ~.90.
  MoE-Spec's Qwen3/OLMoE concentration does not transfer.
  => spec-decode at batch-1 on M3: measured dead end at every level
  (K-sweep, drafter quality, verify kernels, routing dedup, expert
  budget). Standing wins: batch>=2 flips economics (free); EAGLE stays
  opt-in latency-neutral. NEW LEAD (bigger than spec): ~177us FIXED
  cost per gather_qmm call x 114 calls/token ~= 20ms/token IF real in
  the live pipelined path (CAVEAT: standalone eval-per-call sync may
  inflate it) — probe live, then gate_up+down call fusion halves calls.
  Affects BASE decode, not just verify. Instrument: OMLX_M3_ROUTE_TRACE
  env-gated in vendored language.py; raw npz ~/.claude/jobs/62f9cfe9/
  tmp/p0_route/.

## HOST-SERIALIZATION PROFILING (2026-07-04, user: "did we profile omlx for stupid inefficiencies?")
- [x] LIVE WHOLE-PROCESS PROFILE (first ever on the serving stack; /usr/bin/sample, no sudo):
  decode thread during fs5 500-tok run = 52% cond_wait INSIDE mx.async_eval,
  18% Metal command encoding, ~16% Python forward (nn.Module slot_tp_call towers),
  ~9% mlx graph-node ctor; top-of-stack dominated by malloc/free churn.
  => host ~17ms + GPU ~20ms SERIALIZED = 37ms/token = 26-27 tok/s. GPU idles ~45% of wall.
- [x] MECHANISM (verified vs MLX v0.31.2 transforms.cpp): MAX_ACTIVE_TASKS=10 is a
  compile-time constant; eval_impl blocks callers (incl. async_eval) past it.
  BatchGenerator's one-ahead async_eval exists but cannot run ahead -> no overlap.
- [x] KNOB A/Bs ALL FLAT (~26.3 tok/s each): MLX_MAX_OPS_PER_BUFFER 500->2000; MLX_METAL_FAST_SYNCH=1;
  MTP on vs off IDENTICAL (26.26 vs 26.41/25.98) — vlm_mtp bypasses BatchGenerator
  (scheduler.py:6606 _step_vlm_mtp; vlm_mtp.py:466 yields int per round, no lookahead)
  but base is equally serial in practice, so EAGLE break-even verdict stands unchanged.
- [x] CLOSES P0 CAVEAT: the "177us fixed/gather_qmm-call IF live" lead IS live — it is
  this 17ms/token host serialization, measured end-to-end.
- PRIZE: hiding/killing host 17ms => up to ~50 tok/s short-context (biggest lever found all campaign).
  Ranked directions: (1) mx.compile the decode step (kills python+ctor, shrinks encode; multi-day,
  cache-state + custom-kernel tracing to validate), (2) gate_up+down gather_qmm call fusion
  (halves 114 MoE calls -> less encode+churn; bounded), (3) patch MLX MAX_ACTIVE_TASKS + rebuild
  (fast to try, wheel-rebuild risk vs pinned kernel ABI).
- Server restored to production after A/Bs: MTP ON, OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=500,
  fs5 loaded + drafter attached, warm probe healthy. model_settings.json byte-restored (vlm_mtp_enabled=true).
- Artifacts: jobs tmp/ m3_sample.txt (MTP), m3_sample_base2.txt (base), server_ops2000.log, server_fastsynch.log.
- [x] CODEX HOT-PATH SWEEP (2026-07-04, report: 8cafa811 scratchpad/codex_overhead_out.md):
  scheduler per-token path CLEAN (only 256-tok cache materialization + 1024-resp allocator
  cleanup spikes; grammar syncs inactive). Confirms decode path has NO mx.compile (only a mask
  helper at language.py:112). Already-shipped items independently re-derived (fused topk+positions,
  5-way projection pack). NEW candidates: [A1] sparse-entry threshold autotune vs real crossover
  (OMLX_M3_SPARSE_MIN_K=4096, language.py:1425; dense is exact => quality up, low risk),
  [B3] gather+SDPA fused decode kernel skipping compact K/V writes (~1.5-2.5ms, high risk —
  the old "2-pass flash" thread), [A2] verify mx.sort(topk_idx) skip engages live when fused
  kernel active. Sanity: ~8.6us/dispatch host cost — matches the 17ms/token serialization finding.
- [x] GLM HOST/GPU SPLIT MEASURED (2026-07-04, user asked if levers 1/2 help GLM):
  GLM-5.2-Alis 20.8 tok/s = 48ms/token: 69% GPU wait (~33ms), 16.5% encode (~8ms),
  ~15% python (~7ms). Same MAX_ACTIVE_TASKS serialization; host ~15ms on top of GPU.
  => mx.compile / MLX task-cap patch project GLM 21 -> ~30 tok/s (+40-45%). CONFIRMED GENERAL.
  Sample: jobs tmp/glm_sample.txt.
- [x] KV-CACHE QUESTION SETTLED (int8 vs mxfp8): int8 MLA-KV was BUILT+VALIDATED (token-identical,
  42% smaller) then ABANDONED 2026-06-25 — 2x slower prefill from 5-clamp scheduler stack; fp16
  MLA-KV fits even 1M ctx on 512GB (90GB). q8 kernel remains committed capability. mxfp8 latent
  would hit the SAME dequant-on-read wall (no fp8 MMA on Apple GPUs, cache never reaches quantized
  SDPA) with zero quality upside over the already-token-identical int8. VERDICT: test neither on
  this box; revisit only for >1M ctx or smaller machines, reusing the q8 capability.
- [x] MXFP8 KV FOR M3 SCOPED (2026-07-04): feasible unlike GLM (per-token GQA rows, 128-tok MSA
  block alignment, decode dequant bounded to gathered 2048 rows/layer) BUT low value: 4 KV heads
  => fp16 KV is tiny (123KB/tok + 58KB index = 1M ctx fits in ~418GB total TODAY). 8-bit saves
  ~90GB at 1M and ~0.3ms/tok decode (noise). Verdict: capacity feature (concurrent long requests),
  NOT a speed lever; rank below overlap levers 1-3. Do with per-block quant + fp16 recent window
  if ever needed; prefill throttle clamps (GLM lesson) are the risk to re-check.

- [x] EAGLE-TEMP1 CAMPAIGN (2026-07-05, user: "engage" after artifact recheck): CLOSED NO-GO with
  mechanism. User's skepticism found the force_sampling artifact (drafter NEVER engaged in any prior
  server bench); real live acceptance 83-87% (vendor-grade); leg-C discriminator proved the wall =
  MoE expert divergence (verify L=K+1 reads ~(K+1)x expert weights -> spec capped at break-even at
  batch-1, ANY temperature; wrapper slack only ~14%). Two hypotheses retracted via pre-agreed
  discriminators before funding builds. Full analysis: tasks/eagle_temp1.md; lessons.md amended.
  Assets banked for BATCH>=2 campaign: compiled decode (L1), cap wheel, vendor-grade drafter,
  validated alpha harness (jobs tmp eagle_gate0_alpha.py + gate0_ops.sh, never run, auth-blocked).
  Source touch: omlx/speculative/vlm_mtp.py only (env-gated profiler + CLEAR_EVERY knob, defaults
  preserve behavior). Production restored + verified throughout.

- [x] GLM-5.2 OPTIMIZATION SCOUT (2026-07-05, user: "do these!"): three questions answered in one pass.
  (1) GLM base residual @ golden: 23.1 tok/s = 43.3ms/token, 53.6% GPU-wait + 17.5% encode + 12.5% py
  => ~10-12ms non-GPU on critical path, but ALL cheap levers triple-proven dead (M3 cap A/B, M3 compile
  A/B, GLM foray compile 1.06x). Only fused-layer/mega-kernel touches it. (2) Indexer fp8 keys: GLM
  indexer K = 1 small vector/token => O(K) scan ~1.4ms @58k, fp8 saves <1ms => SKIP at real depths.
  (3) GLM MTP: pre-answered by GLM52_MTP_FORAY.md (0.85x, lossless, diffuse ~32ms L=2 overhead in opaque
  kernels, compile tested 1.06x dead, only months-class fused decoder-layer kernel crosses 1.3x) —
  independently confirms the M3 EAGLE verdict cross-model. BONUS: solved the foray's 65x standalone
  mystery (missing mx.set_wired_limit => page-fault storm; Gate-1 hit the same). Foray doc updated with
  §11 cross-campaign addenda. MECHANISM REFINEMENT for the record: GLM's identical-token control shows
  diffuse per-op overhead dominates over expert divergence; M3's split unmeasured (control never run) —
  eagle_temp1.md mechanism wording stands but with this caveat noted here.
  GLM FUNDABLE LIST (final): mega-kernel/fused-layer campaign (helps L=1 base AND unlocks MTP, both
  models, months-class); batch>=2. Everything else measured dead or <1ms.

- [x] oQNVFP4 NEMOTRON PATHFINDER (2026-07-05, user: "continue the campaign... start with the super"):
  Super-120B NVFP4 byte-repacked LOSSLESSLY to MLX (79.3GB->77.5 discovered). Evidence chain: repack
  bit-exact (max|diff|=0.0, D2), relu² ts-fold algebraically exact (fp32 sidecars folded into router
  scores; positivity asserted at convert), fp8 mixers via E4M3 LUT within one bf16 rounding, D4 real-path
  load + live serving. VERDICT: quality TIED vs oQ4e (gsm8k 92/92 serial, mmlu 81/80, arc 95/94);
  decode 0.58x (29.4 vs 50.4 @5k) = MLX nvfp4 gather_qmm gs16 kernel overhead in Super's fast/small
  regime — NOT a correctness issue, does NOT gate Ultra (bandwidth-bound + only viable 550B route).
  NEW FILES: omlx/tools/oqnvfp4_nemotron_convert.py, omlx/patches/nemotron_h_nvfp4_ts.py, +14L hook
  in model_loading.py (flag-gated). Non-interference proven live (oQ4e loads stock).
  *** PRODUCTION BUG FOUND (separate): oQ4e SpecPrefill under 6-way CONCURRENCY collapses gsm8k
  92%->15% (serial fine). User's default model gives wrong answers under concurrent load with
  SpecPrefill enabled. NEEDS OWN INVESTIGATION. ***
  ULTRA (550B, 352GB): downloading to T7; converter deltas being prepped; conversion+gates next.

- [x] oQNVFP4 ULTRA 550B FIRST LIGHT (2026-07-05, serve window after conversion):
  CONVERSION: 357.737GB / 97 shards / 1119 tensors / 96 ts sidecars, positivity green on all 48 MoE
  layers (converter-builder, pre-compaction). D4 REAL-PATH GATES: PASS after two CONFIG-DIALECT fixes
  (Ultra ships layers_block_type w/o num_hidden_layers -> mlx_lm ModelArgs positional TypeError;
  time_step_limit=[0.0,{"__float__":"Infinity"}] tagged-dict -> mx.clip ValueError). Both fixed on-disk
  AND in converter (setdefault num_hidden_layers=len(pattern); decode/drop tagged non-finite tsl).
  Weights needed ZERO changes — pure repack held.
  SERVED (first 550B on this box): pool load 333.55GB actual (~6GB/s off internal SSD, 57s TTFT incl
  load). SPEED (stream-free T1/T256, warmed): decode 8.0 tok/s flat short->5k; prefill ~104-160 tok/s
  @6k, ~143 tok/s @20k. Client-stream "48 tok/s @5k" was an ARTIFACT (see stream bug below).
  QUALITY (serial, non-stream, temp0): gsm8k 91.7 (n=60), mmlu 85.3 (n=150), arc 96.7 (n=150)
  vs Super-oQNVFP4 92/81/95 -> mmlu +4.3pp, arc +1.7pp, gsm8k tie = 550B lift visible, NO quant damage.
  *** THREE ULTRA-ONLY SERVING FINDINGS (omlx stack, not the model): ***
  (1) STREAM BUG: entire answer delivered as ONE SSE chunk at completion (128 tok probe: 1 chunk
      @18.75s). Super (same arch+tokenizer) streams 8 chunks @0.134s cadence. Non-stream API fine.
  (2) PREFIX-CACHE COMMIT LAG: exact-repeat prompt misses (full 40s re-prefill); entry only usable
      ~40s after creating request completes (3rd submission hits). Super hits immediately.
  (3) THROTTLE MISPREDICTOR (benign, traced by throttle-tracer): "adaptive_prefill_throttle
      predicted=155.78GB" = one-time 120GB phys_footprint jump (expert weight wiring, chunk 0)
      charged as per-token rate x 2048 x 1.3. Prompt length never in formula (20k predicted LESS
      than 6k = EWMA decay). Cost: once/request pause + no-op eviction + first chunk 2048->1817,
      ~10s of ms TTFT. Decode UNTOUCHED (8 tok/s is native). Fix = 3-line clamp of measured term
      to K x static estimate in _predicted_chunk_transient (scheduler.py:3257-3289).
  Stream-tracer root-cause in flight; fixes (1)(3) queued for one restart window.

- [x] ULTRA SPEED CAMPAIGN Day-0+live (2026-07-05, "make the ultra run fast", fable-plan/codex-review/
  codex-impl/fable+opus-review/lead-live pipeline): 7.68 -> 13.08 tok/s (+70%), quality HELD
  (gsm8k spot 96.7% n=30 vs 91.7% n=60 baseline), resident 333.6 -> 305.1GB (-28.5GB).
  LADDER (all engagement-verified, stream-free T1/T256 probes, short==5k at every rung):
    leg0 serving fixes + sorted routes: 8.02 (speed flat; sort lever's isolated 3.5ms did NOT
      survive in-stream — lesson instance; kept, free + kill-switched)
    leg1 +DQ8 mamba 96/96: 10.38, -17.7GB | leg2 +moedense 192/192: 12.52, -26.1GB
    leg3 +attn q/o 24/24: 12.85 | leg4 +lmhead 1/1: 13.07-13.08, 305.07GB
  SERVING FIXES live: SF-1 early cache publish -> exact-repeat 5k TTFT 40s -> 3.0s (13x);
  SF-2 clamp -> zero adaptive_prefill_throttle noise. LAUNCH LINE now adds:
  OMLX_ULTRA_DQ8_MAMBA=1 OMLX_ULTRA_DQ8_MOEDENSE=1 OMLX_ULTRA_DQ8_ATTN=1 OMLX_ULTRA_DQ8_LMHEAD=1
  KEY MEASUREMENTS: M0 affine8-gs64 wins (47.2ms/token weighted; attn k/v = LOSS, excluded);
  K1 golden-env: experts 1.87x ideal but expert-specific excess only 2.41ms/token (dense control
  equal) -> fused expert kernel DEAD, P0-K parked to P2, diffuse 1.71x belongs to mega-kernel
  campaign; nvfp4-vs-affine4-vs-mxfp4 all within 4% at Ultra shapes (format fully exonerated);
  sorted-vs-unsorted BIT-IDENTICAL at real dims.
  PIPELINE CATCHES: codex review caught missing hook contract + 34GB transient + gate direction;
  fable review caught DQ8 stage B silently dead under _TsFoldMoE class swap (type-name filter,
  0==0 blind hard-fail) — REPRODUCED, fixed via block_type filters + independent census; codex
  self-caught id-reuse idempotency corruption; codex scope-creep (EAGLE-3 plumbing) kept after
  verification — it FIXES a real bug: stop STRINGS silently ignored on MTP decode (3 prod models).
  NEW FILES: patches/nemotron_h_dq8.py, patches/nemotron_ultra_decode/, tests x3 (202/202).
  FOLLOW-UPS (user decisions): bake DQ8 into offline checkpoint (~305GB disk, no load-time
  quantize pass); T3-T5 trailer tests for SF-1; commit review of the whole tree.

- [x] DQ8 CHECKPOINT PRODUCTIZATION / FULL-PIPELINE TEST (2026-07-05 night, "update the model but
  test the full conversion pipeline"): converter --dq8 (affine8-gs64 baked at convert time),
  shared DQ8_STAGES map (patch+converter import the SAME object, is-identity tested), baked
  detection (3-way classifier linear/baked/other; env vars inert on baked; corrupt still raises).
  212/212 offline tests. FULL RUN: NVFP4 source (T7) -> Nemotron-3-Ultra-oQNVFP4-dq8, 327.192GB,
  1745 tensors (=1119+313x2 exactly), ~45min. OFFLINE GATES: census 313/313 q8 triples + 0 strays
  + experts/ts intact; BIT-PARITY EXACT vs mx.quantize of old checkpoint's bf16 (pipeline
  equivalence proven at tensor level). SWAP: old Ultra deleted (user-approved; NVFP4 master
  archived on T7), baked copied to internal (327GB), discovery restart. LIVE: all 4 "baked
  checkpoint detected" lines, resident 305.08GB (==305.07 load-time), decode 13.09/13.07 tok/s
  (==13.07/13.08), gsm8k 96.67% n=30 (==96.67%), load 68s no-quantize-pass. Serving name is now
  Nemotron-3-Ultra-oQNVFP4-dq8 (new name deliberate: avoids stale SSD-cache poisoning).
  This flow (repack + quant-first bake + gates) is the validated template for GLM-5.2-NVFP4.

- [x] STREAM-BUG CLOSED (2026-07-05, retroactive — supersedes the ":1065 in flight" note): Ultra
  "finding (1) stream bug" was NOT a serving defect. Nemotron-3-Ultra is a thinking model — it streams
  reasoning on delta.reasoning_content, switching to delta.content only for the answer; a content-only
  client sees "dead air" then the whole answer as one chunk at completion (the bogus "48 tok/s @5k").
  Confirmed by a delta-probe (reasoning_content channel); two stream-tracer theories refuted; non-stream
  API + cadence_probe.py correct. RESOLVED, not a regression. (Distinct from the M3 thinking-parser edge at :677.)

- [x] OMLX-PERF SKILL BUILT (2026-07-06, Opus swarm under Fable lead, user-approved plan): repo-level
  knowledge skill at .claude/skills/omlx-perf/ — 23 md + 7 scripts + 35 archived probes, ~4.9k lines.
  All campaign knowledge institutionalized: 12 iron laws, 3 playbooks, 64-entry EXP registry
  (EXP-001..064) + dead-levers graveyard, 30+ gotchas, preflight.py (runnable, serverless),
  per-model dossiers x5, conversion pipeline, kv-cache hot/cold, env/ops runbooks, future-campaigns.
  BUILD: 12 extraction agents -> 3 verifiers (claim-sampler ~700 claims / consistency 14 adjudications /
  completeness critic) -> synthesis -> Sonnet cold-load exam. VERIFICATION CATCHES: 3 blockers
  (laws.md cited a RETRACTED sparse-disabled theory as proof; GLM MTP 512->256 experts; M3 GPU-wait
  52% not Ultra's 46.5%), decode-burst budget DOES exist (engine_core.py:163-175 — extractor searched
  wrong file), OPS=500 for M3 SUPERSEDED by golden 4000/4000 (P0, ledger-chronology adjudicated),
  enforcer is boot-dependent 489-496GB (sysctl set)/464 (unset), 5 stale compaction-memory numbers
  refuted by corpus. Staleness anchor: verified 2026-07-05/06, MLX 0.31.2, omlx 0.4.5.dev1.

## GLM-5.2-oQNVFP4 CONVERSION CAMPAIGN — SHIPPED (2026-07-06, future-campaigns #1)

- Source `~/glm52-nvfp4-src/` (nvidia/GLM-5.2-NVFP4, 464.82 GB verified) → `~/.omlx/models/unigilby/GLM-5.2-oQNVFP4` 427.747 GB / 3130 tensors / 77 shards (~1.2 h convert).
- Recipe: byte-exact NVFP4 expert repack (pre-fused gate_up + ts sidecars) · DQ8 affine8-gs64 shell · kv_b pre-absorbed→embed_q/unembed_out q8 · indexer/router/norms bf16 · MTP layer 78 + input_scale dropped.
- New code: `omlx/tools/oqnvfp4_glm_convert.py`, `omlx/patches/glm_moe_dsa/nvfp4_ts.py` (ts-fold, [GLM-TS] counter, kill OMLX_GLM_DISABLE_NVFP4_TS), model_loading.py glm branch on `omlx_moe_nvfp4_ts`. Tests: 23+11 new, 12 glm regression green. Codex review: 2 P2s fixed (index_topk_pattern fallback; refuse non-empty --out).
- Gates: G1 census 3130 exact · G2 repack max|diff|=0.0 vs ModelOpt formula + DQ8/absorption bit-parity · G3 load 59s ALL PASS · G4 live: ts-fold ENGAGED, native kernels + fused indexer ENGAGED (bf16).
- Live: resident 398.9 GiB (enforcer ceiling 491.7 → ~93 GiB KV slack; 400k fp16 KV ≈ 35 GiB fits). Same-session A/B (t1t256 + gsm8k n=60 seed7): decode 16.25/16.46/19.01* vs 3.5bpw 22.74/23.09 (0.71×, napkin-exact); gsm8k 93.3 vs 95.0 WASH. (*64k rung restore-variance suspect.)
- Settings: `GLM-5.2-oQNVFP4` entry added (400k, fp16 KV, mtp off); backup model_settings.json.bak-glm-oqnvfp4. Server left running (tmux omlx:1, log ~/.omlx/serve-glm-oqnvfp4.log).
- OPEN: mmlu/arc A/B (gsm8k can't discriminate the calibrated-expert claim — M3 precedent); 400k soak (user-deferred); --shared-nvfp4 lever (−1.4 GB, untested); decide serve roster (keep vs 3.5bpw) after mmlu/arc.
- VERDICT (EXP-069, one-hour battery, T0 19:35 done 19:54): mmlu 82.0 vs 85.6 (n=250) / arc 94.7 vs 96.7 (n=150) / gsm8k 93.3 vs 95.0 (n=60), paired temp0 — NVFP4 uniformly WORSE. 3.5bpw STAYS PRODUCTION. Tuned Dynamic 3.5bpw > straight ModelOpt w4a16 on GLM-5.2 (opposite of the M3 outcome — per-model, not a law). Build+source parked pending disk call.
- RCA (EXP-070, workflow 5+judge, 2026-07-06): pipeline EXONERATED (fold 0.51% output rel-RMS vs fp32 ideal; gather_qmm nvfp4 exact; routers/norms byte-identical across builds; live 40-item slice TIED 31-31, failures fluent-wrong, zero degeneration). avlp12 = plain static RTN (NOT calibrated — earlier framing corrected). Gap sub-2σ everywhere; surviving mechanism if real = NVFP4 error character (symmetric grid, no mean preservation) + max_tokens=8 truncation artifact. M3 "win" equally under-powered. Follow-ups ranked: stage2 re-ask, logit-KL probe, mmlu n>=1000 McNemar, expert-swap ablation, re-power M3. WATCH: server died silently 20:45:25 after ~500 rapid requests + swap (no shutdown log) — restarted with golden env.
- PRODUCTION-SETTINGS A/B (EXP-071): mmlu n=50 seed-paired, NO request overrides (server defaults: temp1/top_p.95/thinking ON): oQNVFP4 86.0% vs 3.5bpw 84.0%, zero truncations both. The temp0 −3.6pp FLIPS to +2.0pp → quality is a WASH under production inference; the temp0/8-token protocol penalized reason-first GLM (confirms RCA truncation-artifact hypothesis). Roster rationale updated: 3.5bpw stays on EFFICIENCY (−122 GB, +40% decode at tied quality), not on quality. gsm8k leg cut by user; NVFP4 per-item JSONL lost at boundary kill (headline from progress line); 3.5bpw JSONL in scratchpad.
- SPECPREFILL DISABLED FLEET-WIDE (2026-07-07 00:xx, user order "spec prefill sucks"): specprefill_enabled=false on Qwen3.5-122B-oQ4, Super-oQ4e (fleet default!), GLM-5.1-2.9bit, MiniMax-M2.7-6bit; backup model_settings.json.bak-specprefill-off; server bounced clean (new log serve-glm-specoff.log). Closes the production exposure of future-campaigns #5 (92%->15% concurrency cliff); the BUG remains unfixed/open if SpecPrefill is ever re-enabled. Note: first bounce attempt hit a mid-drain server (a stray 2048-tok request at 22:21) — second attempt via tmux C-c clean.

## OVERNIGHT FLEET BATTERY (EXP-072, 2026-07-07 01:45-07:38, production settings, 7h order met at 5h53m)
| bench | M3-oQNVFP4-fused | Ultra-oQNVFP4-dq8 | GLM-5.2-3.5bpw |
|---|---|---|---|
| mmlu | 87.5 (n=200) | 87.3 (n=110, clock-capped) | 88.5 (n=200) |
| arc | 98.3 (n=120) | 96.25 (n=80) | 98.3 (n=120) |
| gsm8k | 96.7 (n=60) | 97.5 (n=40) | 95.0 (n=60) |
| median mmlu gen-tokens | 103 | 266 | 293 |
| leg wall | 61m | 135m | 157m |
- Protocol: NO request overrides (each model's serving defaults), seed-7 paired items, max_tokens 2048/4096, per-item JSONL in session scratchpad. Zero truncations anywhere; zero error retries; silent-death bug did NOT reproduce under sustained generation (~800 long requests) — narrows it to rapid-tiny-request regime.
- Fleet read: accuracy three-way tie (all within ~1-2pp); M3 is the cheapest per query by far (2.5-3x fewer output tokens AND ~2x decode speed of Ultra); GLM edges mmlu; Ultra edges gsm8k. GLM prod mmlu 88.5 n=200 confirms the earlier n=50 84.0 was noise-low.
- Ops: Ultra copied back T7->internal (byte-verified 327.2GB, ~/.omlx/models/Nemotron-3-Ultra-oQNVFP4-dq8) + NEW settings entry (Super template: temp 0.8/top_p 0.95, 1M ctx, fp16 KV, spec all off — reconstructed, ORIGINAL Ultra settings unknown); server on serve-overnight.log with golden env + DQ8 quartet.

## ALIS 4.5bpw ADOPTION + INDEXER ROPE FIX (2026-07-07)
- 4.5bpw deployed (424GB, drop-in confirmed: no ts sidecars, IndexShare layout matches our vendored path). Decode leg A (standard env): 16k=21.07 tok/s (vs oQNVFP4 16.46 — 4-bit shell pays); 64k=26.1 (T1-jitter suspect); 128k reading invalid (restore variance broke T1/T250 subtraction); 256k GUARD-REJECTED (~505GB est: 424w + 24KV + inflated transient est) — 4.5bpw fp16 ceiling ~<256k practical, use int8-KV or 3.5bpw for longer.
- ROPE BUG (EXP-073): fixed in omlx/patches/glm_moe_dsa/{deepseek_v32,glm_moe_dsa_model}.py + test rewritten (old test asserted the bug). Affects long-ctx retrieval QUALITY only, both Alis builds, retroactively fixed via GLM defaults. SSD cache (200GB!) flushed at 14:0x restart.
- README goldmine (4.5bpw card): int8 KV = latent-only dequant-on-read (same class as ours; their stack shows no prefill penalty — ours had 2x from throttle spiral, re-test before use); int8 100-200x lower MSE than fp8 on the latent; fp16 ceiling ~350K at this weight size; MTP +15-16% on nvfp4 experts (accept 2.61-2.87) vs ~0% on 3-bit — CONFIRMS our foray's regime finding from the other side; 3 MTP footguns documented (concat order, pre-final-norm hidden, normed-hidden chaining).
- Leg B running (batching env + flushed cache): batch 1/2/4 x 16k/64k/128k + fresh 256k attempt.
- NEXT: quality battery rerun; MTP enablement leg (verify footgun #3 in our mlx_lm_mtp glm graft, target ~19 tok/s short); banger settings: 4.5bpw flagship @400k fp16 + MTP.
- MTP LEG (EXP-075): remap/spec plumbing DONE (three dialects reconciled: weights model.layers.78.* / specs mtp.layer.* / ours mtp.0.*; the 3 mystery bits=3 config entries = layer-78 switch, NOT dynamic_quant tiers). Chained K=2 runs but a1=44% (card 75%) at NET-NEGATIVE speed + ~10GB/req retained memory (3x repro; echoes silent-death/rapid-request class + PR#2103). Warm-off A/B: warm HELPS (44 vs 27% a1) — warm not the poisoner. Parked with ranked suspects: (1) MTP-layer attention position/rope offset at decode, (2) MTP-path buffer retention, (3) drafter eh_proj pairing at decode. mtp_enabled=false restored; server healthy (15 models).
- 4.5bpw baselines (standard env, MTP off): short 20.72 / 16k 21.07 / 64k ~18-26(jitter) tok/s; 256k fp16 guard-blocked. REMAINING to crown: quality battery rerun (mmlu200/arc120/gsm8k60 vs 3.5bpw's 88.5/98.3/95.0).
- WORKFLOW glm-int8kv-and-mtp-rescue (8 agents, 47min): MTP track SHIPPED (chained K-draft mode + warm-prefill + FIX3 leak plug in batch_generator.py +423 lines; judges: lossless CONFIRMED, no qwen/dsv4 regression, 114 tests green; integration proven bit-identical to fork via dual-pipeline). Live leg (EXP-076): a1=92%/a2=85%, 2.70 tok/cycle, memory FLAT — but net only ~+3-7% (verify 2.5x/cycle; our fast baseline shrinks MTP headroom; card +15% is vs their 16.3 baseline). mtp_enabled=true KEPT (K=2, lossless, stable). int8 track: NOT implemented — recon+judge surfaced the buried post-mortem (even with clamps 1-4 fixed live prefill stayed 2x; 5th clamp never found; int8=capacity-only on M3) → DECISION GATE for user; full cherry-pick map banked (sibling 3741f8d+c0610a2, kernel rebuild recipe, clamp-hunt checklist in workflow journal wf_c67977dc-853).
- FORK int8 A/B (EXP-077): 16k prefill fp16 124.2 vs int8 121.5 tok/s = NO intrinsic penalty; peak-mem delta 0.68GB confirms engagement. Our 2x was pure omlx throttle interaction. int8 door REOPENED: port = fix the transient estimator (intercept!) + clamp checklist (workflow journal wf_c67977dc-853) with the fork as live reference. Decode-side cost needs a real sample (8-tok run suggested ~0.8x — verify before enabling). Server restored (mtp on, K=2, VERIFY_FAST).
- WATCH-ITEM #2 (2026-07-07 19:01): ABORTED-PREFILL RETENTION — client-killed 500k prefill left ~37GB resident (440.24GB observed vs ~403 expected); subsequent 128k T250 rejected by the 90% prefill safety cap. Same retention-after-teardown family as PR#2103 / MTP warm-holder. Repro: start huge prefill, kill client mid-flight, observe resident. Cleared by restart. Needs a request-abort KV/pool reclamation audit.

## PUZZLE-75B oQ48 CAMPAIGN (2026-07-08/09, EXP-091..098 — distilled truth in .claude/skills/omlx-perf/, dossier models/nemotron-puzzle.md)
- CONVERSION (EXP-091): nvidia Puzzle-75B-A9B BF16 master (156.6GB) → oQ48 = THIRD converter lineage (`omlx/tools/oq_puzzle_convert.py`, bake-from-bf16, no repack/ts-fold): shell affine8-gs64 (257 targets, SHELL_STAGES single-source), experts affine4-gs64 stacked per-layer heterogeneous (block_configs: inter 1280-2688, k 4-18), MTP dropped. 46.9GB/1437 tensors. Vendored model class `omlx/patches/nemotron_h_puzzle/` (per-layer plumbing; torch-ref logits parity 1.2e-6). Offline gates ALL PASS (`oq_puzzle_gate_offline.py`). NVFP4 sibling REJECTED pre-build (Super 0.58x gs16 tax at skinnier shapes); FP8-direct rejected (no MLX container).
- G4 (isolated :8001 instance): decode 54.30 short / 53.94@5k / 49.84@64k (1024-tok window; probe's 80 = Law-15 restore artifact); prefill 656-810 tok/s; resident 43.84GB. gsm8k 92.5 n=40 (=Super) / mmlu 83.0 / arc 94.0; thinking-on spot 15/15. FASTEST real-tier model in fleet. MUST serve from internal SSD (omlx#2098; Clone symlink caught by preflight).
- DECODE AUDIT (EXP-092): no impl bugs; ~17 launches/layer x88; sample 61% GPU-wait pipelined. Pools: mamba glue 6.1ms / router 3.7 / experts 2.8 (EAGER-measured — see EXP-095 for why that mislabels the win).
- MLX 0.32 A/B (EXP-093, side venv .venv-mlx032): single-stream FLAT; qmv_wide → batched decode B=4 1.23x (168 agg tok/s) / B=8 1.32x. Traps: pyproject override-dependencies pins mlx from inside cwd; transformers must stay 5.12.x; nanobind 2.13 → GLM ext rebuild before fleet adoption.
- FUSION (EXP-095): pool A (fused mamba glue) SHIPPED-WIRED +2.9% (54.3→56.0, token-identical 192/192, kill switch OMLX_PUZZLE_DISABLE_FUSED_MAMBA, grep [PUZZLE-FUSE] mamba=40/40); pool B fused router KILLED 7.3x loss (GLM dead-verdict transfers); pool C fused experts KILLED-UNSAFE (Metal cmdbuf fault under PIPELINED decode only → new Law 18). Laws 17 (harness naming — a verify_econ t1 briefly corrupted this ledger's fusion number) and 19 (eager pools are pipeline-hidden) minted.
- MTP DISTILL (EXP-094/096): NVIDIA head weak (a1 0.545; wiring POST-norm_f + embed-first, verified vs TRT-LLM). Full-finetune KL-top8 distill on 16 generic self-gen seqs / 10 steps → a1 0.668 (math .808), QUANT-LOSSLESS, stochastic accept 0.647 at prod params (T=1.0/top_p=.95 ≈ argmax; model sharp at T=1 → Law 16 qmv/qmm identity law minted here). Sidecars: ~/.omlx/mtp_sidecars/puzzle75_mtp_{oq48,bf16,distilled_oq48,distilled_bf16}.safetensors.
- SPEC DECODE (EXP-097b/098): clone-on-verify loop CORRECT (zero-copy mamba snapshot beats MTPLX's deep-copy; quality rail tie-only) but LOSS everywhere: K1 0.789x/0.801x (0.31/0.32). EXP-098 refuted both opt hypotheses (host .item() drains ~nil; reject GPU work NOT free — compute-bound box); EMA-gated pipeline ships +7.9%/cycle, caps 0.82x. Even a1=1 loses at the 26.1ms full-accept floor unless accept-streaks chain → SHELVED, serve PLAIN. Revival bar: sustained a1>=0.85 (hermes-agent-trace re-distill, corpus location TBD from user) or mamba scan-state exposure (+15-20% est, invasive, unbuilt). Artifacts: _src/puzzle_campaigns/{spec_loop,distill,mtplx}.
- OPEN ITEMS: :8000 admin reload to discover the model; model_settings row (alias/sampler, user approves); optional 0.32 batch>=2 adoption window.
