> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Dead levers — the graveyard (check here before re-testing anything)

Fast-lookup verdict index so nobody burns a day re-measuring a settled lever. **DEAD** = measured/proved
not to pay; **PARKED** = shelved with a named revival gate; **UNSUPPORTED** = the mechanism does not exist
in this MLX. Most revival conditions are "batch≥2" or "new MLX version" — stated per row. EXP-05x refs are
in `experiments/ultra-day.md`; pre-Ultra refs are resolved to their `experiments/pre-ultra.md` EXP-NNN
below. Full campaign context: `experiments/ultra-day.md`, non-goals in `tasks/ultra_speed.md §6`.

## Host serialization / execution

| Lever | Verdict | Evidence | Revival condition |
|---|---|---|---|
| `mx.compile` of the decode step | DEAD (3× independent) | GLM foray: `mx.compile(full L=2 forward)` = 1.06× (only ~6% fusible — the diffuse ~32ms is in opaque big kernels `gather_qmm`/SDPA/`quantized_matmul`, not fusible glue); M3 flat; GLM L=1 flat. `GLM52_MTP_FORAY.md:357-360,402`; `tasks/ultra_speed.md:513`; `pre-ultra.md` EXP-046 / EXP-010 (GLM L=2 compile 1.06×) | None foreseeable — single kernels aren't fusible. Only a fused-decoder-layer/mega-kernel rewrite attacks this class. |
| Raise MLX `MAX_ACTIVE_TASKS` via env (task-cap) | DEAD | It's a compile-time constant, not an env var; cap 10→64 measured FLAT on both GLM and M3. `GLM52_MTP_FORAY.md:402`; `tasks/ultra_speed.md:514`; `pre-ultra.md` EXP-045 | Wheel-patch route only (P2-1 banked cap wheel), and only if P0-3 chunked-eval pays but leaves ≥10ms on the table. Process-global; re-verify nanobind 2.12.0 ABI. |
| Buffer env beyond golden 4000/4000 (`MLX_MAX_OPS_PER_BUFFER`/`MLX_MAX_MB_PER_BUFFER`) | DEAD | 500→2000 measured flat on M3; golden 4000/4000 stands. `tasks/ultra_speed.md:515-516`; `pre-ultra.md` EXP-044 | One lead-run re-sweep AFTER P0-2 op-count work lands (10 min, not a workstream). Missing the golden env is itself a harness artifact — see EXP-057. |

## Speculative decode (batch-1)

| Lever | Verdict | Evidence | Revival condition |
|---|---|---|---|
| Batch-1 speculative decode — **MTP and EAGLE-3 both** | DEAD (temp0 AND temp1) | Verify L=(K+1) reads ~(K+1)× the DOMINANT expert weights at batch-1 MoE (distinct experts/token), dense/attn read ~once → verify cost scales ~1:1 with committed tokens → CAPPED at break-even even at α=1. Measured: GPU-wait @5k = 75ms = 1.83× base decode (41ms), commits 1.83 tok = break-even; @16k = 87ms = 2.12×, commits 1.85 = LOSS. Sampling-independent (temp1 accept ≤ temp0 → strictly worse). `tasks/eagle_temp1.md:106-121`; `tasks/lessons.md:59-68`; `GLM52_MTP_FORAY.md:285,396-404`; `pre-ultra.md` EXP-039 / EXP-047 | batch≥2 (expert reads amortize across the batch), OR a DENSE target model, OR the residual-2× dispatch-gap/mega-kernel work. On Ultra also inert: `_is_mtp_compatible` excludes `nemotron_h` (`omlx/utils/model_loading.py:_is_mtp_compatible:456`; :386-397 is the sibling `_has_mtp_heads`) and sanitize drops `mtp.*` (doctrine (d)). |

## Quantization containers & requant

| Lever | Verdict | Evidence | Revival condition |
|---|---|---|---|
| Fused expert-MLP kernel (K3, fc1+relu²+fc2 one dispatch) | PARKED→P2 | K1 in-stream: experts 1.87× ideal but expert-specific excess only **2.41ms/token** (dense-equal-bytes control 1.71× ≈ the unsorted path) → a fused kernel can capture only ~2-4ms/token, nowhere near DQ8. EXP-057; `tasks/ultra_speed.md:18-29,311-325` | batch≥2, OR the future cross-model mega-kernel campaign (which targets the 1.71× diffuse floor itself, alongside GLM §10b's ~32ms). Re-run K1-style attribution per model. |
| Requantize NVIDIA's nvfp4 experts → affine4 (P1-1) | DEAD | Doctrine (b) preserve NVIDIA's weight values, AND measured moot: M1 shows nvfp4-gs16 / affine4-gs64 / mxfp4-gs32 all within ±4% at Ultra shapes — requant to any other 4-bit container buys nothing and is lossy-on-lossy. EXP-055; `tasks/ultra_speed.md:472-474,523-528` | None (doctrine + measured equal). |
| Container swap nvfp4 → mxfp4 (or → affine4) for experts | DEAD | Lossy-on-lossy (double-rounding, group-size mismatch), barred by doctrine (b); AND measured speed-equal (±4%) at Ultra shapes. EXP-055; `tasks/ultra_speed.md:523-525` | None. mxfp4-gs32 stays a measurement-only column, never a serving path. |
| "affine5-gs16" exact container (25-level integer grid) | UNSUPPORTED | MLX 0.31.2 `mx.quantize(..., group_size=16, mode="affine")` rejects gs16 ("group size 16 is not supported"; affine = gs 32/64/128 only); gs32 can't be exact over per-16 NVFP4 scales. Integer-grid math was sound; the container doesn't exist. `tasks/ultra_speed.md:96-101,526-528` | A future MLX that adds affine gs16 — but M1's no-mode-gap result makes any container swap moot regardless. |
| DQ8 (8-bit) of attention **k_proj/v_proj** | DEAD | M0 measured 0.87-0.91× = a LOSS on the skinny 8192→256 shape → k/v EXCLUDED from DQ8 (stage C is q_proj/o_proj only). EXP-054; `tasks/ultra_speed.md:14,345` | None at this shape. Also blocks the 2c qkv-pack (mixed precision in one matmul is impossible). |
| Dense 6-bit / mixed-precision push | REJECTED (codex) | Estimated only ~5ms beyond DQ8 for outsized quality risk. `tasks/ultra_speed.md:520-522`; `tasks/ultra_speed_review_codex.md:38-39` | Reopen only after DQ8 per-module ablations show a large quality margin AND P0-2/P0-3 live deltas are known. |
| DQ8 dense-shell bake of **MiniMax-M3** (fs5 or oQNVFP4) | DEAD (2026-07-06) | Header-only census (`scripts/dq8_costmodel.py`): fs5 = 88.4% U32 + 10.6% U8 (NVFP4 experts+scales), only **2.26 GiB BF16 (1.0%)**; the first-pass script flagged **1.80 GiB** as "dq8-eligible" but **that was a classifier bug** (counted bf16 `.scales`/`.biases` sidecars + the VL vision tower as bakeable weight — fixed in `scripts/dq8_costmodel.py`). M3 decode is also **dispatch-bound, not bandwidth-bound** (`models/minimax-m3.md §6/§8`). **PROVEN NO-OP 2026-07-06** (`scripts/m3_dq8_offline_gate.py`): the M3 converter (`oqnvfp4_convert.py`) emits an affine8-gs64 dense shell BY CONSTRUCTION (no `--dq8` flag — unlike Ultra), so oQNVFP4's text shell is ALREADY **130/130 affine8-gs64** (q/k/v/o + index + dense-mlp + shared + lm_head + embed), **0 GiB bf16 text weight left to bake**; the cost-model's "1.80 GiB eligible" is ~**1.61 GiB vision-tower bf16** (out of the text decode hot path) + affine8 scale/bias sidecars the header classifier miscounts as bf16 weight. A re-bake is non-idempotent (dequant→re-quant re-tightens ~20% of packed words = lossy 2nd rounding, Law 4). So the "bake" is not a candidate — it's the base recipe. EXP-065; `models/minimax-m3.md §8` | None. The fs5 recipe already quantizes the shell (attn5 band 17–44 + fused-nvfp4 shared). A MoE this sparse (~99% experts, already 4-bit) has no bf16 shell left to bake. Only the mega-kernel (dispatch) attacks M3 decode (`future-campaigns.md #3`). |
| Drop MiniMax 5-bit attn band (17-44) to "recover quality" | DEAD (2026-07-06) | Same-session live A/B (EXP-066): `-fused` (uniform affine8 attn) vs `fs5` (5-bit band 17-44) — mmlu 80.8 vs 78.8, arc 92.0 vs 94.7, gsm8k 96.7=96.7 → quality **WASH** (deltas within n/sampling noise); but `-fused` is **~5% SLOWER** (short 27.26 vs 28.54, 16k 25.43 vs 27.08). The band is a **free ~5% speed win at neutral quality**. `models/minimax-m3.md §6/§7` | None — the band costs no measurable quality; keep it (or re-add via `--attn5-layers 17-44`). NB: production adopted `-fused` anyway for the attn-precision margin, accepting the 5%. |

## Cache / KV quantization

| Lever | Verdict | Evidence | Revival condition |
|---|---|---|---|
| int8 / mxfp8 MLA-KV (or latent) cache | DEAD (built then abandoned) | 2× prefill, but fp16 already fits 1M-ctx on 512GB → it's a capacity feature, not a speed lever; mxfp8 latent hits the same wall. `tasks/ultra_speed.md:517`; memory `omlx-glm52-int8-mla-kv`; `pre-ultra.md` EXP-004 | Only if context demand ever exceeds fp16 KV capacity (i.e. as a capacity feature, never for speed). The q8 MLA kernel itself stays committed. |
| Indexer fp8 (fused_index fp16 path) | DEAD-on-arrival | `fused_index.py` required fp16; live M3 runs bf16 → 100% silent fallback (`fused_none=57/57`) doing full-cache fp32 astype+matmul glue. A gated fast-path with no live engagement counter shipped dead. `tasks/lessons.md:17-26`; `pre-ultra.md` EXP-025 / EXP-026 | Rebuild the path for bf16 (live dtype) AND ship an engagement counter — never trust "verified standalone". |

## Parked (not dead — named gate)

| Lever | Verdict | Evidence | Revival condition |
|---|---|---|---|
| 2c qkv-pack (one 8192→8704 matmul + views) | PARKED | Payoff small (~24 dispatches/token); AND once k/v are excluded from DQ8 and q/o are quantized, a packed projection would need mixed precision in one matmul — not possible. `tasks/ultra_speed.md:404-408` | Viable only as all-bf16 (pre-stage-C), or if k/v are folded back into DQ8. Gated behind M4 + a qkv-specific in-stream probe. |
| Sorted expert routes (P0-2d) as a *speed* lever | PARKED (kept only as free correctness-neutral) | Isolated 3.5ms/token did NOT survive in-stream — live speed FLAT at 8.02. Kept because bit-identical and ~zero cost. EXP-058; `tasks/todo.md:1071-1072` | batch≥2; or fold into the 2b-topk fused-route kernel (emits sorted indices in-register for free). |


## Puzzle-75B (nemotron_h_puzzle) fusion & self-speculation — EXP-095..098, 2026-07-09

| Lever | Verdict | Evidence | Revival condition |
|---|---|---|---|
| Fused router kernel | DEAD — 7.3x LOSS live | EXP-095: engaged 40/40 but MoE block 329->3169us; single-threadgroup 512x4096 gemv loses to MLX gemv. GLM's fused-router dead verdict TRANSFERS. Banked unwired: `omlx/patches/nemotron_h_puzzle/fused_router.py` (numerically exact, 1e-7). | None foreseeable for this kernel design — a hand gemv can't beat MLX's. |
| Fused experts pool C | PARKED — killed-UNSAFE | EXP-095: parity PASS + clean eager, but intermittent Metal cmdbuf fault under PIPELINED decode only (Law 18's founding incident). Root cause UNDIAGNOSED. Banked unwired: `omlx/patches/nemotron_h_puzzle/pool_c.py`. | Kernel-safety root cause + a sustained pipelined soak re-clearing Law 18. Bytes-won ceiling is small anyway (Law 19: pool was eager-sized). |
| MTP self-spec decode (ALL forms: naive, clone-on-verify, EMA-gated pipelined) | SHELVED — serves PLAIN | EXP-097: correct loop, LOSS 0.79x/0.80x (0.31/0.32), cycle 41.3ms vs ~23 ideal. EXP-098: host .item() drains NOT the cost (refuted); reject GPU work NOT free (compute-bound; unconditional speculation made K2 15% slower); EMA-gated pipeline ships +7.9%/cycle but caps 0.82x; pure-pipeline full-accept floor 26.1ms/cycle. Head itself is fine: distilled a1 0.668 (from 16 generic self-gen seqs, EXP-096), quant-lossless. | Sustained a1 >= ~0.85 on served content — the untried lever is re-distilling on REAL agent traces (hermes/herdr; never used — the 0.668 head trained on 16 generic sequences), which is high-bar but the only path that keeps the speculation gate armed. |
| Mamba per-token scan-state exposure (make reject-reforward trimmable) | UNBUILT — mapped only | EXP-098 judge estimate +15-20% on low-a1 categories; requires invasive mixer surgery. | Fund only if the hermes-trace distill lands a1 in the 0.75-0.85 gray zone where the reject tax is the remaining blocker. |

## GAPS
- The pre-Ultra EXP IDs above are resolved to `pre-ultra.md` EXP-NNN (mx.compile→EXP-046/010, task-cap→045, buffer→044, EAGLE→039/047, int8-MLA-KV→004, indexer-fp8→025/026); each row keeps its concrete second citation (GLM foray / lessons / ultra_speed non-goals) as a backstop.
- The int8 MLA-KV memory note (`omlx-glm52-int8-mla-kv`) was cited by index, not re-read here; the ultra_speed.md:517 non-goal line is the corroborating primary source I did read.
