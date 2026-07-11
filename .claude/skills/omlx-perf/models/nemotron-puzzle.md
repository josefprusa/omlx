> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Nemotron-Labs-3-Puzzle-75B-A9B — dossier (oQ48 build; G1–G4 ALL PASS 2026-07-08)

Puzzle-NAS-compressed sibling of Super-120B: same NemotronH skeleton, but **per-layer heterogeneous MoE**
(`block_configs`: moe_intermediate_size ∈ {1280,1536,1792,2048,2688}, num_experts_per_tok ∈ {4..18}, avg 11).
Built 2026-07-08 via the `puzzle-oq48-build` workflow (10 agents, EXP-091). **FASTEST real-tier model in the
fleet: 54.3 tok/s decode (Super 49.4, M3 27.5, GLM 21.7, Ultra 13.1) at Super-level quality, 43.84GB resident.**

## 1. Identity
- **Source:** `nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16` (156.6GB / 32 shards, verified byte-exact
  vs index), archived at `$OMLX_COLD_STORAGE/omlx-models/_src/Puzzle-75B-A9B-BF16`. Siblings: NVFP4
  (53.5GB — NOT used: gs16 gather_qmm small-shape 0.58× tax, see nemotron-super.md §6) and FP8 (83GB — not
  needed once BF16 master chosen).
- **Serving artifact:** `NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-oQ48` — REAL COPY on the internal SSD at
  `~/.omlx/models/` (§7: never serve via the Clone symlink). **46.908 GB / 1437 tensors / 88 layers**
  (converter Summary line). `model_type
  nemotron_h_puzzle` served via vendored class `omlx/patches/nemotron_h_puzzle/` (hy_v3-style sys.modules
  registration, wired in `omlx/utils/model_loading.py`). No `model_settings.json` row yet (GAP until G4).
- **Thinking model** — emits reasoning then `</think>`; quality benches MUST run production settings
  (GLM-NVFP4 law) and count both channels when streaming.

## 2. Geometry (from config; BF16 master uses `model.` prefix, NOT `backbone.` — sanitize handles both)
75B total / 8.8B active. hidden 4096, vocab 131072, ctx 262144. **88 layers = 40 mamba2 + 40 MoE + 8 attn**.
MoE: 512 experts in latent-1024 space, relu² gateless, shared 5376, norm_topk_prob=true,
routed_scaling_factor=5.0, e_score_correction_bias (selection-only). Mamba: 128h×64, n_groups 8, **ssm 96**.
Attn: GQA 32/2 hd128, NO RoPE. MTP block in source (**dropped** in oQ48 — tensors + config keys).

## 3. Quant layout — oQ48 (single quantization from bf16, `mx.quantize` deterministic)
| family | container |
|---|---|
| routed experts (stacked/layer, hetero widths) | **affine4 gs64** (80 stacked triples) |
| shell: mamba in/out (80) + shared/latent (160) + attn q/o (16) + lm_head (1) = **257** | **affine8 gs64** |
| k/v_proj, gate+bias, norms, conv1d, A_log/D/dt_bias, embeddings | bf16 / f32 — NEVER |

Map single-source-of-truth: `SHELL_STAGES` in `omlx/tools/oq_puzzle_convert.py` (emitter + gate import it).
No ts sidecars, no runtime fold patch needed (unlike oQNVFP4 builds) — plain strict load.

## 4. Byte budget & ceilings (worked 2026-07-08, this file is the record)
Shell 7.30B (mamba 4.30 + shared 1.76 + latent 0.34 + attn 0.285 + lm_head 0.54) at 8.5bpw + active
experts 1.50B at 4.5bpw → **~8.6 GB/token → hard ceiling ~95 tok/s**. Realistic **~45–60**: the binding
constraint is the 88-layer diffuse per-op overhead floor, same class that capped Super-oQ4e at 49.4
(≈45–50% of its byte ceiling). bf16-shell repack-only would have been 15.4GB/token (ceiling 53) — why
oQ48 exists. NVFP4-container route rejected: Super's 0.58× gs16 tax at even skinnier shapes here.

## 5. Build provenance & verification (all PASS 2026-07-08)
- Workflow `wf_3140178e-263`: recon (incl. NVIDIA reference `modeling_nemotron_h_puzzle.py` math diff) →
  implement → 2 adversarial judges + tiny-checkpoint e2e → 1 fix round (gate sampling missed attn/lm_head
  stages — fixed, forced per-stage coverage).
- Model class vs NVIDIA torch reference: **fp32 logits max|diff| 1.2e-6, argmax identical** (heterogeneous
  5-layer fixture). Router parity incl. large correction bias. Tiny e2e bf16-vs-quant cosine **0.99990**.
- Real artifact: `oq_puzzle_gate_offline.py` **ALL PASS** (census 257+80, bit-parity 24 samples all stages,
  config lint). G3 real-path load + 3/3 greedy probes correct (Paris / 391 / Rayleigh).
- Tests: `tests/test_nemotron_h_puzzle.py`, `tests/test_oq_puzzle_convert.py` (bit-parity vs
  QuantizedLinear.from_linear, stage-map arithmetic 80/160/16/1).

## 6. G4 measured (2026-07-08, isolated omlx instance port 8001, golden env + production OMLX_* set,
## §9-identical; T1/T256 stream-free probes, nonced; load: `Loaded model: … actual: 43.84GB, est 45.87GB` in 7.5s warm)
| leg | tok/s | note |
|---|---|---|
| decode short | **54.30** | = 57% of the 95 byte-ceiling — best realized fraction at this scale (Super 45-50%) |
| decode 5k | 53.94 | prefill prime 7.89s ≈ 656 tok/s |
| decode 16k | ~54 (probe read "61.11" on a 79-tok sample — small-sample artifact, EXP-087 law) | prefill ≈ 810 tok/s |
| decode 64k | **49.84** (1024-tok window) | probe's "80.02" = Law-15 restore-variance artifact at 2.9s window; taper matches +0.86GB/tok KV reads (ceiling 95→87) |
| in-stream serve | 52.4 (server's own 357-tok completion log) | corroborates |

Quality (serial, temp0, `acc_bench_serial.py` thinking-OFF — same harness as the Super baselines):
**gsm8k 92.5% n=40** (Super oQ4e 92%), **mmlu 83.0% n=100** (Super 81%), **arc 94.0% n=100** (Super 95%).
Thinking-ON gsm8k spot 15/15 n=15 (one raw "miss" was a 9.00-vs-9 parser artifact). **No quant damage.**

## 7. Serving notes / open items
- **MUST be served from the internal SSD** — first serve attempt via the Clone-volume symlink was caught
  by preflight (omlx#2098 USB Metal-timeout gotcha). Internal copy at `~/.omlx/models/…-oQ48`; Clone copy
  + BF16 master (`_src/`) are cold archive.
- The user's production server (port 8000) discovered its pool BEFORE this model landed — it needs
  `POST /admin/api/reload` (admin UI) or a restart to see it. No settings row yet (GAP: alias, sampler).
- No engagement counters exist (no gated fast path, no ts-fold) — a clean `Loaded model:` line is the
  entire engagement story. Ultra's decode-opt fingerprints verified NOT matching (no cross-patch risk).
- Speed levers if ever needed: mxfp8 shell (~+3% ceiling, bench mxfp8 M=1 kernel first), batch≥2.
- **MTP: DISTILLED, GATE CLEARED, but SERVE-PARKED on clone-on-verify (EXP-096, supersedes EXP-094).**
  Wiring (verified vs TRT-LLM NemotronHMTPDecoderLayer): concat EMBED-first, hidden = **POST-norm_f**
  (+7.5pp vs GLM pre-norm); pre-norm chain = garbage (0.02). Distilled the weak NVIDIA head (full-finetune
  bf16 MtpHead, KL-top8 + 0.1·CE, 10 AdamW steps, early-stop; tiny 16-seq overfit ceiling): held-out
  6-prompt battery n=640 **a1 0.545→0.668** (code .631 / math .808 / prose .509 / reason .621) — clears
  0.65 gate (+0.123), stretch .75 unreached. **oQ48 quant LOSSLESS: a1 0.668 (Δ+0.000, per-cat ≤±0.003)
  — quantization exonerated again.** Stochastic acceptance prod-params (T=1.0/top_p=0.95, min-sum
  Σ min(p_t,q_d)) = **0.647** (vs untrained .500/.563/.426) ≈ argmax (sharp at T=1). Chained a2|1
  post-norm **0.574** (math .727 held, code REGRESSED .703→.525: head over-specialized to depth-1).
  NET (E[tokens] accounting, mamba FULL-REPLAY, verify_econ t-costs t1..t4=15.5/18.74/21.76/23.37 @0.32 UNFUSED trunk,
  d=1.6): K=1 **only math wins 1.17×(A)/1.05×(B)**; overall 0.97×/0.87×, code .93×, prose .79× — replay
  penalty dominates (the corrected E[tok] denom is HARSHER than EXP-094's /2; even untrained math=0.88×).
  **DECISIVE LEVER = clone-on-verify** (MTPLX `snapshot_untrimmable_cache`: mamba2 state is fixed-size,
  snapshot before verify + restore on reject → kills the replay term): overall K=1 **1.27×(A)/1.14×(B)**,
  math K=3 up to 1.55×. So the serve-win is gated on the recurrent-state spec-loop, NOT more head training.
  Artifacts: `~/.omlx/mtp_sidecars/puzzle75_mtp_distilled_{oq48(1.72GB),bf16(5.9GB)}.safetensors` +
  untrained `puzzle75_mtp_{oq48,bf16}`; bf16 best_head at `_src/…/distill/prod_run/best_head.safetensors`.
  Harnesses `scripts/puzzle_mtp_alpha.py` + `puzzle_mtp_alpha2.py` (point `--ckpt` at the distilled head).
  **EXP-097 (2026-07-09, adversarial review): clone-on-verify loop BUILT + bookkeeping-VERIFIED, serve verdict
  still PENDING — NO-SHIP yet.** Loop at `_src/puzzle_campaigns/spec_loop/spec_decode.py` (own-harness baseline
  inside). Verified: zero-copy mamba snapshot legit (mixer rebinds cache slots; ssm_update functional), reject
  bookkeeping clean (identity smoke tie-only: code 64/64, math div@4 m0.0, prose @13 m0.25). Teacher-forced rail
  K=1: gate(a) PASS; a1 code .602/math .866/reason .661 in band, **prose .701 vs ref .51 VIOLATION** — A1_REF
  miscalibrated (EXP-096 refs are from ITS battery, not prompts.jsonl); cascade-artifact excuse refuted. Missing:
  ALL timed legs (ladder.md empty), gsm8k gate unimplemented, per-venv rail. 1.27x/1.14x are still projections.
  **FIXES (2026-07-09, code-only — box was NOT quiet, live server resident, so nothing re-timed):**
  (1) `A1_REF` re-derived ON prompts.jsonl `{code .602/math .866/prose .701/reason .661}` (the judge's own
  teacher-forced numbers; a1_pos1 is K-independent) → false prose VIOLATION fixed at root, not by widening the band.
  (2) gsm8k gate STRUCK, not stubbed: at temp0 spec ≡ plain within tie-class (identity rail proves it) so spec
  inherits plain's accuracy, and plain oQ48 already hit gsm8k 14/15 (EXP-095); re-add trigger + upgrade path noted
  in ladder.md. (3) `spec_decode.py --ladder` driver added — one model load, persists a timed baseline+distilled
  K1/2/3+untrained K1 table to ladder.md (fixes 'only prints, nothing writes ladder.md').
  **MEASURED 2026-07-09 (quiet box, own-harness plain baselines ~53 tok/s): LOSS EVERYWHERE — CAMPAIGN
  CLOSED by EXP-098: spec SHELVED on Puzzle-oQ48, plain (52.88 tok/s) wins outright; NO production wiring.**
  Final ladder (post-EXP-098 code, 0.31.2, `spec_decode.py --ladder`): distilled K1 **0.789x** / K2 0.675x /
  K3 0.557x, untrained K1 0.834x (best prompts math#3 0.929x/0.976x — nothing reaches parity; gate was 1.10x).
  tok/cycle ≈ EXP-096 alpha theory (K1 1.731, K2 2.109; math#3 K2 2.476 vs 2.50 predicted) — acceptance is NOT
  the problem; the untrained>distilled anomaly DECOMPOSES (untrained tok/cyc 1.777 AND fewer reject-reforwards:
  1.027x1.027 ≈ 44.08/41.74 — battery a1 doesn't transfer per-set). 0.32 leg ABORTED per gate protocol: quality
  rail FAILED = code a1 .684 vs ref .602 (|d| .082 > .08 band, higher-accept direction, qmv_wide tie rounding;
  match ≥99.2%, margins ≤0.5 → A1_REF calibration graze, NOT bookkeeping); re-derive A1_REF per-venv (code ≈.68)
  before any 0.32 verdict. ROOT CAUSE (EXP-098 decomp; ms harness = `bench_ab.py` self-timed single-load, banked
  in spec_loop/): lazy draft chain gained ~NIL (42.1→40.0ms — host .item() drains were never the cost); cycle =
  launch bubble ~7ms (full-accept seq 33.2 vs pipelined floor 26.1) + mamba reject-reforward ~23ms. Unconditional
  pre-built next verify = K2 +15% SLOWER (wasted spec work is NOT free when compute-bound). SHIPPED opt-in
  `--pipeline` (generate_spec_pipe, default+rails untouched): EMA≥0.75-gated 2-deep pipeline, one host sync/cycle
  → K1 39.16ms (+7.9%), K2 56.48 (+4.0%), best math#1 33.6ms ≈1.02x plain — overall still 0.82x. Judge-verified
  on final code (rail log births > code mtime; pipe identity K1/K2 re-run tie-only: code@107 m0.00, math@4 m0.00,
  prose@13 m0.25). Even α=1 loses (floor 26.1 > plain 18.8ms) — verify is COMPUTE-bound; spec only nears parity
  where a1→1 (math), which plain already serves fine. Only lever if EVER revisited: mixer surgery exposing
  per-token mamba scan states → trimmable reject (est +15-20% on a1≈0.6 cats); invasive, unbuilt. Full tables:
  `_src/puzzle_campaigns/spec_loop/{ladder,opt_log}.md`; EXP-097+EXP-098 rows FINAL.
- **MLX 0.32.0 A/B (EXP-093, side venv `.venv-mlx032` kept on disk):** single-stream FLAT (54.2 vs 54.6);
  batched decode is the win — B=2: 92→98, B=4: 137→**168** (per-stream 42), B=8: 179→**237** agg tok/s
  (qmv_wide). Upgrade the production stack only when the batch≥2 / MTP campaign opens; nanobind 2.13
  rebuild required for GLM ext first (future-campaigns.md §MLX v0.32.0). transformers must stay 5.12.x.

## 8. Decode audit (EXP-092, 2026-07-08) — where the 18.4ms/token actually goes
Offline decomp (chained per-block, one eval; golden env; wired limit set) + `/usr/bin/sample`:
**no implementation bugs** — ssm Metal kernel fires at L=1, no gather-sort at M=1, router compiled,
shared expert at 83% of physics, one-ahead offline 54.6 == server 54.3 (serving overhead nil).
The gap to physics is **structural op granularity** (~17 launches/layer × 88 layers), GPU-side:
sample shows **61% GPU-wait** in the pipelined regime — fusion (fewer launches), not host work, is the lever.
| pool (eager µs/layer × n) | measured | physics | pool size |
|---|---|---|---|
| mamba non-proj machinery (conv concat/slice/conv1d/silu/splits/dt/gated-norm) | 152 ×40 | <12 | **6.1ms** |
| router gemv+compiled select | 92 ×40 | ~10 | **3.7ms** |
| expert gather pair (floor-bound: k4/1280=265µs ≈ k12/2688=292µs whole-block) | ~70 ×40 | ~7 | **2.8ms** |
| latent proj pair | 32 ×40 | ~11 | 1.3ms |
| attention | 85 ×8 | — | 0.7ms |
| lm_head (838µs, 83% of physics; affine4 = −0.4ms, quality-gate first) | — | 696µs | 0.14ms |
Eager 23.1ms → one-ahead 18.3ms (pipeline already harvests host overlap). **Verdicts that do NOT
transfer here: Ultra K3 "fused expert MLP DEAD" (Ultra read 130MB/layer, Puzzle 6MB → floor-dominated);
GLM "fused router DEAD, compile wins @36µs" (Puzzle pays 92µs) — **BOTH REINSTATED by EXP-095**: the
fused router LOST 7.3× live (GLM verdict transfers), fused experts = Metal cmdbuf fault under PIPELINED
decode (killed-unsafe, banked unwired). Fusion campaign net = **POOL A (mamba glue) ONLY: 18.42→17.87ms
= 56.0 tok/s (+2.9%)**, token-identity 192/192 — the eager-measured pools were already pipeline-hidden.
70-75 via kernels NOT credible. (An earlier "15.5ms/64.5 tok/s" claim here was a harness-conflation
error — verify_econ t1 on the UNFUSED trunk; corrected 2026-07-09.) EXP-096's net accounting uses the
verify_econ t-costs consistently (ratios valid; ~+3% if pool A ships). Raw
probe preserved: `scripts/puzzle_decode_decomp.py` (box must be idle; loads 44GB offline).
