# EAGLE AT PRODUCTION TEMP=1 — campaign plan (2026-07-04 late, user: "engage")

## Requirement (user)
MTP must pay at MiniMax's recommended production sampling: temp=1.0, top_p=0.95, force_sampling=true.
temp0-only is insufficient. Exactness of the sampling distribution is non-negotiable.

## Measured context (do not re-derive)
- Live greedy acceptance (temp0, engagement verified via vlm_mtp stats lines): 86.9% @544ctx,
  82.6% @4.8k — drafter is vendor-grade. Artifact warning: fs5 force_sampling=true overrides request
  temp to 1.0 BEFORE the EAGLE temp0 routing guard → drafter never engages unless force lifted.
  EVERY bench leg must show vlm_mtp stats/decode-started lines or its numbers are void.
- BUT net SLOWER through the serial wrapper: 27.6 vs 28.2 short, 20.8 vs 24.3 @~5k (−15%).
  Wrapper = omlx/speculative/vlm_mtp.py round loop (yield int per token, no pipelining) +
  scheduler._step_vlm_mtp (scheduler.py:6606). Round ≈ 88ms @5k vs eager L=1 step 41ms and
  verify L=2 ≈ ~45ms → ~35-40ms/round of overhead to kill.
- Base (golden env): 28.4 short / 24.3 @16k / 18.8 @126k. Batch-1 base decode = GPU-bandwidth-bound
  (proven twice); the MTP wrapper is HOST-bound (the one place host work is on the critical path).
- Adapter: omlx/patches/mlx_vlm_mtp/eagle3_minimax.py (taps, fc fusion, prefix-KV seeding, mxfp8
  drafter default-on). Prefix-KV is MANDATORY for chain acceptance (Gate-2 lesson).
- Models: fs5 = ~/.omlx/models/unigilby/MiniMax-M3-oQNVFP4-fs5; drafter = $OMLX_COLD_STORAGE/
  omlx-quant-work/MiniMax-M3-EAGLE3. Server tmux omlx:4, production line:
  env OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000 uv run omlx serve
- Probe: cd ~/.claude/jobs/62f9cfe9/tmp && M3_MODEL=MiniMax-M3-oQNVFP4-fs5 python3 m3_probe.py <REPS> <MAXTOK> <nonce>
  (~43 tok/rep + ~215; temp0. For temp1 legs the driver must send temperature explicitly.)

## GATE 0 — owner: alpha-gater (BLOCKS workstream A)
Measure expected rejection-sampling acceptance at production sampling, OFFLINE (standalone, server
stopped, ops contract: one chained bg command ending in relaunch+health).
- Generate 3 real fs5 continuations at temp=1.0/top_p=0.95 (~150 tok each, contexts ~500/5k tokens).
- At each position: target logits p, drafter logits q (drafter via the REAL eagle3_minimax adapter
  incl. prefix-KV seeding — do NOT hand-roll the head).
- Apply the SAME processors to both (temp 1.0 → softmax; top_p 0.95 filter+renorm — filter each
  distribution by ITS OWN top-p set, as production would).
- α per position = Σ_x min(p̃(x), q̃(x)) (expected accept prob of standard speculative sampling).
- Report: mean/median/std of α, by depth bucket, n≥300 positions; plus greedy-argmax agreement as
  sanity anchor (should be ~0.83-0.87).
- Thresholds (lead decides on report): α≥0.55 strong GO; 0.45–0.55 marginal (lead call); <0.45 NO-GO.

### GATE-0 OUTCOME (2026-07-05) — PARKED, harness validated, NOT run
NO-GO on the temp1-MTP-at-batch-1 campaign came from WORKSTREAM B (leg-C), not
from an α value: the verify wall is GPU-bandwidth from MoE expert divergence
(verify L=K+1 ≈ (K+1)× expert reads at batch-1) — sampling-independent, so no α
can make temp1 MTP pay at batch-1. The α measurement is therefore deprioritized;
it retains value only as inventory for a future **batch≥2** campaign (expert
reads amortize → spec can pay).
- Harness: `~/.claude/jobs/62f9cfe9/tmp/eagle_gate0_alpha.py` (drafter q via the
  REAL eagle3_minimax adapter; teacher-forced causal forward = prefix-KV seeding
  extended over the full real sequence, chunked; EAGLE shift verified:
  token@t ← consume(tgt_hidden@{t-2}, token@{t-1})). README block at top of file.
- Ops chain: `~/.claude/jobs/62f9cfe9/tmp/gate0_ops.sh` (stop omlx:4 → run →
  ALWAYS relaunch prod line + tee server_m3s.log → poll /health).
- Offline validation ALL GREEN: top_p filter+renorm == production apply_top_p
  (mlx_lm/omlx) to <1.6e-7 (α to <6.6e-8); hand-computed α anchors 0.600000 &
  0.315789 exact; α(p,p)=1.0; compiles; imports resolve.
- PENDING first live run (never executed — auto-mode guard blocks the shared-
  server stop/relaunch; needs user permission or run outside auto mode): the
  greedy-argmax anchor (expect code~0.84-0.87 / math~0.95) and the prefill-seed-
  consistency check. Run via `bash gate0_ops.sh` when the batch≥2 campaign opens.

## WORKSTREAM B — owner: pipeline-builder (independent, start NOW)
Kill the wrapper overhead. Sequence:
1. PROFILE FIRST (standalone or instrumented): decompose the ~88ms round @5k into draft-fwd,
   verify-fwd(L=2), hidden/fc plumbing, per-round syncs (.item()/int()), python glue, per-token
   yield→scheduler roundtrips. Numbers before code.
2. Fix candidates (in expected-value order): (a) eliminate per-round hard syncs — materialize round
   N's tokens while round N+1's draft+verify graph is submitted (async_eval pattern); (b) yield the
   whole round's tokens to the scheduler in one step instead of one scheduler step per token;
   (c) buffer reuse / kill per-round re-allocations and re-seeds; (d) stretch: compiled L=2 verify
   bucket (shelved compile work at vendored compiled_decode.py is the foundation — see
   tasks/overlap_levers.md SHELF).
3. Gates: temp0 token-identity vs CURRENT wrapper (same prompts, engagement verified);
   acceptance stats unchanged (~86.9/82.6 anchors); then live A/B temp0 MTP-on vs base at
   short/5k/16k. Target: MTP-on ≥ 1.3× base at ≥5k ctx with greedy accept.
4. All changes env-gated where risky; no commits; ops contract for server windows (announce to lead).

## WORKSTREAM A — owner: rejection-builder (spawned AFTER Gate-0 GO)
Speculative sampling in the verify step, exactness-preserving, behind OMLX_VLM_MTP_REJECTION=1
(default off): accept draft x~q̃ with min(1, p̃(x)/q̃(x)); on reject resample from norm(max(0, p̃−q̃));
bonus token sampled from p̃ as today. Remove/relax the temp0 routing guard when flag on.
Gates: (i) exactness — statistical distribution test vs base sampling (≥10k tokens, same prompts:
per-position χ²/KL within noise; plus the min(1,p/q) unit math vs a reference implementation);
(ii) live temp1 A/B at production settings (force_sampling untouched) with engagement evidence;
(iii) NIAH@32k + reasoning smoke at temp1; (iv) acceptance telemetry logged per request.

## Ops contract (unchanged from prior campaigns)
One chained bg command for stopped-server work, ending in production relaunch + health probe; never
idle with the server down; announce bench windows to lead; engagement evidence on every MTP leg;
numbers-first reports; no git commits/pushes.

---

## WORKSTREAM B RESULTS + VERDICT (pipeline-builder, 2026-07-05) — NO-GO at batch-1

### Measured (live fs5, temp0, force_sampling lifted, engagement verified every leg via decode-started+stats; OMLX_VLM_MTP_PROFILE)
Per-round ms, block_size=2 (L=2 verify), ~1.83 tok/round, 83-86% accept:
```
leg   ctx    tok/s  verify_build  verify_drain  walk_sync  clear  glue  sched  HOST-build  GPU-wait   period
A(clear=1) 5k  20.9   [lumped verify_submit=44.6]        37.2       7.3   0.4   0.25   -           -          89.8
B(clear=0) 5k  21.3   [lumped verify_submit=48.9]        37.3       0     0.6   0.25   -           -          87.7
C(clear=0) 5k  20.8   11.78         37.89        37.28      0     0.67  0.65   12.48       75.17(85%) 88.31
C(clear=0) 16k 18.4   12.28         46.00        40.99      0     0.72  0.92   13.02       86.99(86%) 100.93
```
Base anchors (golden env): 28.2 short / 24.3 @5k / 24.2 @16k. Token-identity across legs: 5k+16k BYTE-IDENTICAL
(sha 9612467a / 4a43cdcb); short ctx diverges between clear cadences (see tie-flip note).

### VERDICT: MTP/EAGLE cannot beat base decode on M3 MoE at batch-1 (temp0 AND temp1). 1.3x target UNREACHABLE.
Leg-C verify_build/verify_drain split (mode 1, no barrier) is the discriminator:
- **verify_build (pure host graph construction) = ~12ms, CONSTANT across 5k->16k** -> host-build is SMALL and not the wall.
- **GPU-wait (verify_drain async_eval + walk_sync .tolist) = ~85% of the round, GROWS with ctx** -> bandwidth-bound.
- The L=2 verify GPU-wait @5k = 75ms = 1.83x a base single-decode (41ms) and commits 1.83 tokens -> GPU-level
  BREAK-EVEN @5k; @16k verify=87ms=2.12x base, commits 1.85 -> GPU-level LOSS.
- MECHANISM: verify L=(K+1) reads ~(K+1)x the DOMINANT expert weights (distinct experts per token in MoE), while
  dense/attention weights are read ~once. So verify cost scales ~1:1 with committed tokens -> spec decode is CAPPED
  at break-even at batch-1, even at alpha=1. block_size=2 caps at break-even; larger K is the same (block_size=4 in
  the standing lessons.md hit it too). This CONFIRMS + quantifies the old "spec doesn't pay / verify bandwidth-bound"
  lesson; it does NOT overturn it.
- Wrapper HOST overhead is only ~12-13ms/round (~14%): host-build 12 + glue 0.7 + sched 0.25 + (clear 6-9 if on).
  Removing ALL of it -> break-even @5k, still a loss @16k.

### Implication for WORKSTREAM A (rejection sampling): same wall, sampling-INDEPENDENT.
The verify forward costs the same regardless of temp/sampling. temp1 acceptance <= temp0 -> fewer committed tokens for
the same (K+1)x verify cost -> strictly WORSE than temp0's break-even. Even exact rejection sampling can't make MTP pay
on M3 MoE at batch-1. Gate-0's alpha runs into this ceiling regardless of its value.

### lessons.md REVISION CANDIDATE (apply on lead sign-off; AMEND, don't overturn):
Append to the existing "SPEC DECODE DOESN'T PAY ON M3's MoE decode" lesson:
  "Quantified 2026-07-05 (block_size=2, L=2, live profile): the round splits into ~12ms CONSTANT host graph-build +
   ~85% GPU-wait (verify_drain async_eval + walk .tolist), GPU-wait 75ms@5k=1.83x base single-decode / 87ms@16k=2.12x,
   commits ~1.83-1.85 tok -> GPU-level break-even@5k, loss@16k. The (K+1)x expert-weight reads (distinct experts/token)
   cap spec decode at break-even at batch-1 EVEN AT alpha=1; wrapper host overhead is only ~14%, so eliminating it
   (compile the L=2 verify graph) buys at most ~12ms -> break-even, NOT a win. Sampling-independent -> temp1/rejection
   is strictly worse (fewer accepts, same verify cost). Confirmed with the verify_build/verify_drain profiler split;
   the earlier mode-1 'verify_submit~44ms looked like host-build' was a LUMPING artifact (host-build + async drain)."

### SHORT-CTX TIE-FLIP note (clear_cache A/B side-finding):
temp0-greedy output is NOT bitwise-reproducible across clear_cache cadence at SHORT ctx: leg A(clear=1) sha=20632efb
vs leg B(clear=0) sha=f6309ce3, each INTERNALLY deterministic (same-server x2 stable). 5k+16k identical across legs.
Cause: bare per-token mx.clear_cache() perturbs async reduction/dispatch order; tight short-ctx logit margins flip an
early argmax tie and cascade. Rule: gate token-identity at the mission-relevant ctx (>=5k here); don't assume temp0
is bitwise-stable across buffer-management changes at short ctx.

### block_size 4-vs-2 observation:
The standing lesson profiled block_size=4 (L=4 verify, ~3 tok/round). Production fs5 now runs block_size=2 (L=2,
~1.83 tok/round) per model_settings.json (vlm_mtp_draft_block_size=2). The break-even cap holds at BOTH: L=(K+1) verify
cost ~ (K+1)x scales with the tokens committed either way. Smaller K just moves both sides down proportionally.

### B2 compile sketch: SHELVED-BY-LEG-C (not written).
The planned "compiled L=2 verify on top of compiled_decode.py" targets the host graph-build, which leg-C shows is only
~12ms/round -> best case break-even, not the 1.3x target. Not worth the compile-campaign risk (the L=1 path took days +
hit cross-request contamination bugs). Revisit ONLY for batch>=2 (expert reads amortize across the batch = compile's
real regime per overlap_levers SHELF) or a dense target. Profiler (OMLX_VLM_MTP_PROFILE) + clear_cache knob
(OMLX_VLM_MTP_CLEAR_EVERY) stay in-tree, env-gated, default-off/1 (behavior-preserving).
