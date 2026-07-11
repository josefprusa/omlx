# Lessons

- 2026-07-03 (decode-kernel foray): Don't declare a ceiling from standalone
  microbenches or naive floor math. Two concrete mistakes to never repeat:
  (1) my "MoE at 60% bandwidth" claim missed 13MB of scales/biases in the
  floor AND conflated in-stream pipelining with kernel inefficiency —
  standalone gather_qmm was at 83%; (2) "fused router" and "split+swiglu"
  targets were already fused by mx.compile in production — read the imported
  code (mlx_lm swiglu is @mx.compile'd) before estimating savings.
- Measure the DELTA in-stream (chained with mx.depends) not standalone; the
  savings that survive pipelining are ~50-60% of the naive sum.
- MLX matmul/qmv kernels are near-peak; real decode headroom is dispatch glue
  and shape-specialized paths (M=1 batched heads -> gather_qmm trick).
- A losing prototype is data, not failure: try the free/native alternative
  first (gather_qmm swap beat my hand-written multi-head qmv kernel).

## 2026-07-03: fused kernel dead-on-arrival in live serving (dtype gate)
- fused_index.py required fp16; live M3 runs bf16 (torch_dtype). Standalone
  verification used set_dtype(float16) -> "bit-exact, engages" was true ONLY
  offline. Live: 100% silent fallback (fused_none=57/57 every step) doing
  full-cache fp32 astype+matmul glue -> the entire anomalous decode slope.
- RULE: any fast-path with a gate (shape/dtype/flag) MUST ship with an
  engagement counter visible in live logs. "Verified standalone" is not
  "engaged live". Census counters (OMLX_M3_DEBUG_PATH=N) now exist — reuse.
- RULE: standalone repros must copy the LIVE dtype/config: read config.json
  torch_dtype, don't assume fp16.

## 2026-07-04: two MoE sanitize layers; test via the ENGINE load, not raw mlx_vlm
- oQ-NVFP4: a truncated `mlx_vlm.utils.load_model` load PASSED, but the live
  server load CRASHED. omlx's engine wraps the load with EXTRA patches the raw
  mlx_vlm path never runs — here `_force_minimax_m3_moe_sanitize_on_load`
  (omlx/engine/vlm.py) force-packs shared_experts into switch_mlp. There are TWO
  MoE sanitize layers: the vendored model `_sanitize_moe_weights` AND this
  engine-level `_pack_mlx_unpacked_moe_weights`. I checked only the first.
- RULE: when a spec says "find the sanitize code", grep ALL of omlx/ for the
  behavior (there can be multiple layers), and VERIFY through the real engine/
  server load path — a raw mlx_vlm.load_model repro skips omlx's wrapper patches.

## 2026-07-04: EAGLE-3 on M3 — live-path law (again), ops, and the MoE spec wall
- LIVE-PATH LAW, reinforced 3x: the EAGLE-3 wiring passed TWO static adversarial reviews + AST + unit tests,
  yet 3 runtime bugs only surfaced by actually RUNNING the server/harness:
  (1) VLMModelAdapter returned bare logits (not the full output w/ .hidden_states) when the eagle3 verify passed
      capture_layer_ids WITHOUT return_hidden; (2) `if not hidden_states:` truth-tested a multi-element mx.array
      -> "length-1 arrays only" ValueError; (3) _eagle3_verify_target expected the target to RETURN .hidden_states
      but M3 uses a hidden_sink OUT-param. RULE: for integration glue (interface/return-shape contracts, mx.array
      truthiness, out-param vs return), a run is worth more than a review. Drive the real path before declaring done.
- OPS CONTRACT (lead, binding): stopped-server work MUST be ONE chained bg command ending in the serve relaunch
  (tmux C-c; sleep; <measurement>; tmux send-keys '<serve>' Enter) so the restart never depends on my wake-up.
  Got dinged twice for "server down + idle waiting on a notification." Never leave the box serving nothing.
- codex exec HANGS on "Reading additional input from stdin..." if launched after a heredoc or without the prompt
  attached. ALWAYS `codex exec ... '<prompt>' </dev/null` (or `- < spec.md`). Sanity-check the log grows >1KB in ~1min.
- Enabling a draft model / editing ~/.omlx/model_settings.json on the LIVE server is a persistent production-config
  change -> auto-mode DENIES it (teammate-messages can't establish user consent). Needs the USER to approve. Plan
  standalone (non-production) validation paths that don't require it.
- SPEC DECODE DOESN'T PAY OFF ON M3's MoE decode: the L=K+1 verify forward pays ~full per-token MoE expert-weight
  reads for all K+1 tokens (distinct experts, memory-bound), so committing K tokens/round ~= baseline. Same wall that
  retired the fused-kernel <=14ms/extra-token bar. Verify = ~85% of the round (profiled); draft-side opt caps ~+5-8%.
  Good acceptance (~3 tok/round, math beats vendor) does NOT imply speedup when decode is bandwidth-bound per-token.
  AMENDED 2026-07-05 (EAGLE temp1 campaign, block_size=2/L=2, live fs5 profile, verify_build/verify_drain split):
  CONFIRMED + quantified. Round decomposes into ~12ms CONSTANT host graph-build (context-independent) + ~85% GPU-wait
  (verify_drain async_eval + walk .tolist). GPU-wait 75ms@5k = 1.83x a base single-decode (41ms), 87ms@16k = 2.12x;
  commits ~1.83-1.85 tok -> GPU-level BREAK-EVEN@5k, LOSS@16k. Wrapper host overhead is only ~14% -> removing ALL of
  it (even compiling the verify graph) reaches break-even, never the 1.3x target. Mechanism: verify L=(K+1) reads
  ~(K+1)x the DOMINANT expert weights (distinct experts/token); dense+attn weights read ~once -> verify cost scales
  ~1:1 with committed tokens -> spec decode CAPPED at break-even at batch-1 EVEN AT alpha=1. Holds for block_size 2
  AND 4 (both sides scale with K). SAMPLING-INDEPENDENT corollary: the verify forward costs the same regardless of
  temp/top_p, so temp1 (accept <= temp0) is strictly WORSE -> rejection sampling can't rescue it either. Only escape:
  batch>=2 (expert reads amortize across the batch), a DENSE target, or the residual-2x dispatch-gap/mega-kernel work.
  TWO HYPOTHESES RETRACTED tonight, kept as traps: (i) "the serial wrapper's per-token yield->scheduler roundtrip is
  the cost" (lead) -> REFUTED, sched roundtrip = 0.23-0.28 ms/round. (ii) "verify_submit~44ms is un-overlapped HOST
  graph-build, so compile the verify -> 1.85x" (pipeline-builder) -> REFUTED: that 44ms was a mode-1 LUMPING artifact
  (host-build 12 + async_eval drain 32); the drain is bandwidth-bound GPU, not host. METHOD TRAP: a single mode-1
  "submit" timer over an async op conflates host graph-ctor with the async_eval drain/backpressure -> ALWAYS split by
  timing construction BEFORE the async_eval (verify_build) vs after (verify_drain); if build is small + constant across
  ctx and drain grows with ctx, it's bandwidth, not host. This is the discriminator that killed the compile campaign
  before it was funded.

## 2026-07-04 — uv run silently reverts patched wheels
Pattern: `uv run` re-syncs the venv against uv.lock on EVERY invocation — a pip-installed patched
wheel (e.g. custom mlx-metal) is silently replaced with the locked version at the next
`uv run omlx serve`. Symptom: patch present in site-packages right after install, gone at serve time;
mtime snaps back. Detection: grep a patch-marker string in the installed lib + check mtime BEFORE
declaring a patched run. Fix: `uv run --no-sync ...` (or install via uv pip into the synced env and
avoid re-sync), never edit uv.lock for experiments. This is the third incarnation of the house
live-path lesson (fp16-gate kernels, spy-vs-compiled-region, now uv-sync revert): ALWAYS verify the
artifact is engaged in the live process, not just installed.
Addendum (builder): `ps eww` cannot read the renamed omlx-server CHILD's env — verify the env on the
PARENT uv process, and verify the patch-marker in the LOADED libmlx, not site-packages on disk.

## 2026-07-04 — single-request PASS ≠ multi-request correct (state reuse)
Compiled decode kept per-bucket KV buffers; reseeded only on bucket GROWTH → a shorter next request
reusing the bucket silently decoded on the PRIOR request's KV (cross-request contamination). NIAH
passed (unique long prompts always grew fresh buckets); only a temp0 token-identity gate comparing
against eager on a REUSED bucket caught it. Rule: any persistent/cached state keyed on capacity or
shape needs an explicit new-request test (A then shorter B, same bucket, different content), offline
AND live. Generalizes the live-path lesson: correctness gates must cross request boundaries.

## 2026-07-05 — Nemotron oQNVFP4 converter (M3 → Nemotron-H pathfinder)
- MODEL PATCH THAT COEXISTS WITH OTHER MODELS IN-PROCESS: never rebind the class method
  (`nh.NemotronHMoE.__call__ = ...` at module scope) — it silently alters the PRODUCTION model
  (oQ4e/Nano) sharing the process. Use a PER-INSTANCE `mixer.__class__ = _SubClass` swap on only the
  target model's layers (Python resolves dunder `__call__` on the type, so an instance-attribute
  `__call__` is ignored — `__class__` swap is the correct mechanism). Self-gate the wrap on a
  checkpoint-specific signal (presence of `.fc1_ts` keys), not just model_type. Proven by loading BOTH
  models in one process and asserting the other's class stays stock.
- KILL-SWITCH SEMANTICS must be chosen explicitly, not assumed: "bit-stock" (no patch, drop sidecars)
  vs "debug/attribution" (keep structure, skip only the math). Lead reconciled to register-ALWAYS +
  skip-fold: strict `load_weights` needs the sidecar params in the module tree, so you cannot skip
  registration even when disabled — only the runtime multiply is gated.
- HOOK POINT: mlx_lm 0.31.3 `load_model` order is sanitize → nn.quantize → load_weights, and it binds
  weights AS-STORED (no cast_predicate/set_dtype pass). So `sanitize` (gets self+weights) is the right
  place to register per-instance params, and fp32 sidecars stay fp32 with no cast patch. Confirm the
  loader's cast behavior before assuming you need a cast_predicate change.
- RELU² MLP (gateless, degree-2 homogeneous): `weight_scale_2` folds as a SINGLE per-expert scalar
  `fc1_ts²·fc2_ts` into router scores — exact, and simpler than SwiGLU's split gate/up ts (M3). The
  ModelOpt NVFP4 byte-repack (U8 codes→U32 view, E4M3 scales byte-copy) is bit-exact (max|diff|=0.0)
  and GENERALIZES across models — re-verified on Nemotron with an independent fp4·E4M3·ts reference.
- Verified through the REAL engine path (`maybe_apply_pre_load_patches` → `mlx_lm.load`), not a
  standalone monkeypatch — same lesson as the M3 "test via ENGINE load" entry. Composed cleanly with
  the MTP sanitize patch (Nemotron declares num_nextn_predict_layers=1 → both wrap sanitize).
- CONCURRENT quality-bench number can be a SERVING-PATH ARTIFACT, not the model: acc_bench (6 workers)
  scored the oQ4e baseline 15% on gsm8k while oQNVFP4 got 95% — I distrusted the delta, and a single-
  shot + a 12-Q SERIAL re-run both put oQ4e at 92% (tied). The 15% was oQ4e's SpecPrefill degrading
  under concurrency, NOT quantization. RULE: before attributing a surprising A/B quality gap to the
  model/quantization, reproduce it SERIALLY — the mirror of the "single PASS ≠ multi correct" trap
  (here a concurrent FAIL was the artifact). Short-answer benches (mmlu/arc, 8-token) hid it; only the
  long-generation bench (gsm8k, 512-token) exposed the concurrency path.
- BENCH CONFOUND: compare like-for-like — oQ4e had SpecPrefill (draft=Nano) configured, the fresh
  oQNVFP4 did not, so TTFT wasn't apples-to-apples (decode was — spec-prefill is prefill-side). Note
  per-model serving config (draft/spec/engine) from the load log before reading a speed table.

## Bandwidth napkin math: use the per-tensor dtype split, never blended bpw (2026-07-05)
Estimated Ultra's decode ceiling at ~26 tok/s from "55B active x 4.5bpw" — wrong by 2.5x.
ModelOpt NVFP4 drops quantize ONLY routed experts; mamba/attention/shared/lm_head ship FP8
(which our converter upcast to bf16). Real reads were 79GB/token (66GB dense bf16 + 12.5GB
nvfp4 experts) -> ceiling 10.4 tok/s, and we were already at 74% of it. The planner caught it.
RULE: before any bandwidth estimate, list every active tensor group with its ACTUAL serving
dtype (source checkpoint dtype x what our converter did to it), then sum bytes. hf_quant_config
exclude_modules + safetensors shard headers give the split in minutes.
COROLLARY (captain's quant-first doctrine, same day): serve tensors in (a container matching)
their source precision — carrying FP8 values in bf16 doubles their bandwidth for zero quality.
Follow NVIDIA's own quantization map for what tolerates 8-bit; their exclusion list marks what
they considered sensitive.

## Compaction memory drifts — re-cite from corpus before institutionalizing (2026-07-06)
Building the omlx-perf skill from 12 parallel extractors exposed how much post-compaction "campaign
memory" had silently drifted: enforcer "470GB" (real: 489-496 boot-dependent), "11GB/min" conversion
pace (real: ~7.3), "GLM 18.8@128k" (that number is MiniMax@66-126k), "0.1s decode burst doesn't
exist" (it does — engine_core.py:163, the extractor searched scheduler.py only), plus a RETRACTED
theory (M3 sparse-disabled) nearly enshrined as an iron-law proof. None of the drifted values
survived because every claim required a source citation and a verifier re-opened ~700 of them.
RULE: summaries and memories are leads, not sources. Anything destined for a ledger, skill, or
config must be re-grepped from code/ledgers at write time; negative claims ("X doesn't exist")
need a whole-tree search, not one file.
- 2026-07-07: zsh does NOT word-split unquoted $VARS (unlike bash) — a $FLAGS variable holding "--mtp --kv-bits 8" passes as ONE argv token and argparse rejects it as a single unknown arg (while printing the flags as valid in usage — confusing signature). Use ${=FLAGS} in the Bash tool (zsh) or arrays. Burned EXP-078's k=3 leg and two verification legs before diagnosis.
