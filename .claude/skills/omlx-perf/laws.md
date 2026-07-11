> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Campaign Laws

Iron invariants earned by INCIDENT across the GLM-5.2, MiniMax-M3, and Nemotron-Super/Ultra
performance campaigns. Each was proven by a specific failure; violate one and you re-pay its
debugging cost. Citations point to the ledger that recorded the incident (`file §heading` for
docs, `file:symbol` for code). Sibling detail: `dead-levers.md` (verdicts), `gotchas.md` (trap
museum), `profiling.md` (engagement-counter standards), `experiments/` (EXP registry).

| # | Law (one line) |
|---|-----|
| 1 | A gated fast path needs LIVE engagement evidence before any benchmark number is believed. |
| 2 | Isolated microbench wins take the in-stream discount — only ~50-60% survives; only in-stream serve legs count. |
| 3 | Bandwidth napkin math uses the per-tensor dtype split, NEVER a blended bits-per-weight. |
| 4 | Quant-first: serve tensors in source-precision containers; fix kernels, don't requantize the vendor's weights. |
| 5 | Composed-reality testing: test the fully patched/swapped model, not stock classes. |
| 6 | Never time an async GPU submit as one wall number (split host-build vs async_eval drain). |
| 7 | Single-request PASS ≠ multi-request correct (state reuse + concurrency artifacts). |
| 8 | Verify on-metal, not from tests: green units prove neither speed nor engagement. |
| 9 | Changed weights = NEW MODEL NAME (the SSD prefix cache survives restarts and poisons on name reuse). |
| 10 | Serve-window discipline: baseline first, one lever per restart, engagement grep before believing tok/s. |
| 11 | Installed ≠ engaged: verify the artifact is loaded in the LIVE process, not just present on disk. |
| 12 | Per-instance `__class__` swap only — never rebind a method at class scope in a multi-model process. |
| 13 | Never time a first request against a fresh server — model load hides behind a healthy `/health`. |
| 14 | Nonce every temp-0 probe request — the server replays exact-duplicate requests instantly. |
| 15 | T1/T256 subtraction is invalid across cache restores — continuous single-request GEN timing only. |
| 16 | Temp-0 byte-identity gates hold only within ONE forward width on quantized models (qmv vs qmm). |
| 17 | HARNESS LAW: name the harness beside every ms/token number; never borrow across harnesses. |
| 18 | Pipelined stability is its own gate — eager survival proves nothing about async pipelined decode. |
| 19 | (extends 2) Eager-measured overhead pools are already pipeline-hidden — size wins vs the PIPELINED baseline. |

---

## Law 1 — A gated fast path needs LIVE engagement evidence before any number is believed

- **WHY:** live serving differs from a probe in dtype (bf16 vs fp16), cache class (batch vs
  singleton), masks, chunked prefill, and stacked server patches. Gates (shape/dtype/flag)
  **fail closed and silent** — the slow fallback runs and looks like the model is just slow.
  `omlx-live-path-verification.md` (the canonical writeup).
- **Proof (3 incarnations):** (a) M3 `fused_index.py` required fp16 while the live model runs
  bf16 (`torch_dtype`) → 100% silent fallback, `fused_none=57/57` every step — that fallback
  WAS the "pathological" decode slope. `tasks/lessons.md §2026-07-03: fused kernel dead-on-arrival`.
  (b) M3's sparse attention silently fell to **slow masked-dense** live because the compact gate
  (`original_mask is None, B==1`) never fires once the scheduler passes a physical batch mask —
  894µs/layer vs 620 dense / 741 compact at 3.7k — a live-vs-probe gate miss (alongside proof (a)'s
  fp16 gate) behind the M3 context-sag. (The earlier "M3
  sparse disabled on ALL layers / plain KVCache" theory was a layer-0 debug artifact and was
  **REFUTED** — fresh requests DO get `MiniMaxM3BatchKVCache`; sparse WAS running, just masked-dense.
  `tasks/todo.md:382-384`; `memory/omlx-glm52-decode-opts.md`; `kv-cache.md`.)
  (c) K1 expert bench first read **13.35× ideal** — pure harness artifact from a missing golden
  env; under golden env it was 1.87×. `tasks/ultra_speed.md §v2.2 changelog`; `tasks/todo.md:1079`.
- **Comply:** every gated fast path ships an engagement counter logged at INFO and greppable
  (M3 `[M3CENSUS]` via `OMLX_M3_DEBUG_PATH=N` with `fused_hit`/`fused_none`; Ultra `[ULTRA-DQ8]
  expected=N actual=N`, **hard-fail on mismatch — never half-engage**, RC8). Grep engagement and
  confirm `actual==expected` BEFORE reading tok/s. Standalone repros must copy the live config
  (read `torch_dtype` from `config.json`; use the scheduler's batch cache class), never assume
  fp16. `tasks/lessons.md §fused kernel` (RULEs); standards in `profiling.md`.

## Law 2 — Isolated microbench wins take the in-stream discount; only in-stream serve legs count

- **WHY:** an eval-per-op harness inflates every op vs a pipelined decode; measured, only
  **~50-60% of a naive standalone delta survives** `mx.depends`-chained in-stream.
  `tasks/lessons.md §Lessons` (line: "savings that survive pipelining are ~50-60%").
- **Proof:** the pre-sorted-expert-index lever showed **3.5ms/token isolated** yet went **flat
  in-stream** — leg0 stayed 8.02 tok/s ("sort lever's isolated 3.5ms did NOT survive in-stream").
  `tasks/todo.md:1071-1072`. Same trap produced the retracted "MoE at 60% bandwidth" GLM claim.
  `tasks/lessons.md §Lessons` (line 6-9). Ultra experts looked like a 24ms/token prize in
  isolation (38% of peak) but K1 in-stream attributed only **2.41ms** as expert-specific.
  `tasks/ultra_speed.md §3 M1 Finding 2`; `§4 P0-K`.
- **Comply:** measure every candidate delta **in-stream** — `mx.depends`-chained ×N layers with
  realistic neighbors — before it counts as evidence; never fund a kernel off a standalone number
  (K1 rule: "in-stream attribution BEFORE any kernel is written"). `tasks/ultra_speed.md §3` rules.
  Cross-ref `profiling.md` for the probe harness.

## Law 3 — Bandwidth napkin math uses the per-tensor dtype split, NEVER a blended bits-per-weight

- **WHY:** a blended bpw hides mixed-precision reads; the dense bf16 shell of a "4-bit" model
  dominates traffic and is invisible to bpw math. `tasks/lessons.md §Bandwidth napkin math`.
- **Proof:** Ultra ceiling was estimated at ~26 tok/s from "55B active × 4.5 bpw" — **wrong by
  2.5×**. Real per-token reads = **79GB** (66GB dense bf16 + 12.5GB nvfp4 experts) → ceiling
  **10.4 tok/s**, and we were already at **74%** of it. ModelOpt NVFP4 quantizes ONLY routed
  experts; mamba/attention/shared/latent/lm_head ship FP8/bf16. `tasks/lessons.md §Bandwidth`;
  `tasks/ultra_speed.md §2 Bytes read per token` (78.123GB pure weights table).
- **Comply:** before any bandwidth estimate, list EVERY active tensor group with its actual
  serving dtype (source-checkpoint dtype × what your converter did to it) and sum bytes.
  `hf_quant_config` `exclude_modules` + safetensors shard headers give the split in minutes.
  `tasks/lessons.md §Bandwidth napkin math`.

## Law 4 — Quant-first: serve in source-precision containers; fix kernels, don't requantize vendor weights

- **WHY:** carrying an FP8 value inside a bf16 container **doubles its bandwidth for zero quality
  gain**; the vendor's own quantization exclusion list is a ready-made sensitivity map.
  `tasks/lessons.md §Bandwidth napkin math` (COROLLARY); `tasks/ultra_speed.md §v2 Captain doctrine (a)(b)`.
- **Proof:** on Ultra, mamba `in_proj`/`out_proj` were shipped FP8 by NVIDIA but our converter
  upcast them to bf16 — a **40.467GB/token upcast tax**. DQ8 stage A just removes OUR OWN tax
  (NVIDIA already validated FP8 there). Requantizing the experts was rejected twice: `P1-1`
  demoted by doctrine, then DELETED when M1 measured nvfp4/affine4/mxfp4 within **±4%** (lossy-on-
  lossy for nothing). `tasks/ultra_speed.md §2 NVIDIA quantization map`; `§4 stage A`; `§4 P1-1 REMOVED`.
- **Comply:** follow `hf_quant_config` `exclude_modules` as the sensitivity map; give each
  vendor-excluded class its own kill-switch + isolated quality A/B (never a blanket squash);
  never requant the vendor's quantized tensors — attack the KERNEL instead. `tasks/ultra_speed.md
  §v2 doctrine (a)-(c)`; `§6 non-goals`.

## Law 5 — Composed-reality testing: test the fully patched/swapped model, not stock classes

- **WHY:** omlx's engine stacks EXTRA patches the raw loader never runs, and filters written
  against class NAMES miss classes swapped via `__class__`. A green raw-loader test proves
  nothing about the served model. `tasks/lessons.md §two MoE sanitize layers`.
- **Proof:** (a) DQ8 stage B was **silently dead** under the `_TsFoldMoE` `__class__` swap — a
  type-NAME filter never matched, and a `0==0` blind hard-fail "passed"; caught only when a
  reviewer tested the composed/swapped model and REPRODUCED it. `tasks/todo.md:1084-1085`.
  (b) oQNVFP4 load: a truncated `mlx_vlm.utils.load_model` PASSED but the live server load
  CRASHED — the engine adds `_force_minimax_m3_moe_sanitize_on_load`; there are **TWO** MoE
  sanitize layers and the raw path ran only one. `tasks/lessons.md §two MoE sanitize layers`.
- **Comply:** verify through the REAL engine path (`maybe_apply_pre_load_patches` → `mlx_lm.load`),
  not a raw `mlx_vlm.load_model` or a module-scope monkeypatch. Ship a composed offline model
  (Ultra's M6: all patches stacked, engagement counters == expected, then flip each kill-switch).
  Filter modules on structural signals / `block_type`, **never on class names**. `tasks/lessons.md
  §two MoE sanitize`; `tasks/ultra_speed.md §3 M6-composed`.

## Law 6 — Never time an async GPU submit as one wall number

- **WHY:** a single "submit" timer wrapped around an async op **conflates** host graph
  construction with the `async_eval` drain/backpressure (command-buffer semantics). The drain is
  bandwidth-bound GPU time masquerading as host cost. `tasks/lessons.md §EAGLE temp1 METHOD TRAP`.
- **Proof:** `verify_submit ≈ 44ms` looked like un-overlapped HOST graph-build → "compile the
  verify graph → 1.85×." **REFUTED:** the 44ms was a mode-1 LUMPING artifact = host-build **12ms**
  + async_eval drain **32ms**; the drain is bandwidth-bound, not host. This discriminator killed
  the compile campaign before it was funded. `tasks/lessons.md §2026-07-04 EAGLE-3 AMENDED 2026-07-05`.
- **Comply:** ALWAYS split timing at the `async_eval` boundary — time construction BEFORE the
  `async_eval` (build) separately from after (drain). If build is small and constant across
  context while drain grows with context, the cost is bandwidth (a kernel/compile can't touch it).
  `tasks/lessons.md §EAGLE temp1` (METHOD TRAP). See also `mlx.md` (command-buffer mechanics).

## Law 7 — Single-request PASS ≠ multi-request correct

- **WHY:** persistent/cached state keyed on capacity or shape can silently reuse a prior request's
  data, and a serving-path bug can appear ONLY under concurrency. Long-unique test prompts hide
  both. `tasks/lessons.md §single-request PASS ≠ multi-request correct`.
- **Proof:** (a) compiled decode kept per-bucket KV buffers reseeded only on bucket GROWTH; a
  shorter next request reusing the bucket **decoded on the PRIOR request's KV**. NIAH passed
  (unique long prompts always grew fresh buckets); only a temp0 token-identity gate on a REUSED
  bucket caught it. `tasks/lessons.md §single-request PASS`. (b) oQ4e SpecPrefill under **6-way
  concurrency** collapsed gsm8k **92%→15%** (serial fine) — **STILL AN OPEN production bug**.
  `tasks/todo.md:1037-1039`.
- **Comply:** any state keyed on capacity/shape gets an explicit new-request test — request A,
  then a SHORTER B in the same bucket with different content — offline AND live. Before blaming a
  surprising A/B quality gap on the model/quant, reproduce it **serially** (a concurrency FAIL can
  be the artifact — the 15% gsm8k was SpecPrefill, not quantization). `tasks/lessons.md §single-
  request PASS`; `§Nemotron converter` (serial re-run rule). Cross-ref `future-campaigns.md`
  (SpecPrefill concurrency bug — production risk).

## Law 8 — Verify on-metal, not from tests: green units prove neither speed nor engagement

- **WHY:** integration glue (interface/return-shape contracts, `mx.array` truthiness, out-param
  vs return) and live engagement only surface when you RUN the real path. `tasks/lessons.md
  §EAGLE-3 — live-path law (again)`.
- **Proof:** EAGLE-3 wiring passed **two static adversarial reviews + AST + unit tests**, yet
  **three runtime bugs** appeared only on running the server/harness: bare logits vs full output
  with `.hidden_states`; `if not hidden_states:` truth-testing a multi-element `mx.array` →
  "length-1 arrays only" ValueError; verify expecting a returned `.hidden_states` while M3 uses an
  out-param hidden-sink. `tasks/lessons.md §EAGLE-3`.
- **Comply:** drive the real server/harness path before declaring done — "a run is worth more than
  a review" for integration glue. Green unit tests are necessary, never sufficient; pair them with
  Law 1's live engagement grep. `tasks/lessons.md §EAGLE-3`.

## Law 9 — Changed weights = NEW MODEL NAME

- **WHY:** the SSD/prefix cache stores blocks keyed by prompt hash and **survives server
  restarts**; reusing a name after a weight change points the cache at logits the new weights no
  longer produce → poisoned reuse. `tasks/todo.md:1102-1103`; hot/cold detail in `kv-cache.md`.
- **Proof:** the DQ8 productization served as **`Nemotron-3-Ultra-oQNVFP4-dq8`** — "new name
  deliberate: avoids stale SSD-cache poisoning." `tasks/todo.md:1102-1104`. (Corollary: cross-model
  cache blocks are incompatible by design — "GLM SSD-cache blocks incompatible with Kimi."
  `tasks/todo.md:261`.)
- **Comply:** any weight change (quantize, requant, re-convert, DQ8 bake) → new serving name/dir;
  never reuse a name across a weight change. Live gates should match the offline gates exactly on
  the new name (Ultra: resident 305.08 == 305.07, decode 13.09/13.07 == 13.07/13.08, gsm8k 96.67
  == 96.67). `tasks/todo.md:1092-1104`. Cross-ref `ops-runbook.md` (swap procedure, mmap trap).

## Law 10 — Serve-window discipline: baseline first, one lever per restart, engagement grep before tok/s

- **WHY:** multiple levers per restart makes deltas unattributable; without a baseline you have no
  delta; without an engagement grep the number may come from a dead path (Law 1).
  `tasks/ultra_speed.md §8 Execution order & telemetry`.
- **Proof:** the Ultra +70% ladder ran exactly this way — each rung engagement-verified with
  stream-free T1/T256 probes, short==5k at every rung: **leg0** serving fixes + sorted routes
  **8.02** → **leg1** +DQ8 mamba 96/96 **10.38** (−17.7GB) → **leg2** +moedense 192/192 **12.52**
  (−26.1GB) → **leg3** +attn q/o 24/24 **12.85** → **leg4** +lmhead 1/1 **13.07-13.08** (305.07GB).
  `tasks/todo.md:1067-1074`.
- **Comply:** OPS contract — stopped-server work is ONE chained background command ending in the
  serve relaunch (never leave the box serving nothing). Baseline leg first; one lever per restart;
  each behind its own env kill-switch; grep the expected-vs-actual census and confirm quality
  smoke (gsm8k) before reading tok/s; write every leg to the `tasks/todo.md` ledger with
  kill-switch states. `tasks/lessons.md §EAGLE-3 OPS CONTRACT`; `tasks/ultra_speed.md §8`.
  Full choreography in `ops-runbook.md`.

## Law 11 — Installed ≠ engaged: verify the artifact is loaded in the LIVE process, not on disk

- **WHY:** `uv run` re-syncs the venv against `uv.lock` on **every** invocation, so a pip-installed
  patched wheel (e.g. a custom mlx-metal) is silently reverted to the locked version at the next
  `uv run omlx serve`. Present in site-packages ≠ loaded in the running process. `tasks/lessons.md
  §uv run silently reverts patched wheels`; `omlx-live-path-verification.md` (third incarnation).
- **Proof:** a patched mlx-metal wheel was present right after install and gone at serve time
  (mtime snapped back); the "patched run" was measuring stock. `tasks/lessons.md §uv run`.
- **Comply:** grep a patch-marker string in the LOADED lib and check its mtime **in the live venv
  at serve time** BEFORE declaring a patched run; serve with `uv run --no-sync ...` (or install via
  `uv pip` into the synced env). Verify env on the **PARENT** uv process — `ps eww` cannot read the
  renamed `omlx-server` child. `tasks/lessons.md §uv run` (Addendum). `scripts/preflight.py` check 4
  verifies the LOADED mlx/nanobind version (catches a version-level revert); the same-version
  patch-marker/mtime check is manual — see `env-setup.md` for the `--no-sync` discipline.

## Law 12 — Per-instance `__class__` swap only — never rebind a method at class scope

- **WHY:** models share a module class in one process; rebinding `SomeModel.__call__ = ...` at
  module scope silently alters EVERY co-resident model (including the production one). Python
  resolves dunders on the type, so an instance-attribute `__call__` is ignored — the `__class__`
  swap on the target instances is the correct mechanism. `tasks/lessons.md §Nemotron oQNVFP4 converter`.
- **Proof:** the Nemotron ts-fold patch had to swap `mixer.__class__ = _SubClass` on only the
  target model's layers; proven by loading BOTH models in one process and asserting the other's
  class stays stock. Ultra risk register calls this out for co-resident Super/Nano.
  `tasks/lessons.md §Nemotron oQNVFP4 converter`; `tasks/ultra_speed.md §5 risk register`.
- **Comply:** per-instance `__class__` swap on fingerprinted target instances only (self-gate on a
  checkpoint-specific signal such as `fc1_ts` presence + `hidden_size==8192`); never rebind at
  class scope; a co-resident-model regression probe must show ABSENT engagement lines on the other
  model. `tasks/lessons.md §Nemotron converter`; full patch conventions in `omlx.md`.

## Law 13 — Never time a first request against a fresh server
`/health` returns healthy before the engine pool loads the model; the first request silently pays
~40-90s of weight loading. Incident (2026-07-08, EXP-084): a "83s SSD restore" launched a whole
batched-IO campaign — [restore-profile] telemetry then showed the REAL 130k restore is 1.3-1.8s;
the rest was model load + probe conflation. Warm the model with a throwaway request first.

## Law 14 — Nonce every temp-0 probe request
omlx replays byte-identical requests (same model/messages/params) instantly from a response path —
a repeated temp-0 probe returns 400 "generated" tokens in 0.0s. Every probe request must carry a
unique nonce in the prompt tail. Incident: EXP-083's wall-clock legs invalidated twice.

## Law 15 — Continuous GEN timing only; T1/T-N subtraction dies across restores
Subtracting a T1 request's wall from a T-N request's wall assumes identical cache state; a restore,
eviction, or pool swap between them breaks it (ghosts measured: "151 tok/s" @128k fp16, "75 tok/s"
@128k int8, "10.5" with a mid-probe re-restore). Time ONE request's generation continuously
(stream timestamps or wall minus a same-state T1 taken seconds before, verified no restore between).

## Law 16 — Temp-0 byte-identity gates hold only within ONE forward width (qmv vs qmm)

- **WHY:** L=1 decode dispatches quantized matrix-VECTOR (qmv); L>1 verify/chunk dispatches quantized
  matrix-MATRIX (qmm) — the kernels round differently at fp32 ties. No bookkeeping fix can make qmv
  and qmm agree on an exact tie.
- **Proof:** Puzzle oQ48 measured max|logit diff| 0.5 between the two dispatch paths; divergences
  occur ONLY at fp32 ties / margins <= the gap (EXP-097, 2026-07-09).
- **Comply:** for any A/B mixing forward widths (spec-decode verify, chunked legs, MTP): gate on
  token match% >= ~98 AND first-divergence margin (top1-top2 logit gap) <= ~1.0 (tie class), plus a
  task-quality spot. A large-margin divergence is still a real bug. Byte-identity stays valid within
  one width (fusion pool A's 192/192 held because it kept L=1 shapes). `gotchas.md` §qmv/qmm.

## Law 17 — HARNESS LAW: name the harness beside every ms/token number

- **WHY:** different bench scripts measure at different pipeline depths; citing one harness's number
  to judge another's result silently corrupts the comparison.
- **Proof:** EXP-095 — an agent conflated `verify_econ` t1=15.5ms (deep-pipelined, unfused trunk)
  with the fusion campaign's result and wrote "+19%" into the ledger; the real fusion win was +2.9%.
  Caught only by tracing the number back to its source script.
- **Comply:** every timing citation names its script/harness; cross-harness comparisons must be
  re-derived by rerunning ONE harness end-to-end. `profiling.md` §6.

## Law 18 — Pipelined stability is its own gate

- **WHY:** async one-ahead decode exercises Metal command-buffer overlap and in-flight submission
  that an eager standalone repro never touches.
- **Proof:** EXP-095 Puzzle fused-experts pool C: parity PASS + clean eager decode, then intermittent
  Metal command-buffer fault (kIOGPUCommandBufferCallbackErrorInnocentVictim, exit 134) under
  PIPELINED decode only. Killed-unsafe, unwired, banked in `omlx/patches/nemotron_h_puzzle/pool_c.py`.
- **Comply:** every new kernel must survive a sustained pipelined-decode soak before being called
  safe; eager-only testing is insufficient. `dead-levers.md` (pool C row).

## Law 19 (extends Law 2) — Eager overhead pools are already partially pipeline-hidden

- **WHY:** on compute-bound decode the GPU is BUSY, not launch-idle; per-op overhead measured in an
  eager decomposition is partially overlapped by the one-ahead pipeline before any fusion is built.
- **Proof:** EXP-095/098 — Puzzle fusion pool A's eager-measured 6.1ms pool delivered 0.53ms real
  against the pipelined baseline; EXP-098 further refuted "host .item() drains are the cost" (~nil
  gain removing them) and "reject GPU work is free" (compute-bound: unconditional speculation made
  K2 15% slower; trunk L2 verify async 20.2ms vs L1 16.3ms).
- **Comply:** size every fusion/overhead lever against the PIPELINED baseline; treat an eager
  decomposition as an upper bound on the pool, not an estimate of the win.
