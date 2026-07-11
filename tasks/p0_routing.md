# P0 — MoE spec-verify routing measurement (MiniMax-M3-oQNVFP4-fs5)

Goal: go/no-go numbers for the MoE spec-verify roadmap (P1 lossless grouped
verify, P2 top-B shortlist). Measure, during L=4 forwards (== EAGLE K=3
verify), per MoE layer (57) per 4-token window: routed top-4 inds (16 picks)
+ full 128 pre-topk gate scores.

## Plumbing (verified)
- Router: `_minimax_moe_select` (vendored language.py, `@mx.compile`), called
  from `MiniMaxSparseMoeBlock.__call__`. 57 calls/forward = MoE layers 3..59
  (moe_layer_freq first-3-dense); shared expert (idx 128) concat happens AFTER
  the router, so recorded `inds` are pure routed top-4 of 128.
- Config: 60 layers, 128 experts, top-4/tok, scoring=sigmoid, use_routing_bias
  => g_i = sigmoid(gates)+correction_bias. routed_scaling_factor=2.0.
- Instrument: env-gated `OMLX_M3_ROUTE_TRACE=1` recorder INSIDE the router,
  runtime-armed via `language._M3_ROUTE_ACTIVE`; harness `mx.disable_compile()`
  during traced forwards (else append captures compile placeholders). Records
  (raw sigmoid scores[128], biased scores[128], inds[4]). Left env-gated-OFF in
  the vendored file (production zero-cost; P2 reuses the hook).
- Load: `maybe_apply_pre_load_patches(MODEL, for_vlm=True)` (installs vendored
  namespace) -> `mlx_vlm.utils.load_model(Path, trust_remote_code=True)` ->
  `mx.set_wired_limit(506GiB)`. Tokenizer: AutoTokenizer(trust_remote_code),
  chat template rendered as string then `.encode()`.

## Method
Per domain (code/agentic, math, prose; >=150 windows each): greedy-continue
several domain prompts -> re-run the generated region as L=4 chunked forwards
against a fresh cache (prompt prefilled untraced) -> capture 57-layer trace per
window. Deliverables: (a) U=|union of 16 picks| per layer-window + f=(16-U)/12,
agg + early/mid/late bands; (b) top-B shortlist mass + pick-recall for B in
{4,6,8,10,12,16}; (c) per-domain splits. BONUS: gather_qmm sorted_indices at
M=16 vs U.

## De-risked before the expensive run (all on synthetic / no-weights)
- mx->np conversion OK; union + mass + band math validated on synthetic.
- gather_qmm bonus call shape OK; even synthetic shows time drops with U.
- tokenizer chat-template path fixed (Encoding-object gotcha).

## OPS
Server stopped + measured + relaunched in ONE chained bg command (contract).
Harness: tmp/p0_route_measure.py; raw npz + report -> tmp/p0_route/.

## Status
- [x] plumbing traced, instrumentation added, harness written + dry-run green
- [x] live measurement run (chained w/ server relaunch) — MEASURE_EXIT=0, server back up
- [x] analysis + go/no-go report -> SendMessage to main

## RESULTS (2026-07-04, 630 windows: 210/domain, raw npz in tmp/p0_route/)
Report: tmp/p0_route/p0_report.md. Raw arrays: {code,math,prose}.npz
(inds[210,57,4,4], biased/scores[210,57,4,128] f16) — re-analyzable offline.

(a) Union U / f=(16-U)/12 per layer-window:
  ALL  U=11.0 f=0.415  | code U=11.6 f=0.364 | math U=10.8 f=0.432 | prose U=10.6 f=0.447
  overlap GROWS with depth (late layers f~0.45-0.50 > early f~0.30-0.38).
  => code f=0.36 > 0.30 threshold (overlap EXISTS).

(b) BONUS gather_qmm (real fs5 nvfp4 gate_up, M=16=L4 verify shape):
  time = ~177us fixed + ~36us x U(distinct experts). sorted/unsorted = 1.00-1.01x.
  => gather_qmm ALREADY auto-amortizes duplicate-expert reads by U, sorted=no-op.
  Live L=4 verify already does ONE gather_qmm/layer (do_sort=False, 20<64) =>
  the f-overlap is ALREADY captured. P1 = NO incremental win.

(b) mass concentration top-B shortlist (rank=summed biased):
  raw-sigmoid mass (diffuse, sigmoid artifact): code B12=0.30, B16=0.36 (LOW).
  routing-WEIGHT mass preserved (true P2 fidelity): code B8/12/16=0.75/0.835/0.879,
  math 0.81/0.884/0.922, prose 0.82/0.905/0.943. pick-recall code B12=0.80/B16=0.86.
  => code needs B~16 for ~88% (still <90%); B=8-12 loses 16-25% weight mass on code.

GO/NO-GO: P1 NO-GO (already-captured, sorted no-op). P2 NO-GO at low risk for
code B<=12 (~75-84% fidelity); marginal math/prose@B12; code needs B~16 (~88%).
Confirms EAGLE ledger (~1:1 exchange rate, verify already U-amortized).
Only lossless lever left = per-call fixed overhead (177us x 114 calls/window ~20ms)
via MoE kernel fusion (gate_up+down / cross-layer) — NOT P1's grouping.

Instrumentation LEFT in vendored language.py, env-gated OFF (OMLX_M3_ROUTE_TRACE);
runtime arm via language._M3_ROUTE_ACTIVE + mx.disable_compile(). P2 reuses it.
