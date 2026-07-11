> Verified 2026-07-05 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Campaign Playbooks

Three decision-tree templates, each mined from a campaign that actually ran. Every step states
the **Do** (command/recipe), the **Expect** (output), and the **GATE** (condition to proceed).
Obey `laws.md` throughout — the playbooks are how the laws are executed. Golden env for ALL serve
and probe runs: `MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000` (`tasks/ultra_speed.md`
headline). Python for standalone probes: `.venv/bin/python` from the repo root — never bare
`uv run` (Law 11).

| Playbook | Use when | Validated by |
|---|---|---|
| A — Speed campaign | make an already-correct model decode faster | Nemotron-Ultra 7.68→13.08 tok/s (+70%), `tasks/todo.md:1067` |
| B — Conversion campaign | bring a new NVFP4/quantized checkpoint onto MLX | Super→Ultra oQNVFP4→DQ8, template for GLM-5.2-NVFP4, `tasks/todo.md:1104` |
| C — Regression diagnosis | a serving number/quality regressed or a fast path "does nothing" | M3 fused_index dead-path, DQ8 stage-B silent death |

---

## Playbook A — Speed campaign

**Root decision (compute FIRST, then branch):** after Steps 1-2 you know decode's % of the
per-tensor-dtype ceiling.
```
decode tok/s vs per-tensor-dtype ceiling (Step 2)?
  ├─ ≥ ~90% of ceiling  → byte-cutting (quant, Step 3-4) or a fused mega-kernel is the ONLY lever;
  │                        host-serialization work is dead here.
  └─ < ~90% of ceiling  → host serialization is ALSO on the table (op-count cut + chunked async_eval),
                           stacked ON TOP of quant — but measure in-stream first: Ultra was 74% yet
                           host work proved already-overlapped (dead) and quant carried it (Law 2).
                           GLM/M3 sit here (GLM 53.6% / M3 52% GPU-wait, `tasks/overlap_levers.md:3`).
```

### Step 1 — Baseline + host-side profile
- **Do:** record baseline decode tok/s at short/5k/16k with a **stream-free** T1/T256 probe (never
  trust streaming TTFT — Law of `profiling.md`). Then take two 10s captures during sustained
  decode: `/usr/bin/sample <server-pid> 10` (no sudo), find the decode thread, bucket its tree into
  cv-wait / Metal-encode / Python-forward. Recipe + worked trees in `profiling.md`.
- **Expect:** a baseline number (Ultra **7.68 tok/s = 130.2 ms/token**, `tasks/ultra_speed.md`
  headline) and a host split (Ultra **46.5% cv-wait / 35.5% encode / 17.1% python**,
  `tasks/ultra_speed.md §1`; GLM **53.6 / 17.5 / 12.5**, `tasks/todo.md:1014`).
- **GATE:** baseline + host split recorded before ANY lever is touched (Law 10).

### Step 2 — Per-tensor-dtype cost model
- **Do:** list every active tensor group with its actual serving dtype (source dtype × converter
  action), sum bytes/token, divide by 819GB/s → ideal ms/token → ceiling tok/s. Note % of ceiling.
  Source the dtype split from `hf_quant_config` `exclude_modules` + safetensors shard headers (Law 3).
- **Expect:** a per-class byte table + ceiling (Ultra **79.0GB/token → 10.36-10.48 tok/s**,
  `tasks/ultra_speed.md §2`).
- **GATE:** ceiling computed; branch on the root decision above. NEVER estimate from blended bpw
  (Law 3 — the 26 tok/s Ultra error).

### Step 3 — Microbench gates (offline, no model load)
- **Do:** per module class, synthetic-weight probe (generate uint32 codes / uint8 scales directly —
  never `mx.quantize` a multi-GB tensor), ≥12 reps + warm-up eval + median, activations in the
  **live dtype (bf16)**, under golden env, and measure BOTH standalone AND `mx.depends`-chained
  ×N-layers in-stream with realistic neighbors (Law 2). Scaffolds: `tasks/ultra_speed.md §3`;
  sanitized runners in `scripts/` (cross-ref `profiling.md`).
- **Expect:** per-class ratio `bf16_time / q_time` + a weighted model-level ms/token saving
  (Ultra M0: aggregate **47.2ms/token**, mode **affine8-gs64**, `tasks/ultra_speed.md §3 MEASURED`).
- **GATE:** weighted projected saving ≥ your model-level bar (Ultra RC5: **≥25 ms/token**, i.e.
  per-shape `bf16_time/q_time ≥ 1.6`) → GO. **Exclude any class that measures a LOSS** (Ultra attn
  k/v = 0.87-0.91× → excluded). Project live by applying the ~50-60% in-stream discount (Law 2).

### Step 4 — Staged live ladder (one lever per restart)
- **Do:** baseline leg first; then add ONE lever per restart, each behind its own env kill-switch,
  via a single chained background restart command ending in the serve relaunch (OPS contract,
  Law 10; choreography in `ops-runbook.md`). Per leg: grep the expected-vs-actual engagement census
  (hard-fail on mismatch), then stream-free tok/s at short/5k/16k, then gsm8k smoke.
- **Expect:** monotone ladder with engagement proof each rung (Ultra: leg0 **8.02** → +mamba 96/96
  **10.38** → +moedense 192/192 **12.52** → +attn q/o 24/24 **12.85** → +lmhead 1/1 **13.07-13.08**,
  resident −28.5GB, quality held; `tasks/todo.md:1067-1074`).
- **GATE:** per leg, engagement `actual==expected` AND gsm8k smoke holds BEFORE the tok/s is
  believed (Laws 1, 10). Write every leg to `tasks/todo.md` with kill-switch states. A lever whose
  isolated win evaporates in-stream (Ultra sorted-routes 3.5ms→flat) is KEPT if free + kill-switched
  but claims no speed (Law 2).

### Step 5 — Productize (bake the winner offline)
- **Do:** bake the winning transform into an offline checkpoint under a **NEW model name** (Law 9);
  the upstream loader then quantizes-before-load_weights (no boot-time pass, no transient peak).
  Re-run the offline gate battery (census, bit-parity vs `mx.quantize`, real-path load) then swap
  (Playbook B Step 6). Ultra DQ8 bake: `--dq8`, shared `DQ8_STAGES` map imported by BOTH patch and
  converter (is-identity tested). `tasks/ultra_speed.md §4 P1-3`; `tasks/todo.md:1092-1104`.
- **Expect:** live == load-time gates on the new name (Ultra: resident 305.08==305.07, decode
  13.09/13.07 == 13.07/13.08, gsm8k 96.67==96.67, `tasks/todo.md:1100-1103`).
- **GATE:** offline bit-parity EXACT + live-equals-load-time before deleting the source checkpoint.

---

## Playbook B — Conversion campaign (NVFP4 / DQ8 onto MLX)

### Step 1 — Scout the per-tensor dtype split FIRST
- **Do:** read the source `hf_quant_config` `exclude_modules` + safetensors shard headers before
  writing ANY converter code. This is both the bandwidth map (Law 3) and the sensitivity map (Law 4).
- **Expect:** the vendor's quantization map (Ultra: routed experts → nvfp4; mamba in/out_proj →
  FP8; latent/o/q/k/v/shared/router/embeddings/lm_head → bf16-excluded, `tasks/ultra_speed.md §2`).
- **GATE:** split fully enumerated before code; it dictates what you repack vs upcast vs leave.

### Step 2 — Pathfinder on a small sibling model
- **Do:** convert the SMALL sibling end-to-end first and serve it, proving the whole chain on
  cheap iterations. `tasks/todo.md:1028-1036` (Super-120B before Ultra-550B).
- **Expect:** repack bit-exact (**max|diff|=0.0**), ts-fold algebraically exact, live serve +
  quality TIE vs the prior quant (Super gsm8k **92/92** serial, mmlu 81/80, arc 95/94).
- **GATE:** pathfinder green before touching the large model; a 550B mistake costs hours per loop.

### Step 3 — Converter build
- **Do:** byte-repack NVFP4 (U8 codes → U32 view, E4M3 scales byte-copy — bit-exact, GENERALIZES
  across models); fold ts sidecars algebraically (relu² degree-2 → a single per-expert scalar
  `fc1_ts²·fc2_ts` into router scores, EXACT; SwiGLU → output-side elementwise fold); assert ts>0
  at convert. Fix config-dialect traps in the converter too. Cross-ref `conversion.md` for the
  algebra. `tasks/todo.md:1042-1048`; `tasks/lessons.md §Nemotron oQNVFP4 converter`.
- **Expect:** N shards / M tensors / K ts sidecars, positivity green on ALL MoE layers (Ultra:
  1119 tensors / 96 ts sidecars, 48/48 layers green, `tasks/todo.md:1043`).
- **GATE:** positivity green + tensor counts match the arithmetic (`=1119+313×2` for the DQ8 bake).

### Step 4 — Offline gate battery
- **Do:** (D1) tensor census — expected q8 triples + **0 strays** + experts/ts intact; (D2)
  **bit-parity EXACT** vs `mx.quantize` of the source bf16 (proves pipeline determinism); (D4)
  real-path load through the **ENGINE** (`maybe_apply_pre_load_patches` → `mlx_lm.load`), NOT raw
  `mlx_vlm.load_model` (Law 5). Watch config-dialect traps: `layers_block_type` without
  `num_hidden_layers` → mlx_lm ModelArgs positional TypeError; tagged-infinity `time_step_limit`
  → `mx.clip` ValueError (both fixed on-disk AND in converter). `tasks/todo.md:1044-1047,1096-1098`.
- **Expect:** census N/N (Ultra DQ8: 313/313 q8 triples, 0 strays), bit-parity EXACT, load PASS.
- **GATE:** all three green through the real engine path before serving (Law 5).

### Step 5 — Live parity legs
- **Do:** serve the checkpoint; run quality SERIALLY (gsm8k / mmlu / arc, non-stream, temp0) and
  decode speed stream-free. `tasks/todo.md:1052-1053`.
- **Expect:** quality tie-or-lift with no quant damage (Ultra vs Super: mmlu +4.3pp, arc +1.7pp,
  gsm8k tie — the 550B lift with no quant damage).
- **GATE:** parity holds (Law 7 — reproduce any surprising gap serially before blaming the model).

### Step 6 — Swap (NEW NAME)
- **Do:** **bounce the server BEFORE `rm`** (APFS mmap holds the file's space while the live server
  maps it); copy the new checkpoint to internal SSD (never serve from USB — Metal timeout #2098);
  serve under a NEW name (Law 9); discovery restart. Full trap list in `ops-runbook.md`.
- **Expect:** live gates == offline gates on the new name (Ultra DQ8 all `==`, `tasks/todo.md:1100-1103`).
- **GATE:** new-name live == offline before archiving/deleting the source (Ultra: NVFP4 master
  archived to T7 only after full regate).

---

## Playbook C — Regression diagnosis

Use when a served tok/s or quality number moved, or a shipped fast path "does nothing."

### Step 1 — Preflight
- **Do:** run `scripts/preflight.py` (prose in `preflight.md`): golden env set, `sysctl
  iogpu.wired_limit_mb` raised, **MLX/nanobind version in the LIVE venv (loaded lib, not on-disk)**
  (catches a uv-sync wheel revert, Law 11), `~/.omlx/model_settings.json` audit (**force_sampling** can
  silently kill temp0 gates), no-USB-serving. (Engagement/census is Step 2 — preflight is serverless and
  does NOT grep the running server.)
- **Expect:** PASS per item + a one-line antidote pointer on any FAIL.
- **GATE:** environment sane before you suspect the model or a kernel — most "regressions" are a
  reverted wheel or an unset golden env (Law 11; K1's 13.35× artifact, Law 1).

### Step 2 — Engagement greps
- **Do:** grep the live server log for the fast-path census — `[M3CENSUS]` (via
  `OMLX_M3_DEBUG_PATH=N`) `fused_hit` vs `fused_none`; `[ULTRA-DQ8] expected=N actual=N`.
  `tasks/lessons.md §fused kernel`; `omlx-live-path-verification.md`.
- **Expect:** engaged-path counts per step.
- **GATE:** if the path shows **all fallback** (e.g. `fused_none=57/57`) you have found the
  regression — a silently-dead gate (fp16-gate-vs-bf16 class, `tasks/todo.md:350-353`). Fix the
  gate, don't chase the model (Law 1).

### Step 3 — A/B via env kill-switches
- **Do:** one restart per switch (OPS contract), kill-switch OFF (old behavior) vs ON, same prompt,
  stream-free probe, engagement grep on EACH leg. If the symptom is a quality gap, **reproduce it
  SERIALLY** before attributing it to weights/quant.
- **Expect:** a delta attributable to exactly ONE switch.
- **GATE:** attribution is unambiguous AND survives a serial repro — a concurrency-only FAIL is a
  serving artifact, not the model (oQ4e SpecPrefill gsm8k 92%→15% under 6-way concurrency,
  `tasks/todo.md:1037-1039`; Laws 7, 10).

### Step 4 — Bisect the stage
- **Do:** for a timing regression, split any async "submit" number at the `async_eval` boundary —
  host-build vs drain (Law 6); for a multi-stage transform, flip each kill-switch one at a time
  against the composed offline model (Ultra M6) to isolate the guilty stage.
- **Expect:** the stage carrying the cost/bug is named (build-constant + drain-grows = bandwidth,
  not host; a stage whose flip flips the symptom = that stage).
- **GATE:** mechanism identified before proposing a fix (the build/drain split killed the compile
  campaign before it was funded — `tasks/lessons.md §EAGLE temp1 METHOD TRAP`; the stage-B flip
  reproduced the DQ8 silent death — `tasks/todo.md:1084-1085`).
