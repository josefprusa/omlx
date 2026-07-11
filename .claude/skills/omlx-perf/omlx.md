> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# omlx serving-stack map

omlx is an MLX-based, OpenAI-compatible inference server for Apple Silicon. This file is the
orientation a fresh model needs before touching the serve path: how a request flows, what the
scheduler and engine pool do, and the **house rules for fast-path patches** (the part most likely
to bite you). KV/prefix-cache internals live in `kv-cache.md`; kernels in `metal.md`/`mlx.md`;
converter/quant in `conversion.md`; launch env in `env-setup.md`; per-model quirks in `models/*.md`.

Path convention below: `path §heading` for docs, `path:symbol` for code.

## Request lifecycle (one diagram)

```
HTTP (OpenAI/Anthropic API)            omlx/server.py, omlx/api/*
  → resolve model id / alias           EnginePool.resolve_model_id (engine_pool.py)
  → pool.acquire(model_id) [lease]     engine_pool.py:EnginePool.acquire  (async ctx mgr)
      • already loaded → return engine  (updates last_access, in_use += 1)
      • else admit vs memory ceiling → maybe evict LRU → _load_engine
  → BatchedEngine / VLMBatchedEngine   omlx/engine/{batched,vlm}.py
  → EngineCore (per-engine MLX executor thread)
  → Scheduler.step()                   omlx/scheduler.py  (continuous batching, env-gated,
        default OFF: when ON, prefill chunks INTERLEAVE with decode in one step loop)
  → BatchGenerator (sample tokens)     mlx-lm / omlx/patches/mlx_lm_mtp/batch_generator.py
  → cache managers                     paged KV + block-aware prefix cache + tiered SSD store
  → SSE deltas stream back             server.py  (delta.content / delta.reasoning_content)
```

Leasing is atomic under the pool lock (`engine_pool.py:EnginePool.acquire`), so a model cannot be
evicted mid-request even on exception. One process hosts **many** models concurrently; every patch
you write shares that process with unrelated models — hence the per-instance rules below.

## Scheduler (omlx/scheduler.py, ~477 KB — grep, don't read whole)

### Chunked prefill
`config.chunked_prefill` (default **False**, `scheduler.py:SchedulerConfig`) spreads a long prompt
across multiple `step()` calls, one `prefill_step_size` chunk per step, so decode of other requests
is not starved. Resume state lives in `scheduler.py:_PrefillState` (accumulated cache,
`tokens_remaining`, `tokens_processed`, boundary-snapshot bookkeeping). Chunks advance in
`scheduler.py:_step_prefill_chunk` / `_advance_chunked_prefills`; each step processes ONE chunk then
yields the loop so decode and new prefills interleave. Chunk size is set by the adaptive throttle
(below), never a fixed constant.

### Adaptive prefill throttle
Purpose: keep each chunk's **predicted peak** under a safety margin below the hard Metal cap so a
single MoE prefill chunk (tens of MB/token transient) can't trip the *uncatchable async* Metal OOM
(`kIOGPUCommandBufferCallbackError OutOfMemory` → SIGABRT).

- `scheduler.py:_adaptive_chunk_size` — if the full requested chunk fits under
  `target = min(hard_cap*0.90, abort_cap)` it runs unchanged (no effect on healthy traffic);
  otherwise shrink to `headroom/per_token`, floored at `_prefill_min_chunk_tokens`, with a secondary
  discrete-tier clamp (`_PREFILL_STEP_TIERS = (1024, 512)`) once in the caution band.
- `scheduler.py:_predicted_chunk_transient` — per-token cost = `max(measured last-delta/n, EWMA
  bytes_per_token, static SDPA+KV estimate) × 1.3` (`_PREFILL_TRANSIENT_SAFETY`). Anchored on the
  most recent measurement because the transient scales with `query_len × kv_len` (grows with
  context), so a long-run EWMA lags and under-predicts.

**phys_footprint mispredictor incident (benign, once/request)** — `tasks/todo.md §"THREE ULTRA-ONLY
SERVING FINDINGS"`: at chunk 0, Nemotron-3-Ultra's one-time ~120 GB whole-process `phys_footprint`
jump (routed-expert weight *wiring*, a fixed cost) was measured as a per-token delta and charged as
`rate × 2048 × 1.3` → `predicted=155.78 GB`. Prompt length was never in the formula (a 20k prompt
predicted *less* than 6k, pure EWMA decay). Cost was a single pause + no-op eviction + first chunk
`2048→1817`, tens of ms of TTFT. Decode untouched. It is a *predictor* artifact, not a real leak.

**SF-2 clamp fix** — `scheduler.py:_predicted_chunk_transient` clamps the two **tracker-derived**
terms (measured last-delta, EWMA) to `K × static_per_token`, default **K=8** via env
`OMLX_TRANSIENT_CLAMP_K` (`_env_float(..., 8.0)`). The *static* term is never clamped (it is the
trusted floor); `K=0` disables the clamp and reproduces the old unclamped `max`. Emits a greppable
`Clamping prefill transient …` INFO line. Regression tests: `tests/test_scheduler.py:TestPrefillTransientClamp`
(poisoned 58 MB/token measured → clamped to 8× static; static-only path untouched;
`_PREFILL_ABORT_MARGIN`/`_PREFILL_TRANSIENT_SAFETY` asserted unchanged). Result: "zero
adaptive_prefill_throttle noise" (`tasks/todo.md §"SERVING FIXES live"`).

### Store-cache worker + "burst" protection
Post-completion KV extraction and tiered-SSD persistence run OFF the decode step on a **single-worker**
`ThreadPoolExecutor` (`scheduler.py:_store_cache_executor`, built in `Scheduler.start`). Concurrency is
bounded by `scheduler.py:_StoreCacheGate` (`cap` from `config.max_num_seqs`, retuned live by
`ProcessMemoryEnforcer`): the cap bounds concurrent extracted-KV count — the OOM guard for the
**burst-finish RAM growth** in #1383. Backpressure is applied at *admission* (`_schedule_waiting`
declines new prefills while `in_flight >= cap`, #1496), so token generation never stalls on an SSD
write. NOTE: this store-cache `_StoreCacheGate` "burst" (a RAM-growth spike) is a **different**
mechanism from the decode-burst time-slice documented below — do not conflate them.

### SF-1 early cache-index publish (owned by kv-cache.md)
`scheduler.py:_publish_cache_index_metadata` registers prefix-cache block metadata BEFORE the async
SSD persist completes, so an exact-repeat prompt hits the cache immediately instead of waiting ~40s
for the store to commit (Ultra exact-repeat 5k TTFT **40s → 3.0s, 13×**, `tasks/todo.md §"SERVING
FIXES live"`). Kill-switch `OMLX_DISABLE_EARLY_INDEX_PUBLISH=1` (`scheduler.py:_early_index_publish_disabled`);
retraction on persist failure via `_retract_early_cache_index_publish`. Tests:
`tests/test_scheduler.py:TestEarlyCacheIndexPublish`. **Mechanism details → `kv-cache.md`.**

### EngineCore decode-burst budget + batch≥2 toggles
`EngineCore._step_burst` (`omlx/engine_core.py:308-351`, driven by `_engine_loop` :376-378) runs
decode in **time-sliced bursts** — it chains `scheduler.step()` calls until a step yields no output,
`decode_burst_max_steps` (**64**, a host output-list cap, NOT a memory knob) is hit, or a wall-clock
budget expires. The budget is **regime-dependent** (`engine_core.py:331-340`): **1 active request →
0.1s** (`decode_burst_budget_single_s`, aggressive); **≥2 concurrent → 0.03s**
(`decode_burst_budget_s`, tight, to keep admission/abort latency low) — so batch-1 and batch-2 decode
economics genuinely differ. Env `OMLX_DECODE_BURST_BUDGET_SINGLE_S` / `_BUDGET_S` / `_MAX_STEPS`
(defaults 0.1 / 0.03 / 64, `engine_core.py:163-175`); the `settings.py:138-144` Burst-Decode UI mode
(`off|light|balanced|aggressive`) + `admin/routes.py:3383` hot-tune only `max_steps` + the
single-stream budget. (This mechanism is REAL — it supersedes the earlier note that claimed no
decode-burst budget exists.)

**Batch≥2 is OFF by default.** `OMLX_CONTINUOUS_BATCHING` (default **`"false"`**, `omlx/config.py:199`)
gates continuous batching — the lifecycle diagram interleaves prefill/decode only once it is ON.
Concurrency cap: `OMLX_MAX_CONCURRENT_REQUESTS` (alias `OMLX_MAX_NUM_SEQS`, `omlx/settings.py:930`);
**effective default 8** (`SchedulerSettings.max_concurrent_requests`, `settings.py:261`), NOT the 256
raw dataclass default (`scheduler.py:1343`) which the server path overrides (`settings.py:1417`; a
second `SchedulerConfig` at `config.py:89` also defaults to 8 — a footgun). Batch≥2 is
future-campaigns #2's entry condition; production today is single-stream.

## Engine pool (omlx/engine_pool.py)

Manages many model engines with LRU eviction under a memory ceiling. Pre-load admission is in
`engine_pool.py:EnginePool.get_engine`; unload/settle in `_unload_engine`; load in `_load_engine`.

### Admission & eviction
- Ceiling = `enforcer.get_final_ceiling()` (min of static + dynamic caps); `ceiling == 0` ⇒ enforcer
  off ⇒ admit unconditionally.
- `current = max(active_memory − freed, phys_footprint − freed, _current_model_memory)` — the
  `max()` prevents over-commit when a settled model reads low on the live gauge but is still resident
  (**#1623**). `projected = current + estimated_size`; while `projected > ceiling`, evict LRU
  (`_find_lru_victim`, which skips pinned / `in_use>0` / active-request models), else raise
  `ModelTooLargeError` (model alone > ceiling) or `InsufficientMemoryError`.

### Memory accounting: estimated vs actual, and the double-count fix
- `estimated_size` — pre-computed from safetensors headers (+~5% overhead, `model_discovery.py`).
- `actual_size` — observed process-memory delta after load settles: `max(0, post − pre)` where each
  is `max(mx.get_active_memory(), get_phys_footprint())` (`_load_engine`); falls back to
  `estimated_size` if the delta reads 0.
- Log line **`Loaded model: X (actual: A, estimated: E, total: T)`** (`_load_engine`): A = observed
  resident delta, E = safetensors estimate, T = running `_current_model_memory`. A ≫ E hints at
  load-time transients not yet reclaimed; A ≈ E is healthy.
- **Freed-memory double-count fix** (commit `5a26eb1` "don't double-count freed memory in load
  admission"): `phys_footprint`/active gauges LAG a large free, so reloading a model (or evict-then-
  load) counted the freed weights twice and spuriously tripped the ceiling — GLM-5.2 projected
  **642 GB vs the real 464 GB** on a same-model reload. Fix: accumulate `freed_pending` within the
  admission call and subtract it from the lagging live gauges (`get_engine` loop ~L730-772; same fix
  in the prefill-eviction loop `_evict_idle_lru_for_prefill`). The `_current_model_memory`
  accumulator still floors the estimate (#1623 preserved).
- Unload uses a **settle barrier** (`_unload_engine`) polling `mx.get_active_memory()` up to 10
  rounds; under concurrent serving the delta is unmeasurable so it bails ("indeterminate") rather
  than serializing gc/clear_cache against live decode. (Accounting only — see `kv-cache.md` for KV.)

## Patch conventions (house rules — READ BEFORE writing any fast-path patch)

Exemplars, all real and shipping: `omlx/patches/nemotron_h_dq8.py` (load-time DQ8 quantize),
`omlx/patches/nemotron_ultra_decode/moe_fastpath.py` (decode MoE fast path),
`omlx/patches/nemotron_h_nvfp4_ts.py` (pre-load sanitize wrap). The rules exist because **one
process hosts many models**; a sloppy patch corrupts unrelated production models.

| # | Rule | Do | Never | Cite / incident |
|---|------|----|-------|-----------------|
| 1 | Per-instance class swap | `module.__class__ = Patched` on target instances only | class-level `Base.__call__ = …` (alters other models in-process) | `nemotron_h_nvfp4_ts.py:_patched_sanitize` (`mixer.__class__ = _TsFoldMoE`); `moe_fastpath.py:apply_nemotron_ultra_moe_fastpath`; `tasks/lessons.md §2026-07-05` |
| 2 | Idempotency by instance marker | `if getattr(model,"_x_applied",False): return` then set it | an `id()`-keyed registry — CPython **reuses freed addresses**, so a new object at a recycled id looks "already patched" and is skipped | `moe_fastpath.py` (`model._ultra_moe_fastpath_applied`); incident `tasks/todo.md §"PIPELINE CATCHES"` ("codex self-caught id-reuse idempotency corruption") |
| 3 | Select by structural fact, not class NAME | gate on `config` `block_type` / live shape / weight-key presence | `type(m).__name__ == "..."` — a name filter **misses `__class__`-swapped modules** and silently disables a stage while tests pass | `nemotron_h_dq8.py §docstring`; incident `tasks/todo.md §"PIPELINE CATCHES"` ("DQ8 stage B silently dead under `_TsFoldMoE` class swap … type-name filter") |
| 4 | Independent census BEFORE the loop | compute `expected` from a separate classification pass, then hard-fail on `expected != actual` | let the transform loop define its own success (a broken filter agrees with itself; `0==0` blind pass) | `nemotron_h_dq8.py:apply_ultra_dq8` (expected derived L208-247 before the quantize loop; RC8 docstring) |
| 5 | Engagement counters at INFO, greppable | log `expected=%d actual=%d` / `engaged=%d/%d` every step | assume "verified standalone" == "engaged live" | `nemotron_h_dq8.py` (`[ULTRA-DQ8] … expected=… actual=…`); `moe_fastpath.py:_RouteCensus` (`[ULTRA-DECODE] sorted_routes=%d/%d`); `tasks/lessons.md §2026-07-03 (dtype gate)` |
| 6 | Env kill-switch per patch | one `OMLX_*` env that disables the fast path (revert to stock) | ship a fast path with no live off-switch | table below |
| 7 | Correct hook point | pre-load wraps `sanitize`; post-load runs `apply_post_load_transforms` | mutate before weights/quant state exists | `utils/model_loading.py` (below) |
| 8 | Composed-reality tests | build the model **as patched in production** (all patches stacked) | test one patch in isolation and call it done | `tests/test_nemotron_h_dq8.py` (below) |

Idempotency note: a *pre-load* sanitize wrap may use a module-global flag (`nemotron_h_nvfp4_ts.py:_WRAPPED`)
because it wraps a class once per process; a *per-model post-load* patch must use an **instance**
marker (rule 2). Kill-switch **semantics** are a deliberate choice — "bit-stock" (drop sidecars, no
patch) vs "debug/attribution" (keep structure, skip only the math). `OMLX_NEMO_DISABLE_NVFP4_TS`
chose the latter: registration + `__class__` swap still happen so strict `load_weights` finds the
sidecar params; only the fold multiply is skipped, so output is mis-scaled — a DEBUG switch, **not a
correctness mode** (`nemotron_h_nvfp4_ts.py §docstring`; `tasks/lessons.md §2026-07-05`).

### Env kill-switch inventory
| Env var | Default | Effect | file:symbol |
|---------|---------|--------|-------------|
| `OMLX_ULTRA_DQ8_MAMBA` / `_MOEDENSE` / `_ATTN` / `_LMHEAD` | unset (off) | enable load-time DQ8 (affine8-gs64) per dense stage; `=1` on | `nemotron_h_dq8.py:_stage_enabled` (reads `DQ8_STAGES[name].env`); launch line `tasks/todo.md §"SERVING FIXES live"` |
| `OMLX_ULTRA_DISABLE_TSPRE` | off | revert ts-precombine to stock two-gather fold | `moe_fastpath.py:_tspre_disabled` |
| `OMLX_ULTRA_DISABLE_SORTED_ROUTES` | off | revert pre-sorted expert indices to stock unsorted call | `moe_fastpath.py:_sorted_routes_disabled` |
| `OMLX_NEMO_DISABLE_NVFP4_TS` | off | skip the ts-fold multiply (DEBUG; mis-scales output) | `nemotron_h_nvfp4_ts.py:_ts_disabled` |
| `OMLX_TRANSIENT_CLAMP_K` | 8.0 | clamp K for tracker-derived prefill-transient terms; `0` disables | `scheduler.py:_predicted_chunk_transient` |
| `OMLX_DISABLE_EARLY_INDEX_PUBLISH` | off | disable SF-1 early prefix-cache publish | `scheduler.py:_early_index_publish_disabled` |
| `OMLX_M3_DEBUG_PATH` | unset | MiniMax-M3 fast-path census counter level (`=N`) | `tasks/lessons.md §2026-07-03 (dtype gate)` — M3 domain, see `metal.md` |

### apply_post_load_transforms hook contract (utils/model_loading.py)
`utils/model_loading.py:apply_post_load_transforms(model, model_settings=None)` runs **after**
`mlx_lm.load()` returns a fully-loaded, already-quantized model. It receives the live model instance
and (optionally) `ModelSettings`. Order matters:
1. **Env-gated** transforms run FIRST, *before* the `model_settings is None` early return, because
   they must engage even when no settings are loaded: `apply_ultra_dq8(model)` then
   `apply_nemotron_ultra_moe_fastpath(model)`.
2. Settings-gated transforms (e.g. IndexCache, `index_cache_freq >= 2`) run after.

Pre-load patches are the other half: `utils/model_loading.py:maybe_apply_pre_load_patches` wraps
`sanitize`/config *before* `mlx_lm.load()` (dispatched by `config.json` `model_type`; e.g.
`nemotron_h_nvfp4_ts`, `glm_moe_dsa`, MTP). Decision: sidecar registration or sanitize behavior →
**pre-load wrap**; quantize/patch a loaded module tree → **post-load transform**.

### Composed-reality testing (tests/test_nemotron_h_dq8.py)
`_build_tiny_ultra_model(apply_tsfold_swap=True)` builds the model with the production ts-fold
`__class__` swap already applied (production-faithful by default).
`TestUltraDq8StagedEngagement::test_composed_with_moe_fastpath_both_patches_apply_correctly` stacks
**DQ8 then the MoE fast path** in the same order `model_loading.py` uses, and asserts the composed
output is finite and numerically matches stock (not just "each patch works alone"). This is the test
shape that would have caught rule-3's silent-dead-stage bug.

## model_settings.json anatomy (~/.omlx/model_settings.json — READ-ONLY, never edit)

Shape: `{"version": …, "models": {<model_id>: {<per-model settings>}}}` (29 model entries here).
Per-model keys seen in the wild (one entry, structure only): `max_context_window`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `repetition_penalty`, **`force_sampling`**, `model_type_override`,
`model_alias`, `thinking_budget_enabled`, `guided_grammar_enabled`, `turboquant_kv_enabled` /
`turboquant_kv_bits` / `turboquant_skip_last`, `specprefill_enabled`, `dflash_enabled` (+`dflash_*`
cache knobs), `mtp_enabled`, `vlm_mtp_enabled`, `is_pinned`, `is_default`, `trust_remote_code`. The
engine-construction subset (`mtp_enabled`, `turboquant_*`, `dflash_*`, `specprefill_*`, `vlm_mtp_*`,
`trust_remote_code`, `index_cache_freq`) forms the reload signature in
`engine_pool.py:_engine_runtime_signature` — changing one triggers an unload+reload, not a live edit.

**The force_sampling trap.** `force_sampling: true` (global or per-model, `server.py:get_sampling_params`
L1191; the `force = global_sampling.force_sampling or …` line is L1231) forces token-selection knobs
*even at `temperature=0`*, and a model-level `temperature`
override then sets the **effective** temperature. Greedy-only / temp-0 fast paths and quality gates
read that effective value and silently change behavior **before their own guard sees the request**:
- `scheduler.py:_vlm_mtp_eagle3_sampling_skip_reason` (:6721, called at :6711/:6891; the
  force_sampling/top_p/min_p skip is at L6731) skips EAGLE-3/spec decode when `force_sampling` pins
  stochastic filtering alongside temp=0.
- **EAGLE acceptance artifact**: `fs5` (MiniMax-M3) ships `force_sampling=true` + `temperature=1.0`,
  so temp-0 requests ran at effective temp=1 → the drafter **never engaged** and early
  acceptance/quality numbers were serving-path artifacts until force_sampling was lifted
  (`tasks/eagle3_build.md §"KEY CONFIG INTERACTION"`; `tasks/eagle_temp1.md` L9; `tasks/todo.md`
  L1003 "user's skepticism found the force_sampling artifact"). Rule: when A/B numbers surprise,
  check `force_sampling`/`temperature` overrides FIRST and re-run with them lifted.

## Streaming (delta.reasoning_content for thinking models)

Thinking models (Nemotron-3) stream **thinking tokens on `delta.reasoning_content` while
`delta.content` stays empty**, then switch to `delta.content` for the answer (`server.py` L4295-4447,
"Emit reasoning_content delta"). Templates that expose `message.reasoning_content` natively (Qwen
3.6+) are detected via `server.py:uses_native_reasoning_content`. SSE granularity = `stream_interval`
(tokens batched per chunk, default **1** = every token; `engine/batched.py` L45/57). Observed cadence
was ~**0.134 s**/chunk for Nemotron-3-Super (`tasks/lessons.md §Ultra`).

**Phantom dead-air misdiagnosis** (`tasks/todo.md §"THREE ULTRA-ONLY SERVING FINDINGS"` (1)):
Nemotron-3-Ultra delivered the *entire* answer as ONE SSE chunk at completion (128-token probe: 1
chunk @18.75 s) while Super streamed 8 chunks @0.134 s cadence; the **non-stream API was fine** and
decode was native 8 tok/s. **RESOLVED (2026-07-05):** the apparent "dead air" was the `reasoning_content`
channel (thinking streams there first), not a serving/decode regression — count BOTH channels and do not
attribute stream stalls to the kernel until you have compared the non-stream path. Probe recipe → `profiling.md`.
