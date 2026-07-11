> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Gotchas — the trap museum

Every entry that cost this campaign a day. **Grep the SYMPTOM text**: the heading
and the `Symptom:` line carry the exact error string / log signature / observable
weirdness, so a fresh session hitting the trap finds the antidote in one read.
`Root cause:` is the mechanism, `Antidote:` is the exact fix, `Ref:` is the
citation (sibling `file §heading`, `path:line`, `omlx#NNNN`, or `EXP-NNN` in
`experiments/index.md`). The general *principle* behind a trap lives in
`laws.md`; the *procedure* in `ops-runbook.md` / `conversion.md`. Ordered by how
likely a fresh session trips it. Run `scripts/preflight.py` to catch the static
ones before you serve.

---

## Bench numbers drift / A/B legs disagree / a wild first number — golden env not set
- Symptom: decode tok/s doesn't match the ledger; two runs of the "same" leg disagree; a first result like "13.35×" that later evaporates.
- Root cause: `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` unset → different Metal command-buffer batching than every ledger baseline was measured under.
- Antidote: put `MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000` on EVERY serve/bench line (the golden env).
- Ref: tasks/ultra_speed.md:5,21; env-setup.md; preflight.py check 1.

## Patch present after install but GONE at serve time / mtime snaps back — `uv sync` reverts the wheel
- Symptom: a pip-installed patched wheel (e.g. custom mlx-metal) is in site-packages right after install but absent at serve; its mtime resets; the patch-marker string is missing from the loaded lib.
- Root cause: `uv run` (and `uv sync`) re-syncs the venv against `uv.lock` on EVERY invocation, silently restoring the locked wheel.
- Antidote: `uv run --no-sync` (or install into the synced env, never edit uv.lock for experiments). Verify the marker in the LOADED libmlx, not on disk; check the PARENT uv process, not the renamed child.
- Ref: lessons.md:78-88; memory omlx-live-path-verification.md:27; laws.md Law 11; preflight.py check 4.

## temp-0 gate gives sampled output / spec drafter never engages — force_sampling overrides temperature
- Symptom: a temp-0 quality gate returns non-deterministic answers; an EAGLE/spec drafter shows zero engagement (no `vlm_mtp` stats / decode-started lines) though enabled; gsm8k gate noisy.
- Root cause: `force_sampling:true` in `~/.omlx/model_settings.json` overrides the caller's temperature to the model default (1.0) BEFORE the temp0 routing guard → the greedy-only drafter never engages.
- Antidote: lift `force_sampling` for temp0 gates and engagement benches. EVERY MTP leg must show `vlm_mtp` stats/decode-started lines or its numbers are VOID. Affects MiniMax-M3-oQ4 and -fs5.
- Ref: tasks/eagle_temp1.md:9-11; todo.md:869,1003; env-setup.md:160; EXP-047; preflight.py check 7.

## "Bit-exact standalone" kernel moves ZERO live numbers / census shows fused_none=57/57 — silent gate fallback
- Symptom: a verified-standalone kernel changes nothing live; a census line reads `fused_none=57 scores_fallback=57` every layer every step; an unexplained decode slope.
- Root cause: the fast-path gate fails CLOSED and SILENT — required fp16 while the live model runs bf16 (`torch_dtype`), or gated on `mask is None` while restored/batch decode carries a bool mask. Full-cache fp32 fallback glue = the anomalous cost.
- Antidote: ship every gated path with a live engagement COUNTER; standalone repros must copy the live dtype (read `config.json` torch_dtype) and the batch cache class. Grep the census AFTER a warm request, not at boot.
- Ref: lessons.md:17-26; memory omlx-live-path-verification.md; laws.md Law 1; EXP-025/026 (M3 instance), EXP-020 (GLM analog).

## Standalone script ~65× slower than in-server (~2.3 s/token vs ~40 ms) — no wired limit
- Symptom: an isolated `model(x, cache)` crawls at seconds/token; the same code in the server runs ~40 ms/token.
- Root cause: no `mx.set_wired_limit` → weights are unwired → GPU page-fault storm on every weight read. The server raises the limit at startup (process_memory_enforcer); a fresh script does not.
- Antidote: `mx.set_wired_limit(506*1024**3)` before the first forward in ANY standalone repro/probe/harness.
- Ref: GLM52_MTP_FORAY.md:388-392; todo.md:1020; metal.md. (Distinct from the sysctl `iogpu.wired_limit_mb` — preflight.py check 3.)

## Microbench says −3.5 ms but live tok/s is flat — isolated win vanishes in-stream
- Symptom: a lever/kernel measures a clear win standalone; the live serve leg shows no change.
- Root cause: standalone eval-per-call sync inflates the apparent gain; in-stream pipelining absorbs it. Surviving savings ≈ 50-60% of the naive sum.
- Antidote: measure the DELTA in-stream (chained `mx.depends` / real serve leg). Only in-stream serve legs decide ship/kill.
- Ref: lessons.md:3-15; todo.md:1072; laws.md Law 2; EXP-058.

## `adaptive_prefill_throttle predicted=155GB` on a box with headroom / chunks shrink — throttle mispredictor
- Symptom: throttle predicts a huge transient, prefill chunk size collapses (e.g. 2048→1817), a one-time pause + no-op eviction; a longer prompt is predicted CHEAPER than a shorter one (EWMA decay).
- Root cause: `_predicted_chunk_transient` takes `MAX(last_delta, EWMA, static)×1.3`; a one-off spike (≈120 GB expert-wiring / cache materialize) poisons `last_delta`/EWMA. Whole-process `phys_footprint` delta is charged as a per-token rate.
- Antidote: clamp both tracker signals to `K×static` (K≈8), env `OMLX_TRANSIENT_CLAMP_K` (0=off); static path untouched (the OOM SIGABRT margin must stay generous). Benign once/request; decode untouched.
- Ref: tasks/ultra_speed.md:556-566; todo.md:1059-1064; scheduler.py:_predicted_chunk_transient:3728 (`_PREFILL_TRANSIENT_SAFETY=1.3` at :3724); EXP-061.

## Reconverted checkpoint gives coherent-but-wrong / stale output — prefix cache poisoned by name reuse
- Symptom: a freshly reconverted model served under its OLD name emits stale/wrong-but-fluent text; startup logs "SSD cache scan complete … indexed=NNNN blocks".
- Root cause: the omlx SSD/prefix cache stores fp16 blocks keyed by MODEL NAME and SURVIVES restarts; changed weights under the same name reuse poisoned blocks.
- Antidote: changed weights → NEW MODEL NAME (e.g. add a `-dq8` suffix). Never reuse a name across a weight change.
- Ref: todo.md:1102,175; memory omlx-ultra-550b.md:40; laws.md Law 9; conversion.md:209.

## Identical repeat prompt re-prefills for ~40s instead of hitting — prefix-cache commit lag
- Symptom: an exact-repeat prompt misses the cache (full re-prefill); only the 3rd submission hits; big models only.
- Root cause: the token→block index entry becomes visible only when the store WORKER finishes; a follow-up arriving during the store window misses.
- Antidote: SF-1 early index publish (publish on the inference thread after `mx.eval`); kill switch `OMLX_DISABLE_EARLY_INDEX_PUBLISH=1`. Cuts exact-repeat 5k TTFT 40s→3.0s.
- Ref: tasks/ultra_speed.md:534-554; todo.md:1057,1075; EXP-060.

## Metal GPU command-buffer TIMEOUT / hang when the model lives on `/Volumes/*` — USB/NFS serving
- Symptom: Metal GPU timeout or hang for a model whose files resolve to a USB/NFS mount.
- Root cause: serving weights over the USB/NFS bus. On this box six models are `$OMLX_COLD_STORAGE/` symlinks — safe as COLD storage, fatal if SERVED.
- Antidote: copy the checkpoint onto the internal SSD before serving it. Keep only cold/unloaded models on the external drive.
- Ref: omlx#2098; env-setup.md:113,229-239; playbooks.md; preflight.py check 6.

## `rm` of an old model frees NO disk space / delete hangs under a live server — APFS + mmap
- Symptom: `rm` succeeds but `df` shows no space reclaimed; or deleting the file corrupts/hangs the running model.
- Root cause: APFS holds a file's blocks while a live process has it mmap'd; space frees only when the maps close.
- Antidote: bounce the server BEFORE `rm`. Ordered: stop pool → `rm` → restart/discovery.
- Ref: conversion.md:211-212; playbooks.md:132; ops-runbook.md §7; laws.md Law 9.

## NIAH passes but a shorter follow-up request decodes garbage — single-PASS ≠ multi-correct
- Symptom: long unique prompts pass; a shorter request reusing the SAME cache bucket after a longer one produces garbage; only a temp0 token-identity check on the reused bucket catches it.
- Root cause: persistent/cached state keyed on capacity or shape, reused across requests without a reseed.
- Antidote: test request A then a SHORTER B on the same bucket with different content — offline AND live — for any capacity/shape-keyed state.
- Ref: lessons.md:90-96; laws.md Law 7.

## gsm8k 92% serial but 15% under concurrent load — SpecPrefill concurrency bug (OPEN)
- Symptom: a model scores ~92% gsm8k serial and ~15% under 6-way concurrency; short benches (mmlu/arc, 8-token) hide it, only long generation (gsm8k, 512-token) exposes it.
- Root cause: SpecPrefill degrades under concurrency — a serving-path artifact, NOT the model/quant. Still OPEN.
- Antidote: reproduce ANY surprising A/B quality gap SERIALLY before blaming the model; bench spec-decode single-request until fixed.
- Ref: lessons.md:121-127; todo.md:1037; scripts/acc_bench_serial.py:5-9; EXP-049.

## A patch stage reports "0 modules" / 0==0 hard-fail / silently dead while tests pass — class-name filter
- Symptom: a stage finds no modules, or hard-fails on a `0==0` count, or does nothing live; unit tests still pass; the target model swapped its class (e.g. to `_TsFoldMoE`).
- Root cause: filtering modules by type NAME misses instances whose `__class__` was swapped.
- Antidote: filter by `block_type` / structural signal, not class name; add an independent engagement census per stage (expected==actual).
- Ref: todo.md:1085; lessons.md:99-105; laws.md Law 5, Law 12.

## Idempotency guard mis-fires after an object is freed — id()-keyed registry + address reuse
- Symptom: an "apply once" guard skips a NEW model, or double-applies, after some object was garbage-collected.
- Root cause: the registry/guard is keyed on `id(obj)`; CPython recycles freed addresses, so id() is not a stable identity.
- Antidote: use a per-INSTANCE attribute marker (`model._x_applied = True`), never an id()-keyed set/dict.
- Ref: todo.md:1085-1086; omlx/patches/nemotron_ultra_decode/moe_fastpath.py:201,229; laws.md Law 12.

## Patching model A corrupts production model B in the same process — class-scope method rebind
- Symptom: enabling a patch for one model degrades another model sharing the process; `SomeClass.__call__ = ...` at module scope hits every instance.
- Root cause: dunder `__call__` resolves on the TYPE; a class-scope rebind is global to all instances.
- Antidote: per-instance `mixer.__class__ = _SubClass` swap on ONLY the target layers; self-gate on a checkpoint signal (e.g. `.fc1_ts` keys). Prove non-interference by loading BOTH models and asserting the other stays stock.
- Ref: lessons.md:99-105; laws.md Law 12.

## gather_qmm errors on rhs_indices / batch legs produce garbage — index shape must be (M,1,TOP)
- Symptom: `mx.gather_qmm` batch legs (M=2/4) error on `rhs_indices` broadcasting, or return wrong values.
- Root cause: gather_qmm wants indices shaped `(M, 1, TOP)`; `(M, TOP)` collides with the broadcast contract.
- Antidote: reshape indices to `(M, 1, TOP)` (at M=1: `mx.arange(nh).reshape(nh, 1)`).
- Ref: tasks/ultra_speed.md:58-59,259; omlx/patches/dsv3_decode_opts.py:51,69; decode_kernels.py:385; mlx.md.

## gather_qmm MoE result plausible but wrong / false parity — take_along_axis on the wrong axis
- Symptom: a MoE weighted-sum "passes" parity by luck or fails oddly; the output looks reasonable but mixes experts.
- Root cause: for a `[1,1,IN]`-shaped input the experts land on OUTPUT axis 1 (e.g. `[1,TOPK,1,OUT]`); reducing or `take_along_axis`-ing on the wrong axis silently blends experts.
- Antidote: reduce the EXPERT axis explicitly (`.sum(axis=1)` / epilogue `.sum(axis=-2)`); verify against an independent reference at REAL dims, not toy shapes (toy shapes hide axis bugs).
- Ref: tasks/compile_spikes/spike_de_scale.py:93-100; omlx/patches/glm_moe_dsa/switch_layers.py:205,254; ultra_speed.md:423; mlx.md.

## Fused SwiGLU `gate_up` folded/split on the wrong half silently blends gate & up — out-axis is `[gate; up]`
- Symptom: a SwiGLU MoE (M3, GLM) converts/serves without error but output is subtly wrong; a ts-fold or `silu(gate)·up` split "passes" a toy test yet blends the two projections.
- Root cause: the fused `switch_mlp.gate_up_proj.weight` packs on the **out-axis** as `[gate(3072); up(3072)]` — `split(axis=-1)` first half is **gate**. The ts-fold `gate_up_ts[E,2]=[gate_ts, up_ts]` and the runtime `silu(gate)·up` both depend on this order; reversing it silently mixes gate/up (same family as the take_along_axis wrong-axis trap above).
- Antidote: preserve `[gate; up]` order; verify the fold/split against an independent reference at REAL dims (E=128/129 M3, 256/257 GLM), not toy shapes.
- Ref: tasks/oqnvfp4_build.md:58-60; conversion.md §Stage 2 / §GLM deltas; mlx.md.

## Separate-shared NVFP4 checkpoint crashes at engine load on a down_proj concat — MoE sanitize force-fuses shared into routed
- Symptom: a separate-shared oQNVFP4 checkpoint (routed nvfp4 + a separate affine8 shared MLP) loads fine via raw `mlx_vlm` but **crashes in the omlx engine** on a concat shape mismatch, e.g. `(128,6144,384)` vs `(1,6144,768)`.
- Root cause: `omlx/engine/vlm.py:_pack_mlx_unpacked_moe_weights` (:759) force-fuses `shared_experts.down_proj` into `switch_mlp.down_proj` whenever BOTH keys exist (:805-814) — a second, engine-side MoE sanitize layer the raw loader never runs (Law 5's "two sanitize layers").
- Antidote: emit `omlx_moe_shared_expert_mode` in config and **skip** the fuse when it is set (`if getattr(args,"omlx_moe_shared_expert_mode",None) is not None: return 0`, :765-766). Requires a **server restart** — the engine imports `omlx.engine.vlm` at startup.
- Ref: tasks/oqnvfp4_build.md:150-158; omlx/engine/vlm.py:759,765,805-814; conversion.md §GLM deltas; laws.md Law 5.

## Converter crashes: ModelArgs TypeError or `mx.clip` ValueError — config dialect not normalized
- Symptom: `mlx_lm` ModelArgs positional TypeError (no `num_hidden_layers`); or `mx.clip` ValueError on `time_step_limit=[0.0,{"__float__":"Infinity"}]`.
- Root cause: Ultra/Nemotron configs ship only `layers_block_type` (no `num_hidden_layers`) and tag non-finite floats as `{"__float__": …}` dicts.
- Antidote: `cfg.setdefault("num_hidden_layers", len(pattern))`; decode/drop the tagged non-finite `time_step_limit`. Normalize in the converter — the WEIGHTS need no change (pure repack holds).
- Ref: omlx/tools/oqnvfp4_nemotron_convert.py:321-332; todo.md:1045-1047; conversion.md:112-115.

## 3-tuple cache truncated to 2 / length-2 state crashes reconstruct — legacy V2 cache format
- Symptom: a 3-tuple cache state loses its 3rd element, or a length-2 legacy state crashes cache reconstruct; placeholder layers carry `state == ()`.
- Root cause: the default 2-tuple keys/values mapping truncates the newer 3-tuple (pooled) state; older "V2-truncated" states need tolerance.
- Antidote: dispatch via `deserialize_state` (not the 2-tuple map); tolerate `len==2` by filling the missing pooled slot with None.
- Ref: omlx/patches/deepseek_v4/cache_handlers.py:111-127; scheduler.py:2004,5793,5826; kv-cache.md.

## Batch≥2 / compiled decode over the real MiniMaxM3BatchKVCache — four cache landmines
- Symptom: a batch≥2 or compiled-decode integration over the live cache raises `Slice indices must be integers, not array`, or the scheduler declares "unrecoverable cache corruption" and clear+re-prefill loops, or the regime gate wrongly engages on a padded batch.
- Root cause (four distinct traps): (1) live batch caches carry `offset` as an `mx.array`, so a python slice `keys[:,:,:offset,:]` raises → **seed by BUFFER SHAPE via `mx.pad`** (`_seed_to_cap`); (2) `MiniMaxM3BatchKVCache.offset` is a **READ-ONLY property** (delegates to inner `.kv_cache.offset`) — setting it on writeback raises → scheduler reads it as corruption → clear+re-prefill loop → **unwrap to the inner settable offset** (`_advance_offset`); (3) the regime gate must accept a scalar `mx.array` offset (`_as_int`) + an all-True L=1 mask but **REJECT padded batches** (per-row offsets aren't single-seq-safe); (4) build the offline repro the way the scheduler does — `type(c).merge([c])` per layer rebuilds the live cache with no server.
- Antidote: the four fixes above; regression-gate with `compile_spikes/spike_phaseb_{smoke,batch,reuse}.py` + `phaseb_real_parity.py`.
- Ref: tasks/overlap_levers.md:154-168; tasks/compile_spikes/PHASE_B_STATUS.md §3b-3d; spike_phaseb_batch.py:52-54; future-campaigns.md #2.

## Client shows dead air during thinking / TTFT≈0 / bogus prefill tok/s — wrong SSE delta field
- Symptom: a client counting only `delta.content` sees phantom dead-air (thinking tokens arrive in `delta.reasoning_content`); or TTFT≈0 / inflated prefill tok/s because the first role chunk has `content:""` and matches naively.
- Root cause: measuring the wrong streaming field, or treating the empty role chunk as the first token.
- Antidote: for TTFT detect the first NON-EMPTY `content` OR `reasoning_content` delta; count both fields for tok/s; cross-check against the server log, not the stream alone.
- Ref: memory omlx-glm52-native-kernels.md:104; todo.md:676-677,1050-1051; profiling.md.

## 128-token answer arrives as ONE SSE chunk at completion — the reasoning_content channel (RESOLVED)
- Symptom: a whole response streams as 1 chunk at ~18.75s while a sibling model (same arch/tokenizer) streams ~8 chunks at ~0.13s cadence; the non-stream API is fine.
- Root cause: NOT a serving defect. Ultra is a thinking model — it streams thinking on `delta.reasoning_content` then the answer on `delta.content`; a client counting only `delta.content` sees dead air then one final content chunk (session-verified 2026-07-05, two stream-tracer theories refuted).
- Antidote: count BOTH `content` and `reasoning_content` deltas; use the non-stream API / stream-free T1/T256 for decode rate (the "48 tok/s @5k" was this artifact; real decode 8.0). See profiling.md §3, cadence_probe.py.
- Ref: todo.md:1055-1056,1065 (ledger still says "in flight" — closure never written); EXP-051; profiling.md §3; models/nemotron-ultra.md.

## Grep right after "Application startup complete" finds no census/engagement line — lines stream LATE
- Symptom: no `[M3CENSUS]` / `[ULTRA-DQ8] expected==actual` / `sorted_routes=48/48` / "patch applied" / "baked checkpoint detected" line immediately after boot.
- Root cause: those fire at model LOAD and on the FIRST decode step (once per trace/step) — minutes after "startup complete" (e.g. patch applied ~11 min later, census only on first decode).
- Antidote: grep engagement AFTER a warm request completes, not at boot. A line's ABSENCE after a warm request = the path never engaged.
- Ref: tasks/compile_spikes/serve_211919.log:45,47,82; scheduler.py:1703; ops-runbook.md §3.

## HF repo storage dwarfs the actual checkpoint — usedStorage counts all revisions
- Symptom: the Hugging Face storage figure is far larger than the served checkpoint size.
- Root cause: `usedStorage` is repo-wide — it counts ALL revisions / LFS history, not just `main` (general HF API behavior, not measured here).
- Antidote: size from `?expand=safetensors` shard bytes (summed) or a local `du -sh` of the RESOLVED snapshot; never plan disk from `usedStorage`.
- Ref: tasks/glm_quant_matrix.md:20,37 (sizing method); conversion.md.

## A mid-convert `du -sh` under-reports and looks stalled — conversion pacing artifact
- Symptom: `du` on an in-progress conversion/download shows far below the final size and appears stuck.
- Root cause: shards are written and flushed in bursts; a mid-run `du` samples a warmup/flush trough, not the true pace.
- Antidote: trust the converter's `Summary:` line + census (tensor/shard/GB counts), not a mid-run `du -sh`.
- Ref: conversion.md:184.

## `tail` of a crashed bench shows only `ZeroDivisionError`, not the real cause — tail eats pre-crash lines
- Symptom: post-mortem `tail -N` of a crashed run shows the final traceback (e.g. `ZeroDivisionError` on `ok / len(items)`) but not the earlier root-cause lines.
- Root cause: the fatal traceback is at the END of the log; the actual cause (an n=0 suite, an upstream 400/timeout) scrolled past above it.
- Antidote: read the WHOLE log (or `head`) on a crash, not just `tail`; guard divide-by-n (skip n=0 suites, don't divide).
- Ref: scripts/acc_bench_serial.py:11-14,72.

## `ps eww <child-pid>` shows a renamed process without the launch env — check the parent
- Symptom: `ps eww` on the omlx-server child shows a renamed process and none of the launch env vars; you can't confirm what the serving process actually got.
- Root cause: the server child is renamed and its env isn't exposed there.
- Antidote: read the env on the PARENT `uv` process; and verify the patch-marker in the LOADED lib, not on disk.
- Ref: lessons.md:87-88; laws.md Law 11.

## `command not found: and` / "no matches found" / `timeout` missing / `ls -t` fails — Bash tool is zsh, not fish
- Symptom: `(eval):N:` errors; `command not found: and`; an unquoted glob like `--include=*.md` ABORTS the whole command with "no matches found"; `timeout` → command not found; `ls -t`/`ls -lt` → "Option --time (-t) has no … setting".
- Root cause: the Bash tool runs under zsh/POSIX (the fish login shell only handles the `! cmd` prompt prefix); zsh aborts on non-matching globs; macOS has no `timeout`; `ls` is aliased to eza.
- Antidote: POSIX syntax (`&&`, `||`, `VAR=val`, `$?` — not fish `and`/`set`/`$status`); quote globs or feed files via `find … | tr '\n' '\0' | xargs -0 grep`; use `/bin/ls -t` or `command ls`; use `gtimeout` or a subprocess timeout (`===` is not a shell operator; foreground `sleep` is blocked — use a Monitor until-loop); `mkdir -p` the scratchpad before redirecting into it.
- Ref: memory bash-tool-uses-zsh.md; observed this session (the `--include` glob abort and missing `timeout`).

## Subagent can't see the plan / live model_settings edit DENIED / box idle with server down — multi-agent ops
- Symptom: a spawned subagent lacks the session context; an agent's edit to `~/.omlx/model_settings.json` on the live server is denied in auto-mode; the server sits stopped while an agent waits.
- Root cause: subagents don't share session chat (the plan/handoff FILE is the whole context); a live `model_settings.json` edit is a production-config change that teammate messages can't consent to (auto-mode denies — only the USER can); a stopped server wastes the box.
- Antidote: hand off via a file/plan, not chat; a live model_settings change needs USER approval; stopped-server work MUST be ONE chained background command ending in the serve relaunch + `/health` probe (never idle with the server down).
- Ref: lessons.md:47-54; memory omlx-perf-skill-swarm.md:12; firstmate-herdr-setup.md:18; ops-runbook.md §6, §8.

## Server killed mid-benchmark during a model swap — the pool loaded a 2nd ~230GB model before evicting the 1st
- Symptom: the `omlx serve` process dies (client sees RemoteDisconnected → Connection refused) exactly when a request switches from one big model to another (e.g. `-fused`→`fs5`, or M3→Ultra). Wired memory had climbed toward the ~493GB enforcer ceiling.
- Root cause: two ~230-305GB models resident at once (2×230 = 460 + working set > the 492.9GB balanced-tier ceiling). The engine_pool admitted the 2nd load before the 1st freed (the "double-count freed memory in load admission" class — commit 5a26eb1). **The box holds ONE big model at a time.**
- Antidote: for an A/B or any swap, load each big model SOLO on a FRESH server (restart between models) — never trigger a live pool swap between two 230GB+ models. In production a concurrent **Ultra(305G)+M3(230G)** request pair can recur this — standing watch-item (the enforcer kills the whole process, not just the request). Ref: `tasks/todo.md:773-775` (dual 260+240GB stalled → sequential lesson); EXP-066; ops-runbook.md §memory.

## A fresh number looks like a regression/win vs the ledger — but the ledger is another DAY, not same-session
- Symptom: you compare today's decode/accuracy to a STASHED ledger figure and conclude "no penalty" / "big win" — then a same-session A/B flips it. (2026-07-06: `-fused` short 27.26 read as "no penalty" vs the 07-04 fs5 ledger 27.09; same-session fs5 was actually **28.54** → a real ~5% penalty.)
- Root cause: the decode tree + sampling drift across days (M3 fs5 16k **21.64 @07-04 → 27.08 @07-06**). Absolute numbers are comparable only WITHIN one serve session, same env/tree.
- Antidote: never A/B a fresh model against a stashed number — re-measure the baseline in the SAME session, same battery, back-to-back. Stashed numbers are historical context only. Ref: EXP-066; `models/minimax-m3.md §6`; Law 10 (serve-window discipline).

## Session 2026-07-07/08 additions
- **zsh word-splitting (Bash tool):** unquoted `$FLAGS` does NOT split in zsh — "--mtp --kv-bits 8"
  passes as ONE argv token; argparse rejects it while printing the flags as valid in usage. Use
  `${=FLAGS}` or arrays. Burned three bench legs before diagnosis.
- **Admin API needs a browser session cookie** (`require_admin`, auth.py:257) — the bearer key 401s.
  Scriptable path for built-in evals: drive `omlx.eval.BENCHMARKS` classes directly over `/v1` with an
  HTTPEngine shim (`chat_template_kwargs.enable_thinking` passes through the API). See
  scratchpad/overnight_intel.py pattern; banked in EXP-085.
- **Aborted-prefill retention (OPEN watch-item):** client-killed mid-prefill leaves ~37GB pool watermark
  until restart (same family as MTP warm-holder). Preflight-off + pool credit mask the admission symptom;
  the memory still sits. Repro: kill a 500k prime mid-flight.
- **Server replays exact-duplicate temp-0 requests** (Law 14) and **/health lies about model readiness**
  (Law 13) — both are probe-killers, not serving bugs.
- **mx.eval/clear_cache thread doctrine:** `_sync_and_clear_cache` from the asyncio thread = documented
  SIGABRT (#300/#888/#1106). Counter reads (`mx.get_cache_memory`) are safe anywhere; real reclaims only
  on the inference thread (`_reclaim_prefill_headroom`, `_process_pending_reclaim`).
- **hy_v3 vendored code: conversion-time router hazard (codex, 2026-07-08):** hy_v3_model.py's
  quant_predicate quantizes mlp.router.gate to 8-bit and cast_predicate doesn't protect the router
  weight — fine for SERVING Alis checkpoints (config excludes router -> bf16 load + fp32 matmul), but
  NEVER convert/re-quantize Hy3 through this class without overriding both predicates (router bf16 is
  doctrine: near-tie top-8 margins). Also: the self-retire check (__init__.py:131) catches broad
  ImportError — on a future mlx-lm pin bump, verify natively-shipped hy_v3 isn't being shadowed.

## Off-by-one comparing per-step decode logits to a token trajectory
- Symptom: distribution-overlap metrics (min-sum acceptance, KL) come out absurdly low (~0.01) while argmax-match on the same data reads ~0.5.
- Root cause: in a decode loop `toks[j] = sample(LG[j])` — `LG[j]` is the distribution that SAMPLED `T_j`, not the distribution FOR `T_{j+1}`. Comparing a head's dist-for-`T_{k+2}` against `LG[k+1]` silently compares neighboring positions.
- Fix/guard: the dist for token `T_m` is `LG[m]`. Sanity-invariant before trusting any overlap metric: `mean(argmax(LG[m]) == T_m)` must be ~1.0 on a greedy trajectory (EXP-094 addendum, 2026-07-09).

## Temp-0 byte-identity gates are INVALID across forward widths on quantized models (qmv vs qmm)
- Symptom: a spec-decode / MTP / chunked leg diverges from plain greedy at temp0 and gets misdiagnosed as a state-bookkeeping bug that "must" be fixable.
- Root cause: L=1 decode dispatches quantized matrix-VECTOR (qmv); L>1 verify/chunk dispatches quantized matrix-MATRIX (qmm). The kernels round differently — measured max|logit diff| 0.5 on Puzzle oQ48; divergences occur ONLY at fp32 ties / margins <= the gap (EXP-097 campaign probe, 2026-07-09). No bookkeeping can make qmv and qmm agree on an exact tie. Production GLM MTP serves with this same property.
- Fix/gate: for mixed-width A/Bs use a margin-gated rail instead: token match% >= ~98 AND every first-divergence margin (top1-top2 logit gap) <= ~1.0 (tie class), plus a task-quality spot (gsm8k). A LARGE-margin divergence is still a real bug. Byte-identity remains valid only within a single forward width (e.g. fusion pool A's 192/192 held because it kept L=1 shapes).

## A kernel measured safe standalone faults the GPU only when pipelined
- Symptom: a fused kernel passes parity and runs cleanly under eager standalone decode, then the live pipelined decode intermittently faults the Metal command buffer (kIOGPUCommandBufferCallbackErrorInnocentVictim, exit 134).
- Root cause: one-ahead decode overlaps command-buffer submission in ways eager decode never exercises; a kernel racing that overlap only manifests under it. (Pool C's specific root cause is UNDIAGNOSED — do not assume.)
- Antidote: Law 18 — soak every new kernel under sustained pipelined decode before calling it safe.
- Ref: EXP-095 (Puzzle fused-experts pool C, banked unwired in omlx/patches/nemotron_h_puzzle/pool_c.py).

## A verdict evaporates on reread — two numbers came from different harnesses
- Symptom: a ledger row claims a win/regression that contradicts the campaign's own bench; the cited number turns out to be from a different script (e.g. a verify_econ latency mistaken for a fusion result).
- Root cause: harness conflation — different scripts measure at different pipeline depths.
- Antidote: Law 17 — name the harness beside every ms number; re-derive cross-harness comparisons by rerunning one harness end-to-end.
- Ref: EXP-095 correction note (the "+19% fusion" that was really verify_econ t1).
