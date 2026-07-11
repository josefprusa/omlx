> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Preflight — before you serve or bench

Run this **before every serving session and before every A/B bench**. Half the
campaign's wasted days were a mis-set env var, a silently-reverted wheel, or a
bench measuring the wrong thing. `scripts/preflight.py` automates all eight
checks below; this file is the human-readable rationale (WHAT / WHY / HOW /
antidote). The script and this doc mirror each other 1:1.

```
.venv/bin/python .claude/skills/omlx-perf/scripts/preflight.py
```

It is **serverless**: local file reads + `sysctl`/`ps` only. No HTTP to
localhost:8000, no tmux, no model load, no network — safe to run while the
production server is live. Exit code 1 if any FAIL, else 0 (WARN never fails).
For any FAIL/WARN the antidote points into `gotchas.md` by its grep-able heading.

---

## 1. Golden MLX buffer env
- **WHAT**: `MLX_MAX_OPS_PER_BUFFER=4000` and `MLX_MAX_MB_PER_BUFFER=4000` are set in the current env.
- **WHY**: final campaign golden values; a missing/wrong value shifts decode tok/s and voids A/B comparisons — the Ultra "13.35x" first result was a missing-golden-env artifact (tasks/ultra_speed.md:5,21).
- **HOW**: script reads `os.environ`; manually `env | grep MLX_MAX`.
- **ANTIDOTE**: put both on the serve/bench line. gotchas.md "golden buffer env".

## 2. Per-model OMLX_* switch inventory
- **WHAT**: report which `OMLX_*` switches are set; WARN if a `*_DISABLE_*` kill switch is on (an optimization is OFF).
- **WHY**: kill switches default OFF in production; one left set silently disables a shipped kernel and you bench a degraded path (env names: grep `OMLX_` omlx/patches/ omlx/scheduler.py).
- **HOW**: script scans `os.environ` against the known inventory (Ultra DQ8 stages, M3 fused-* disables, telemetry knobs).
- **ANTIDOTE**: unset kill switches unless deliberately A/B-ing that lever. gotchas.md "kill switch left on".

## 3. Metal wired-limit sysctl
- **WHAT**: `iogpu.wired_limit_mb` >= 518144.
- **WHY**: raises the Metal residency cap 464->506GB (+42GB KV headroom); too low starves long-context KV and can OOM (tasks/todo.md:227,243).
- **HOW**: `sysctl -n iogpu.wired_limit_mb`.
- **ANTIDOTE**: `sudo sysctl iogpu.wired_limit_mb=518144` (USER must run; server warns at startup if unset). gotchas.md "standalone 65x slowdown" covers the *process-level* wired-limit cousin.

## 4. MLX/nanobind version + wheel provenance
- **WHAT**: installed MLX == 0.31.2 and nanobind == 2.12.0, via the repo `.venv/bin/python`.
- **WHY**: production is STOCK mlx 0.31.2 (tasks/overlap_levers.md:105); `uv sync`/bare `uv run` re-syncs against uv.lock and SILENTLY reverts a pip-installed patched wheel at every launch (memory omlx-live-path-verification.md:27). nanobind 2.12.0 is the GLM native-kernel ABI pin (commit 3c224ed).
- **HOW**: subprocess `import mlx.core; print(mx.__version__)` (no model load). NEVER trust site-packages-on-disk — verify the LOADED lib.
- **ANTIDOTE**: `uv run --no-sync` (never bare `uv run`/`uv sync`). gotchas.md "uv sync reverts the patched wheel".

## 5. Disk free on ~/.omlx/models volume
- **WHAT**: free GB vs a table of known serving footprints (Ultra-dq8 327GB, Kimi 434GB, M3-fs 246GB, M3-oQ4 228GB, Puzzle-oQ48 43.84GB).
- **WHY**: a full-precision download/conversion needs real headroom; **Hugging Face `usedStorage` is repo-wide (all revisions)** so it overcounts, and a mid-run `du` under-reports (sizing method: tasks/glm_quant_matrix.md:20; du pacing: conversion.md:184).
- **HOW**: `shutil.disk_usage(~/.omlx/models)` (instant, no `du` I/O storm on the live box).
- **ANTIDOTE**: size from summed safetensors shard bytes or the converter's Summary line, not HF storage or a mid-run `du`. gotchas.md "HF usedStorage", "du mid-convert".

## 6. No USB/NFS-volume serving (omlx#2098)
- **WHAT**: FAIL if any model path under `~/.omlx/models` resolves (realpath) to `/Volumes/*`.
- **WHY**: serving weights off a USB/NFS volume -> Metal GPU command-buffer TIMEOUT (omlx#2098). On this box six models are `$OMLX_COLD_STORAGE/` symlinks — fine as COLD storage, fatal if SERVED.
- **HOW**: `os.path.realpath` each entry (depth 1 + namespace subdirs), check `/Volumes/` prefix.
- **ANTIDOTE**: copy the checkpoint onto the internal SSD before serving it. gotchas.md "USB/NFS serving -> Metal timeout".

## 7. model_settings force_sampling / temperature audit
- **WHAT**: WARN per model in `~/.omlx/model_settings.json` with `force_sampling: true`.
- **WHY**: `force_sampling` OVERRIDES a caller's `temperature: 0` -> every temp-0 quality gate and every spec-decode/EAGLE temp0 engagement guard is silently defeated. The MiniMax-M3 models (oQ4, fs5) set it (tasks/eagle_temp1.md:9 — "drafter never engages unless force lifted").
- **HOW**: parse the JSON (read-only — NEVER edit it live; that's a production-config change auto-mode denies, lessons.md:52).
- **ANTIDOTE**: lift `force_sampling` for temp-0 gates / engagement benches. gotchas.md "force_sampling overrides temp-0".

## 8. Quiet box (no competing GPU-heavy processes)
- **WHAT**: WARN if more than one >4GB python/mlx/omlx process is running.
- **WHY**: a second bench/convert/model-load contends for the 819GB/s bus and skews decode tok/s; batch-1 decode is bandwidth-bound so any co-tenant reads directly into your numbers.
- **HOW**: best-effort `ps -Ao pid,rss,comm` (never contacts the server — just lists it).
- **ANTIDOTE**: run benches solo; let conversions finish first. gotchas.md "noisy box / co-tenant".

---

## What preflight deliberately does NOT check
- **Live engagement** of gated fast paths — that requires the running server's
  census/log lines (`[M3CENSUS]`, `[ULTRA-DQ8] expected==actual`,
  `sorted_routes=48/48`), which arrive LATE (on first load + first decode, not
  at "Application startup complete"). Grep them AFTER a warm request, not at
  boot. See gotchas.md "engagement lines stream late" and the live-path law in
  laws.md. Preflight is static; it cannot prove a kernel engaged.
- **Correctness across requests** — the prefix-cache-poisoning and
  single-PASS-≠-multi-correct traps need a real 2-request probe (gotchas.md).
