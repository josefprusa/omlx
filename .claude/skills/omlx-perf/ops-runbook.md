> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# ops-runbook — the serve-window operations manual

The loop for running a serve window: restart → verify engagement → bench → (swap). Setup
and env discipline are in `env-setup.md`; the relaunch template is
`scripts/restart_server.sh`. Per-model perf/quality tables live in `models/*.md`; this file
is the PROCEDURE.

Repo `$OMLX = $OMLX`. Server in tmux `omlx:4`, port 8000.

> **GOVERNANCE — omlx is LOCAL-ONLY.** Never `git push`, never open a PR / pull request, no
> destructive git (force-push, hard-reset shared refs, branch deletion). The captain does not
> want GitHub PRs. The campaign record is the ledgers `tasks/*.md` + `memory/*.md` (currently
> git-untracked — their distilled truth lives in THIS skill; the ledgers should be committed).
> Bank before any destructive fs/config action (§7-8); the **user** approves production changes
> (`memory/firstmate-herdr-setup.md`).

---

## 1. Restart choreography

Ordered, each step gated on the previous (never bench a half-loaded server):

1. **Stop** the running server (in `omlx:4`): `C-c`, or from a shell
   `pkill -INT -f "omlx serve"`.
2. **Confirm it is gone**: poll `pgrep -f "omlx serve"` until it returns nothing (the pool
   shutdown emits `Engine pool shutdown complete`, `~/.omlx/stock-v045.log`). Old process
   keeps port 8000 until fully drained — do not relaunch on top of it.
3. **Relaunch** with the target model's env (see §2), teed to a fresh log.
4. **Poll the log** for `Application startup complete.` (accepting requests) then
   `Loaded model: <name>` (target resident).
5. **Engagement grep** (§3) BEFORE any tok/s claim.

`scripts/restart_server.sh` automates 1–4 and prints the §3 reminder.

> OPS CONTRACT (binding, `tasks/lessons.md` 2026-07-04): stopped-server work must be ONE
> chained command that ENDS in the relaunch — never leave the box serving nothing waiting
> on a human/notification. "Got dinged twice for server down + idle."

---

## 2. Launch lines — VERBATIM, per model

One server hosts all discovered models, but the **env is set per launch** and is tuned for
the model you are about to serve/bench. Restart to change env (one lever per restart, §6).
All are plain `uv run` because production runs stock mlx 0.31.2 (`env-setup.md` §1); add
`--no-sync` only if a custom MLX wheel is installed.

**Nemotron Ultra — CURRENT PRODUCTION** (13.07–13.09 tok/s, `omlx-ultra-550b.md`):
```
env OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000 \
    OMLX_ULTRA_DQ8_MAMBA=1 OMLX_ULTRA_DQ8_MOEDENSE=1 OMLX_ULTRA_DQ8_ATTN=1 OMLX_ULTRA_DQ8_LMHEAD=1 \
    uv run omlx serve --log-level info
```
The four `OMLX_ULTRA_DQ8_*` vars are **INERT on the baked `-dq8` checkpoint** — expect four
`baked checkpoint detected … load-time DQ8 skipped` INFO lines, not `expected==actual`
(`omlx-ultra-550b.md` UPDATE; `tasks/todo.md:1100`). Keep them in the line anyway: they
re-engage load-time DQ8 if you ever point it at the un-baked NVFP4 master.

**GLM-5.2-Alis-MLX-Dynamic-3.5bpw** (golden env, 23.0 tok/s, `omlx-glm52-decode-opts.md`
2026-07-04 LEVER#2):
```
env OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000 \
    uv run omlx serve --log-level info
```

**MiniMax-M3-oQNVFP4-fs5** (production — golden env, same as the fleet):
```
env OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000 \
    uv run omlx serve --log-level info
```
The MiniMax-M3 nvfp4 kernels can time out the Metal command buffer, but the golden
**`MLX_MAX_MB_PER_BUFFER=4000` cap (~one M3 layer of weights → commit-per-layer) is the fix** —
it bounds the buffer and prevents the timeout while keeping OPS=4000 for throughput. This
**supersedes** the earlier `OPS=500` workaround (`tasks/oqnvfp4_build.md:127`,
`tasks/eagle3_build.md:5` — both stale; the golden line shipped 2026-07-04,
`tasks/eagle_temp1.md:22`, `memory/omlx-glm52-decode-opts.md:183`). Cross-ref
`models/minimax-m3.md` §4, `gotchas.md`. (Ultra's nvfp4 experts do NOT hit this either.)

`OMLX_M3_DEBUG_PATH=256` is the permanent census instrument (engagement visibility), keep
it on always (`tasks/todo.md:569`).

---

## 3. Engagement verification (before benching)

**Procedure:** after `Loaded model:`, grep the teed log for the model's engagement lines;
every count must match its target and fallbacks must be zero. A wrong tok/s from a silently
de-engaged fast path is the single most repeated failure here
(`omlx-live-path-verification.md`). Per-model grep sets belong to `models/*.md`; the worked
example:

**Ultra (baked, current production)** — load-time census, four lines:
```
[ULTRA-DQ8] mamba    baked checkpoint detected (96 modules); load-time DQ8 skipped
[ULTRA-DQ8] moedense baked checkpoint detected (192 modules); load-time DQ8 skipped
[ULTRA-DQ8] attn     baked checkpoint detected (24 modules); load-time DQ8 skipped
[ULTRA-DQ8] lmhead   baked checkpoint detected (1 modules); load-time DQ8 skipped
```
…and per decode step, the MoE fast-path:
```
[ULTRA-DECODE] sorted_routes=48/48 (fallbacks: batch=0 size=0 disabled=0)
```
`48/48` and all-zero fallbacks = fully engaged (`~/.omlx/logs/server.log`, live 2026-07-06).
On the **un-baked** NVFP4 master the census reads `expected=96 actual=96` (…192/24/1)
instead of "baked … detected" (`~/.omlx/logs/server.log.2026-07-05:5752`). Counts:
mamba 96, moedense 192, attn 24, lmhead 1 (`omlx-ultra-550b.md`).

**GLM-5.2** — at load: `GLM MoE DSA native kernels available` (kernels present, not
fallen back), plus the `OMLX_M3_DEBUG_PATH` census lines. Detail → `models/*.md`.

---

## 4. Engine pool: load / evict / memory

The engine pool discovers all models, loads on demand (or `is_pinned` at startup), and
**evicts to fit** — it can host the active large model plus smaller ones within the memory
budget (§5). Switching which loaded model serves does NOT need a server bounce; the pool
loads the requested one and frees others as admission requires.

**Load-log format** (admission accounting) — `actual` vs conservative `estimated`:
```
Loaded model: Nemotron-3-Ultra-oQNVFP4-dq8 (actual: 305.08GB, estimated: 319.96GB, total: 319.96GB)
Loaded model: MiniMax-M3-oQNVFP4-fs5       (actual: 228.60GB, estimated: 239.83GB, total: 239.83GB)
```
(`~/.omlx/logs/server.log.2026-07-05:15168,19839`). `actual` is measured resident;
`estimated` is the pre-load admission estimate (always ≥ actual → conservative, won't
over-admit). Watch `omlx.process_memory_enforcer` for ceiling pressure during load; a
freed-memory double-count in admission was a real bug (commit 5a26eb1
"don't double-count freed memory in load admission").

---

## 5. Memory budget math (worked, this box)

512GB physical. With `sudo sysctl iogpu.wired_limit_mb=518144` (`env-setup.md` §7):

| Quantity | Value | Source |
|---|---|---|
| Metal wired cap (sysctl set) | **506.0GB** | `server.log.2026-07-05:5747` |
| Metal cap (sysctl UNSET, Apple default) | 464.0GB | `stock-v045.log` |
| Enforcer ceiling, balanced tier (sysctl set) | **~489–496GB** (boot-dependent) | `server.log.2026-07-05:5748` (496.3), `serve_211919.log:44` (493.4); low-pressure boots 489–492 |
| Enforcer ceiling (sysctl unset) | 464.0GB | `stock-v045.log` |
| macOS / non-Metal floor | ~6GB (512−506) | arithmetic |

Ultra resident **305.08GB** (§4) → headroom under the enforcer ceiling ≈ 490 − 305 ≈
**~185GB** for KV cache + prefill transients. Prefill transients are real: a one-time
~120GB phys-footprint jump on chunk 0 (expert-weight wiring) was mispredicted by the
throttle and clamped in a serving fix (`tasks/todo.md:1060`, SF-2). Set the sysctl before
loading any model >400GB.

---

## 6. Serve-window discipline

The rules that make a serve window trustworthy (full rationale in `laws.md`, method in
`profiling.md`):
- **Baseline leg first** — measure the current production config before any change.
- **ONE lever per restart** — never stack two env/code changes in a leg; you can't
  attribute the delta.
- **Engagement grep before believing tok/s** (§3). A gated fast path that silently
  disengaged (fp16-gate vs bf16-live) invalidated a whole M3 measurement set
  (`omlx-live-path-verification.md`).
- **Ledger every leg** to the campaign task file (env, model, ctx, tok/s, engagement
  evidence) — the ledgers in `tasks/*.md` are the institutional record.
- **Repeated-prompt TTFT is cache-served** — omlx has a ~62GB SSD prompt cache; use
  *decode* tok/s for A/Bs, not TTFT (`omlx-glm52-decode-opts.md`). Probe with a fresh
  nonce to force a real prefill.

---

## 7. Model swap procedure (ordered, with the why)

Use when you replace a model's **on-disk weights** (convert / overwrite). The Ultra DQ8
productization is the template (`tasks/todo.md:1099-1103`).

1. **Bounce the server FIRST.** The live server mmaps the safetensors; deleting or
   overwriting a model dir under a held mmap does not reclaim the space and risks serving
   half-written pages. Stop the server (§1) before touching the files. (Switching between
   *already-on-disk* models does NOT need this — that is a pool eviction, §4.)
2. **Copy/move the new weights** into `~/.omlx/models/<org>/<name>` (or archive-then-swap
   from T7; keep the old master until the new one is validated — bank before destructive,
   §8).
3. **Give it a NEW name if the weights changed.** The SSD prefix cache survives restarts and
   is keyed by model name — a same-name replacement serves stale cached prefixes. Ultra was
   renamed `…-oQNVFP4` → `…-oQNVFP4-dq8` "deliberate: avoids stale SSD-cache poisoning"
   (`tasks/todo.md:1103`; cross-ref `kv-cache.md`).
4. **Update `model_settings.json`** if keys changed (context, sampler, draft model) — bank
   the `.bak` first; the user approves live-server edits (`env-setup.md` §4).
5. **Relaunch** (§1) → **engagement grep** (§3) → **T1/T256 spot** decode probe to confirm
   speed and correctness before declaring the swap done.

---

## 8. Crew pipeline pattern (how this campaign shipped)

Multi-agent dev workflow (sources: `firstmate-herdr-setup.md`, `tasks/lessons.md`,
`tasks/todo.md:1067`, `subagents-use-opus.md`):
- **Pipeline:** strong architect (Fable/Opus) plans → peer review (a second model, e.g.
  Codex gpt-5.5) → implement (Codex, boring/fast) → cross-review (Fable+Opus) → tests →
  **LEAD runs the live legs on-metal** (only the human-owned lead touches `omlx:4`).
- **File-based handoffs:** subagents cannot see session messages — pass specs/results as
  files (`tasks/*.md`, job tmp). `codex exec` hangs on stdin unless the prompt is attached
  and stdin is closed: `codex exec … '<prompt>' </dev/null`; sanity-check the log grows
  (`tasks/lessons.md`).
- **Countermand discipline:** teammate messages cross in flight; a correction may arrive
  after a worker has moved on — re-read the latest before acting.
- **Bank before destructive:** `cp` the target (checkpoint, settings, binary) before any
  revert/delete; teammate messages can't establish user consent for production changes —
  the user approves (`tasks/lessons.md` 2026-07-04 OPS).
- Reviews catch real bugs static-only misses: the live-path law held repeatedly (runtime
  bugs surfaced only by running the server), and Codex/Fable cross-review caught a
  silently-dead DQ8 stage and a transient mispredict (`tasks/todo.md:1083-1087`).
- **Verify patches via the venv pytest** (never `uv run`): `$OMLX/.venv/bin/python -m pytest
  tests/ -k <area>` — a patch's test file names its area (`test_glm_moe_dsa_patch`,
  `test_mlx_lm_mtp_patch`, `test_nemotron_ultra_dq8_convert`, `test_scheduler*`). After any
  correction, APPEND the lesson to `tasks/lessons.md` (self-improvement loop).

## §9 Current production config (2026-07-08)
Launch: `env OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000
OMLX_MTP_DRAFT_K=2 OMLX_MTP_VERIFY_FAST=1 uv run omlx serve --log-level info` (tmux omlx:1, tee a log).
MTP context gate defaults to 2048 (`OMLX_MTP_MAX_CONTEXT`; 0 = ungated — bench only).
Memory guard (~/.omlx/settings.json, backup .bak-watermark): tier=custom ceiling 506GB,
soft 0.93 (470.6), hard 0.97 (490.8) — sized for 4x256k GLM agents (484GB). If a
"process memory limit exceeded" abort appears, first lever back: hard 0.97 -> 0.95.
preflight_guard=false fleet-wide (predictive admission off; real-usage guards active).
GLM settings: int8_mla_kv_enabled=true bits=8 start=4096, ctx 600k, mtp_enabled=true.
Engagement greps after warm request: `[INT8KV]`, `restore format=`, `[prefill-gate]`,
`MTP context-gated` / `MTP[2] chained`, `[restore-profile]`.

## §10 Puzzle-75B oQ48 — serving notes (one reload away from rotation)

Model at `~/.omlx/models/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-oQ48` (INTERNAL SSD — never the Clone
symlink, preflight §6), 43.84GB resident. **:8000 has NOT discovered it** (pool globbed pre-landing)
— needs admin-UI reload (`POST /admin/api/reload`, session-cookie auth) or a restart. **No
`model_settings.json` row** (GAP: alias, sampler defaults) — bank `.bak` first, user approves.
Launch line: plain golden env, NO Puzzle-specific switches:
```
env OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000 uv run omlx serve --log-level info
```
**Fusion pool A auto-engages** via `apply_post_load_transforms` (fingerprint-gated); kill switch
`OMLX_PUZZLE_DISABLE_FUSED_MAMBA=1`. Engagement grep after a warm decode: `[PUZZLE-FUSE] mamba=40/40`
with zero fallbacks (Law 1). Pools B (router, 7.3× loss) and C (experts, pipelined-unsafe) are NOT
wired — their files exist banked; do not add their switches expecting a win. Expected decode
**54.3–56.0 tok/s** (EXP-095). Spec/MTP: SHELVED — serve PLAIN; no `OMLX_MTP_*` vars apply to Puzzle
(no mlx_lm_mtp patch wired). Dossier: `models/nemotron-puzzle.md`.
