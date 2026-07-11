> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# env-setup — zero-to-serving on this box

Setup and environment discipline for the omlx inference server on the 512GB M3 Ultra.
For the restart/bench/swap loop see `ops-runbook.md`. For the ready-to-run relaunch
template see `scripts/restart_server.sh`.

Repo: `$OMLX` (call it `$OMLX`). Venv: `$OMLX/.venv`.

> **GOVERNANCE — omlx is LOCAL-ONLY.** Never `git push`, never open a PR / pull request, no
> destructive git (force-push, hard-reset shared refs, branch deletion). No PRs to GitHub — the
> captain does not want them. The campaign record is `tasks/*.md` + `memory/*.md` (git-untracked;
> their distilled truth lives in this skill). Bank before any destructive fs/config action; the
> **user** approves production changes (`memory/firstmate-herdr-setup.md`).

---

## 1. uv / venv discipline — read this first (it has bitten us)

The venv is a `uv`-managed project env. **Two invocation traps:**

1. **Bare `uv run` and `uv sync` re-sync the venv to `uv.lock`** (which pins
   `mlx==0.31.2`). If a *custom* MLX wheel is installed (the shelved cap wheel, or any
   source rebuild), a bare `uv run` silently **reverts it to the stock wheel**
   (`~/mlx-src/SHELF.md` §"Restore to stock": "Plain `uv run omlx serve` (no --no-sync)
   re-syncs the venv to `mlx==0.31.2` … and restores stock libmlx automatically").
2. In **production today the reversion is a no-op** because production runs *stock* mlx
   0.31.2 + golden env (`omlx-glm52-decode-opts.md` 2026-07-04 "prod verified stock mlx +
   golden env"). That is why every ledger launch line is a plain `uv run omlx serve`
   (`tasks/todo.md:1076`, `tasks/eagle3_build.md:5`). Plain `uv run` is only dangerous
   when a custom wheel is in play.

**Sanctioned invocations**

| Situation | Command |
|---|---|
| Run any Python (probes, checks) | `$OMLX/.venv/bin/python …` |
| Pip op that must NOT be reverted | `$OMLX/.venv/bin/pip install --no-deps …` |
| Serve, stock mlx (production today) | `env <MODEL_ENV> uv run omlx serve --log-level info` |
| Serve WITH a custom MLX wheel installed | `env <MODEL_ENV> uv run --no-sync omlx serve …` (`~/mlx-src/SHELF.md` §"Install + run": "MUST launch with --no-sync") |

> Agent-safety note: if you are an assistant, not the operator, do **not** run bare
> `uv run`/`uv sync` — use `$OMLX/.venv/bin/python` or drive the server the operator
> already has. A stray sync can revert a live custom wheel.

**Check what is installed** (read-only, all verified working here):
```
$OMLX/.venv/bin/python -c "import mlx.core as mx; print('mlx', mx.__version__)"   # -> mlx 0.31.2
$OMLX/.venv/bin/pip list | grep -iE '^mlx|nanobind|^omlx|mlx-lm|mlx-vlm'
```
Expected (2026-07-05): `mlx 0.31.2`, `mlx-metal 0.31.2`, `nanobind 2.12.0`,
`mlx-lm 0.31.3`, `mlx-vlm 0.6.3`, `omlx 0.4.5.dev1` (editable at `$OMLX`).

---

## 2. MLX: pinned 0.31.2 + the custom-kernel rebuild

Stock `mlx-metal==0.31.2` is the pin. omlx ships **native Metal extensions** (GLM MoE DSA
sparse attention + MiniMax-M3) as compiled `.so`s inside the package:
`omlx/custom_kernels/{glm_moe_dsa,minimax_m3}/_ext.cpython-312-darwin.so`.

### The ABI story (why a rebuild is ever needed)
The exts are keyed to nanobind's **NB_INTERNALS_VERSION** (binary tag
`vNN_system_libcpp_abi1`), *not* the package version. `mlx-metal 0.31.2` ships **v19**;
`nanobind 2.12.0 == v19` is the unique match (2.11=v18, 2.13=v20)
(`omlx-glm52-native-kernels.md` "Root cause"). If the ext is built against the wrong
nanobind (e.g. 2.13.0 → v20), the GLM/M3 kernels crash or silently fall back to pure-MLX.

**Pre-load ABI gate** (must equal mlx core's `v19`):
```
strings $OMLX/omlx/custom_kernels/minimax_m3/_ext.cpython-312-darwin.so \
  | grep -oE 'v[0-9]+_system_libcpp_abi1' | sort -u        # -> v19_system_libcpp_abi1
```
Verified `v19_system_libcpp_abi1` on this tree 2026-07-05. Same probe works on the
glm_moe_dsa ext.

### Rebuild recipe (only when the ext is missing / ABI-mismatched / after an MLX bump)
From `omlx-glm52-native-kernels.md` "BUILD" + "Fix":
```
$OMLX/.venv/bin/pip install nanobind==2.12.0          # pin FIRST (v19)
cd $OMLX && rm -rf build
CMAKE_ARGS="-DPython_EXECUTABLE=$OMLX/.venv/bin/python" \
  OMLX_WITH_CUSTOM_KERNEL=1 \
  $OMLX/.venv/bin/python setup.py build_ext --inplace --with-custom-kernel
# builds BOTH glm_moe_dsa + minimax_m3 exts
```
Then re-run the ABI probe (§above) — it must print `v19`. `pyproject.toml`
`build-system.requires` now pins `nanobind==2.12.0` (was unpinned → drift footgun; only
`--no-build-isolation` had been saving it) (`omlx-glm52-native-kernels.md` "Fix").
One-time prerequisite already done on this box: `xcode-select` → full Xcode +
MetalToolchain installed.

To match the branch's mlx-lm pin without disturbing mlx 0.31.2:
`$OMLX/.venv/bin/pip install --no-deps --force-reinstall "mlx-lm @ git+…@2ed22318…"`
(`omlx-glm52-native-kernels.md` "BUILD"). Installed here: mlx-lm 0.31.3.

**Kill switches** (A/B the native path off): `OMLX_GLM_DISABLE_NATIVE=1`
(`omlx-glm52-native-kernels.md` "Fix"). If the exts are absent the server logs "GLM MoE
DSA native kernels available" *only when present* — its absence is the fallback signal.

### `ov_KILL_SWITCH.md` — GAP
The shelved MLX cap wheel ledger says it was banked "at ~/mlx-src + SHELF.md +
ov_KILL_SWITCH.md" (`tasks/overlap_levers.md:110`, `tasks/todo.md:1008`). **That file does
not exist** at `~/mlx-src` or anywhere in the repo (searched 2026-07-05). Its content is
not lost: the cap-wheel kill-switch lives in `~/mlx-src/SHELF.md` §"Kill switch /
behavioral off" — "unset `MLX_MAX_ACTIVE_TASKS` (or =10) → bit-for-bit stock". Treat
SHELF.md as the source of truth; `ov_KILL_SWITCH.md` is a dangling reference.

The cap wheel itself (`MLX_MAX_ACTIVE_TASKS` env-configurable, v0.31.2 stage-2, ABI-safe)
is SHELVED — inert at batch-1, insurance for batch≥2. Full story: `~/mlx-src/SHELF.md`.
Do not install it for single-request serving.

---

## 3. Model directory layout

Models live under `~/.omlx/models/<org>/<name>` (org dir groups a publisher). Some entries
are **symlinks to the T7 external drive** (`… -> '$OMLX_COLD_STORAGE/omlx-models'/…`) —
those are cold/archival; serving-critical models are on internal SSD. Never serve the big
production models off USB (see §8; cross-ref `gotchas.md`).

Current inventory (`ls ~/.omlx/models/`, 2026-07-05) — production-relevant:

| Path | Disk | Role |
|---|---|---|
| `unigilby/Nemotron-3-Ultra-oQNVFP4-dq8` | 327G | **PRODUCTION** (13.07–13.09 tok/s; 305G resident) |
| `unigilby/MiniMax-M3-oQNVFP4-fs5` | 246G | M3 production variant (228.6G resident) |
| `avlp12/GLM-5.2-Alis-MLX-Dynamic-3.5bpw` | 311G | GLM production |
| `unigilby/Nemotron-3-Super-oQNVFP4`, `MiniMax-M3-oQ*`, `inferencerlabs/GLM-5.2-MTP-MLX-Q4` | — | other served/experimental |

(`du -sh` on the internal dirs; the org/name list is the raw `ls`.)

**A served model dir MUST contain** (verified against
`unigilby/Nemotron-3-Ultra-oQNVFP4-dq8`):
- `config.json` — arch/quant config
- `model.safetensors.index.json` — shard map (present)
- `model-000NN.safetensors` — weight shards
- tokenizer set: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
- `generation_config.json`, `chat_template.jinja`
- model-specific extras allowed (e.g. `ultra_v3_reasoning_parser.py`)

---

## 4. model_settings.json anatomy

File: `~/.omlx/model_settings.json` (JSON, `version: 1`, top-level `models: {}` keyed by
model name). Timestamped `.bak-*` copies sit beside it — **bank before editing**. Per-model
keys (real example, the GLM production entry):
```jsonc
"GLM-5.2-Alis-MLX-Dynamic-3.5bpw": {
  "max_context_window": 400000, "temperature": 1, "top_p": 0.95,
  "model_type_override": "llm",          // forces engine=batched (see startup log)
  "enable_thinking": true,
  "turboquant_kv_enabled": false, "turboquant_kv_bits": 4,
  "specprefill_enabled": false, "mtp_enabled": false, "vlm_mtp_enabled": false,
  "force_sampling": false, "is_default": false,
  "active_profile_name": "glm52"   // reasoning_parser NOT stored here — resolves via this profile

}
```
Common keys: `max_context_window`, `max_tokens`, sampler defaults, `turboquant_kv_*`
(KV quant), `specprefill_*` (draft model), `dflash_*`, `mtp_enabled` / `vlm_mtp_enabled`
(speculative decode), `is_pinned` (load at startup), `is_default` (serve if no model
named), `trust_remote_code`, `model_type_override`.

**The `force_sampling` trap** (one line; detail in `gotchas.md`): `force_sampling: true`
plus `temperature: 1.0` (the MiniMax-M3 entries) *overrides the request temperature before
any temp==0 guard* — it silently disabled every EAGLE/MTP drafter in every bench because
the temp0 spec-decode gate never saw temp0 (`omlx-glm52-decode-opts.md` 2026-07-04 LATE
CORRECTION). If you A/B speculative decode, check `force_sampling` first.

`vlm_mtp_enabled` (+ `vlm_mtp_draft_model`, `vlm_mtp_draft_block_size`) gates the VLM EAGLE
path (MiniMax-M3-oQNVFP4-fs5 has the draft wired but `vlm_mtp_enabled: false`). Editing
this file on the LIVE server is a production-config change — an assistant cannot self-
consent to it; the **user** must approve (`tasks/lessons.md` 2026-07-04 OPS).

---

## 5. tmux + launch convention

Server runs in tmux session **`omlx`, window 4** (`omlx:4`). Launch with stdout teed to a
log, then **poll the log for startup-complete before probing** — do not bench a half-loaded
server. Startup markers, in order (from `~/.omlx/logs/server.log.2026-07-05`):
```
omlx.engine_pool - INFO - Discovered 19 models
omlx.server - INFO - Server initialized with 19 models
omlx.server - INFO - Default model: NVIDIA-Nemotron-3-Super-120B-A12B-oQ4e
omlx.process_memory_enforcer - INFO - Metal wired limit raised: 0.0GB -> 506.0GB …
INFO:     Application startup complete.                 # uvicorn — server is up
omlx.engine_pool - INFO - Loaded model: <name> (actual: …GB, estimated: …GB, total: …GB)
```
Poll pattern: grep the teed log for `Application startup complete.` (server accepting) and
`Loaded model:` (target model resident). The server discovers ALL models and loads on
demand / on request; `is_pinned` models load at startup. `scripts/restart_server.sh`
implements this poll loop.

> OFF-LIMITS to assistants: session `omlx` and port 8000 are the user's live test bed.
> Never send tmux keys or HTTP to :8000 while the user is testing.

---

## 6. API key handling

- API key file: `$OMLX_API_KEY_FILE` (path only —
  never read or print it).
- The server enables auth at startup (`omlx.server - INFO - API key authentication:
  enabled`, `~/.omlx/stock-v045.log`).
- Bench/probe scripts take **`--api-key-file <path>`** and read it themselves — never inline
  the key on a command line or in a log.

---

## 7. sysctl: iogpu.wired_limit_mb (the USER runs sudo)

House rule: the **user** runs `sudo` themselves; an assistant never does.
```
sudo sysctl iogpu.wired_limit_mb=518144
```
Effect: raises the Metal wired cap **464GB (Apple default) → 506GB**, i.e. +42GB usable for
KV/long context (`tasks/todo.md:227`). Live confirmation in the log:
`Metal wired limit raised: 0.0GB -> 506.0GB (target=506.0GB, iogpu sysctl cap=506.0GB)`
(`~/.omlx/logs/server.log.2026-07-05:5747`).

If UNSET, the server warns at startup and stays at Apple's 464GB cap:
`Metal cap (464.0GB …) is below the oMLX static ceiling (506.0GB); … Raise it with: sudo
sysctl iogpu.wired_limit_mb=518144` (`~/.omlx/stock-v045.log`). The process memory enforcer
then guards a *balanced-tier ceiling* just under the cap (**~489–496GB** boot-dependent with the sysctl
set; 464GB without) — see `ops-runbook.md` §"Memory budget math". Set the sysctl before
loading any >400GB model.

---

## 8. Archive conventions — T7 "M3 Max Clone" (BACKUP DRIVE)

External T7 mounts at `$OMLX_COLD_STORAGE/` (4TB, ~1.7T free). It is a **backup drive**:
- `$OMLX_COLD_STORAGE/omlx-models/` — cold model store (symlink targets from
  `~/.omlx/models/`).
- `$OMLX_COLD_STORAGE/omlx-quant-work/` — quant masters + NVFP4 masters
  (`Nemotron-3-Ultra-NVFP4`, `MiniMax-M3-NVFP4`, …) and conversion scripts
  (`tasks/todo.md:1099` "NVFP4 master archived on T7").

Rules: **write only NEW folders under `omlx-quant-work/`** — "only our new folder touched"
(`tasks/todo.md:758`). **Never serve the big production models from USB** (omlx#2098;
incident detail in `gotchas.md`). The production Ultra/GLM/M3 models live on internal SSD
precisely so serving never depends on the USB bus.

## Side venv `.venv-mlx032` (MLX 0.32.0 A/B) — outside-cwd install trap

Exists beside the production `.venv` (mlx 0.32.0 + git-pinned mlx-lm @2ed2231 + transformers 5.12.1),
for A/B benching only; production stays 0.31.2. **Trap:** `pyproject.toml` `[tool.uv]
override-dependencies` forces mlx==0.31.2 on any uv install run from inside the repo cwd, even when
targeting the side venv's interpreter — install from OUTSIDE the repo. Keep transformers 5.12.x
everywhere our git-pinned mlx-lm loads (5.13 breaks tokenizer registration; EXP-093).
