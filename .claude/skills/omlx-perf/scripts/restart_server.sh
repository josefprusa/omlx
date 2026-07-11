#!/usr/bin/env bash
# Verified 2026-07-05 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1
#   · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal.
#
# restart_server.sh — TEMPLATE for the omlx serve-window restart choreography.
# Companion to ops-runbook.md §1-3. POSIX/zsh-compatible; verify with:  zsh -n restart_server.sh
#
# ############################################################################
# ##  DO NOT RUN while another operator owns the server (tmux session omlx,  ##
# ##  window 4, port 8000). This graceful-stops and relaunches THE server.   ##
# ##  Confirm you own the serve window before executing.                     ##
# ############################################################################
#
# What it does (ops-runbook.md §1): SIGINT the running server, poll until the process is
# gone, relaunch under `env <MODEL_ENV> uv run omlx serve` teed to a log, poll the log for
# startup-complete + model-loaded, then print the engagement-grep reminder (§3).
#
# Usage:
#   MODEL_ENV="<env prefix>" LOG=/path/to/serve.log ./restart_server.sh
# Defaults below launch the CURRENT PRODUCTION model (Nemotron Ultra, ops-runbook.md §2).

set -eu

# ---- Config (override via environment) -------------------------------------
OMLX="${OMLX:-$(git rev-parse --show-toplevel)}"

# Env-var prefix for the target model (ops-runbook.md §2). Default = Ultra production line.
# GLM golden: "OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000"
# M3 nvfp4 (REQUIRES OPS=500): "OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=500"
MODEL_ENV="${MODEL_ENV:-OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000 OMLX_ULTRA_DQ8_MAMBA=1 OMLX_ULTRA_DQ8_MOEDENSE=1 OMLX_ULTRA_DQ8_ATTN=1 OMLX_ULTRA_DQ8_LMHEAD=1}"

LOG="${LOG:-$OMLX/serve.log}"           # teed server output; poll target
SERVE_PATTERN="${SERVE_PATTERN:-omlx serve}"   # pgrep/pkill match
STARTUP_MARKER="${STARTUP_MARKER:-Application startup complete.}"
LOADED_MARKER="${LOADED_MARKER:-Loaded model:}"

# `uv run` re-syncs uv.lock (mlx==0.31.2). Set UV_FLAGS="--no-sync" if a CUSTOM MLX wheel
# is installed, or it will be reverted (env-setup.md §1; ~/mlx-src/SHELF.md).
UV_FLAGS="${UV_FLAGS:-}"

STOP_TIMEOUT="${STOP_TIMEOUT:-60}"      # seconds to wait for graceful shutdown
START_TIMEOUT="${START_TIMEOUT:-600}"   # seconds to wait for model load (big models ~1-3min)

# ---- 1. Graceful stop (ops-runbook.md §1) ----------------------------------
if pgrep -f "$SERVE_PATTERN" >/dev/null 2>&1; then
  echo "[restart] stopping running server (SIGINT to '$SERVE_PATTERN')..."
  pkill -INT -f "$SERVE_PATTERN" || true
  waited=0
  while pgrep -f "$SERVE_PATTERN" >/dev/null 2>&1; do
    if [ "$waited" -ge "$STOP_TIMEOUT" ]; then
      echo "[restart] SIGINT timed out after ${STOP_TIMEOUT}s; sending SIGTERM once..."
      pkill -TERM -f "$SERVE_PATTERN" || true
      sleep 5
      if pgrep -f "$SERVE_PATTERN" >/dev/null 2>&1; then
        echo "[restart] ERROR: server still alive. Stop it manually (do NOT blind SIGKILL" >&2
        echo "          a 300GB model mid-write). Aborting." >&2
        exit 1
      fi
      break
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "[restart] server stopped."
else
  echo "[restart] no running server matched '$SERVE_PATTERN'."
fi

# ---- 2. Relaunch, teed to log (ops-runbook.md §2) --------------------------
echo "[restart] launch env: $MODEL_ENV"
echo "[restart] logging to: $LOG"
: > "$LOG"    # truncate so the startup poll below can't match a stale marker
cd "$OMLX"
# shellcheck disable=SC2086  # MODEL_ENV and UV_FLAGS are intentionally word-split
env $MODEL_ENV uv run $UV_FLAGS omlx serve --log-level info >>"$LOG" 2>&1 &
SERVER_PID=$!
echo "[restart] launched (pid $SERVER_PID); waiting for startup..."

# ---- 3. Poll the log for startup-complete, then model-loaded ---------------
waited=0
while :; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[restart] ERROR: server process exited during startup. Tail of $LOG:" >&2
    tail -n 30 "$LOG" >&2
    exit 1
  fi
  if grep -qF "$STARTUP_MARKER" "$LOG" 2>/dev/null; then
    echo "[restart] uvicorn up ('$STARTUP_MARKER')."
    break
  fi
  if [ "$waited" -ge "$START_TIMEOUT" ]; then
    echo "[restart] ERROR: no '$STARTUP_MARKER' after ${START_TIMEOUT}s." >&2
    exit 1
  fi
  sleep 3
  waited=$((waited + 3))
done

# Model may be pinned (loads at startup) or on-demand. Wait a bounded time for the load line.
waited=0
while [ "$waited" -lt "$START_TIMEOUT" ]; do
  if grep -qF "$LOADED_MARKER" "$LOG" 2>/dev/null; then
    grep -F "$LOADED_MARKER" "$LOG" | tail -n 1
    break
  fi
  sleep 3
  waited=$((waited + 3))
done

# ---- 4. Engagement-grep reminder (ops-runbook.md §3) -----------------------
cat <<REMINDER
[restart] Server is up. BEFORE benching, verify engagement (ops-runbook.md §3):
  Ultra (baked):  grep -E 'baked checkpoint detected' $LOG            # expect 4 lines (96/192/24/1)
  Ultra (decode): grep -E '\[ULTRA-DECODE\] sorted_routes=48/48' $LOG # fallbacks must be 0
  GLM-5.2:        grep -F 'GLM MoE DSA native kernels available' $LOG
  Then: probe DECODE tok/s with a FRESH nonce (repeated prompts are SSD-cache-served).
REMINDER
