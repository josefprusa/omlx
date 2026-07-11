#!/usr/bin/env bash
# LEVER #1 Phase B — fs5 REAL parity window (ops contract, ONE chained command).
# Stop server -> standalone fs5 parity gate -> relaunch server (PRODUCTION env)
# -> health probe. NEVER leaves the server down. Fire ONLY on the lead's GO
# (server must be idle). Run in background:  bash run_parity_window.sh
set -u
REPO="${OMLX_REPO:-$(git rev-parse --show-toplevel)}"
cd "$REPO"
TS=$(date +%H%M%S)
PLOG="$REPO/tasks/compile_spikes/parity_$TS.log"
SLOG="$REPO/tasks/compile_spikes/serve_$TS.log"
NSTEPS="${1:-220}"

echo "[chain] stopping server (tmux omlx:4 C-c) ..."
tmux send-keys -t omlx:4 C-c

echo "[chain] waiting for 'omlx serve' to exit (free memory) ..."
for i in $(seq 1 300); do
  pgrep -f "omlx serve" >/dev/null 2>&1 || { echo "[chain] server exited after ${i}s"; break; }
  sleep 1
done
if pgrep -f "omlx serve" >/dev/null 2>&1; then
  echo "[chain] WARN server still up after 300s; sending C-c again"
  tmux send-keys -t omlx:4 C-c
  sleep 20
fi
echo "[chain] settling 10s for memory reclaim ..."
sleep 10

echo "[chain] === PARITY GATE (standalone) -> $PLOG ==="
OMLX_M3_SPARSE_MIN_K=4096 OMLX_M3_DEBUG_PATH=0 \
  uv run python "$REPO/tasks/compile_spikes/phaseb_real_parity.py" "$NSTEPS" 2>&1 | tee "$PLOG"
echo "[chain] parity exit: ${PIPESTATUS[0]}"

echo "[chain] === RELAUNCH server (PRODUCTION env) -> $SLOG ==="
tmux send-keys -t omlx:4 \
  "env OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000 uv run omlx serve --log-level info 2>&1 | tee $SLOG" Enter

echo "[chain] health probe (poll /health up to 600s) ..."
UP=""
for i in $(seq 1 600); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then UP="$i"; break; fi
  sleep 1
done
if [ -n "$UP" ]; then
  echo "[chain] server healthy after ${UP}s:"
  curl -s http://127.0.0.1:8000/health 2>/dev/null | head -c 400; echo
else
  echo "[chain] ERROR server NOT healthy after 600s — CHECK $SLOG / tmux omlx:4"
fi
echo "[chain] DONE. parity=$PLOG serve=$SLOG"
