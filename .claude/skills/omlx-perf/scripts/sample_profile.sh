#!/bin/sh
# sample_profile.sh - wrap /usr/bin/sample to profile the live omlx model process.
#
# > Verified 2026-07-05 . Mac Studio M3 Ultra 512GB (819GB/s) . MLX 0.31.2 .
# omlx 0.4.5.dev1 . branch glm5.2-native-kernels-v0.4.5 (uncommitted tree).
# Measured here, not universal - re-verify after MLX/omlx upgrades.
#
# /usr/bin/sample needs NO sudo for your own processes. It samples every thread's
# call stack N times and prints a per-thread call tree with per-frame sample
# counts -- the honest host-side decode decomposition on this box.
#
# USAGE:
#   sh scripts/sample_profile.sh [-p PATTERN] [-s SECONDS] [-i INTERVAL_MS] [-o OUTFILE]
#     -p  process match pattern (default: omlx). The server child is often
#         renamed "omlx-server" and may show as python3.12 under uv; this script
#         picks the HIGHEST-RSS match, which is unmistakably the 300GB+ model.
#     -s  sample duration seconds (default 10 -- one steady-state decode window)
#     -i  interval milliseconds  (default 1)
#     -o  output file            (default ./omlx_sample.txt)
#
#   Run it WHILE a decode is in flight (fire a long generation first, e.g.
#   scripts/t1t256_probe.py in another shell), so the hot thread is busy.
#
# EXPECTED OUTPUT:
#   [sample] target pid=7225 rss=305GB pattern='omlx'
#   [sample] sampling 10s @1ms -> ./omlx_sample.txt
#   Sampling process 7225 for 10 seconds with 1 millisecond of run time between samples
#   Sample analysis of process 7225 written to file ./omlx_sample.txt
#   ... then the per-thread decomposition hint block below.

set -eu

PATTERN=omlx
SECONDS_ARG=10
INTERVAL=1
OUT=./omlx_sample.txt

while getopts "p:s:i:o:h" opt; do
  case "$opt" in
    p) PATTERN=$OPTARG ;;
    s) SECONDS_ARG=$OPTARG ;;
    i) INTERVAL=$OPTARG ;;
    o) OUT=$OPTARG ;;
    h) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "bad option; -h for help" >&2; exit 2 ;;
  esac
done

# Highest-RSS process whose args match PATTERN (skip the grep itself).
PID=$(ps -Ao pid,rss,args | grep -E "$PATTERN" | grep -v grep | \
      sort -k2 -n -r | head -1 | awk '{print $1}')

if [ -z "${PID:-}" ]; then
  echo "[sample] no process matching '$PATTERN' -- is the server up?" >&2
  echo "[sample] candidates:" >&2
  ps -Ao pid,rss,comm | grep -iE "python|omlx|uv" | grep -v grep >&2 || true
  exit 1
fi

RSS_KB=$(ps -o rss= -p "$PID" | tr -d ' ')
RSS_GB=$(( RSS_KB / 1024 / 1024 ))
echo "[sample] target pid=$PID rss=${RSS_GB}GB pattern='$PATTERN'"
echo "[sample] sampling ${SECONDS_ARG}s @${INTERVAL}ms -> $OUT"

/usr/bin/sample "$PID" "$SECONDS_ARG" "$INTERVAL" -f "$OUT"

cat <<'HINT'

------------------------------------------------------------------------------
DECOMPOSE THE DECODE THREAD (see ../profiling.md for worked GLM/Ultra splits):

1. The main thread (DispatchQueue_1: com.apple.main-thread) is the IDLE uvicorn
   kevent loop -- nearly all its samples are in kevent. IGNORE it.
2. The decode thread is the one whose tree is dominated by
   mlx::core::async_eval -> mlx::core::eval_impl. Find it:
       grep -n 'Thread_' OUTFILE                # list threads
       grep -n 'async_eval' OUTFILE             # decode thread lives here
3. Three buckets inside that thread (sample count / thread total = %):
   (a) GPU-WAIT / starvation:
         eval_impl -> std::condition_variable::wait -> __psynch_cvwait
   (b) METAL ENCODE (host CPU building/dispatching kernels):
         eval_impl CPU: gpu::eval, Matmul::eval_gpu -> gemv, binary_op_gpu,
         MetalAllocator, dispatch_threadgroups
   (c) PYTHON FORWARD (nn.Module tower):
         _PyEval_EvalFrameDefault / slot_tp_call chains ~20-40 frames deep,
         plus fast::metal_kernel call-setup for custom kernels
4. Tokenizer/detokenizer frames scatter across worker threads at <=7 samples --
   OFF the hot path; do not chase them.
5. Also note concurrent (NOT on the decode thread) Metal service threads:
   com.Metal.CompletionQueueDispatch / CommandQueueDispatch (~7% each) -- these
   are GPU work overlapping the decode thread, expected.
------------------------------------------------------------------------------
HINT
