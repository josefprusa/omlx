> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# profiling.md — measuring this box honestly

Decode on this box is **host-serialized, not purely bandwidth-bound**: the decode
thread spends ~half its time blocked waiting for the GPU while the CPU builds the
next token's graph, because MLX v0.31.2 pins `MAX_ACTIVE_TASKS=10` and the
one-ahead pipeline cannot run ahead (`tasks/ultra_speed.md §1`; mechanism in
`mlx.md`). Every speed claim in this campaign had to survive three
measurements; skipping any one produced a retracted number.

## The three measurement laws (full statements in `laws.md`)

1. **A streamed tok/s is not a decode rate.** Use the stream-free T1/T256 probe.
2. **A standalone microbench is not an in-stream saving.** Only ~50–60% of a
   naive delta survives pipelining (`tasks/lessons.md:10`); the Ultra sort lever's
   isolated 3.5 ms/token vanished live (`tasks/todo.md:1073`).
3. **"Verified standalone" is not "engaged live."** Require an expected-vs-actual
   counter in the server log before believing a fast path fired.

---

## 1. `/usr/bin/sample` — host-side decomposition (no sudo)

`/usr/bin/sample` needs no sudo for your own processes. It snapshots every
thread's stack N times and prints a per-thread call tree with per-frame sample
counts — the honest split of where host wall-time goes during decode.

**Recipe** (wrapper: `scripts/sample_profile.sh`, which finds the PID and prints
the decomposition hint):

```sh
# 1. Fire a long decode so the hot thread is busy (separate shell):
.venv/bin/python scripts/t1t256_probe.py --base-url http://127.0.0.1:8000 \
    --model <model> --prompt-tokens 5000 &
# 2. Sample the model process (highest-RSS match — the 300 GB+ one):
sh scripts/sample_profile.sh -p omlx -s 10 -i 1 -o omlx_sample.txt
```

Raw `sample` form if you skip the wrapper: `/usr/bin/sample <pid> 10 1 -f out.txt`
(pid, seconds, interval-ms). Find the PID by highest RSS — the server child is
often renamed `omlx-server` and can appear as `python3.12` under `uv`
(`tasks/lessons.md:87`), so match on memory, not name.

**Decompose the output** (`ultra_sample1.txt` line refs from `tasks/ultra_speed.md §1`):

- The **main thread** (`DispatchQueue_1: com.apple.main-thread`) is the idle
  uvicorn `kevent` loop — ~all samples in `kevent`. **Ignore it.**
- The **decode thread** is the one whose tree is dominated by
  `mlx::core::async_eval → mlx::core::eval_impl`. Find it: `grep -n async_eval out.txt`.
- Split that thread into three buckets (frame samples ÷ thread total = %):

| bucket | stack signature | what it is |
|---|---|---|
| **GPU-wait** | `eval_impl → std::condition_variable::wait → __psynch_cvwait` | decode thread blocked on the GPU (starvation) |
| **Metal encode** | `eval_impl` CPU: `gpu::eval`, `Matmul::eval_gpu→gemv`, `binary_op_gpu`, `MetalAllocator`, `dispatch_threadgroups` | host building/dispatching kernels |
| **Python forward** | `_PyEval_EvalFrameDefault` / `slot_tp_call` towers ~20–40 deep; `fast::metal_kernel` call-setup | nn.Module forward + custom-kernel binding |

Tokenizer frames scatter across worker threads at ≤7 samples — off the hot path.
Concurrent Metal service threads (`com.Metal.CompletionQueueDispatch`,
`CommandQueueDispatch`, ~7% each) are overlapping GPU work, expected.

**Worked examples** (both at golden env, sustained decode):

| model / run | tok/s (ms/tok) | GPU-wait | encode | python | source |
|---|---|---|---|---|---|
| Nemotron-Ultra baseline | 7.68 (130) | 46.5% (~60 ms) | 35.5% (~46 ms) | 17.1% (~22 ms) | `tasks/ultra_speed.md §1` (files `ultra_sample1.txt`, `ultra_sample2.txt`, thread `Thread_70414112`) |
| GLM-5.2 residual | 23.1 (43.3) | 53.6% | 17.5% | 12.5% | `tasks/todo.md:1014` |
| GLM-5.2-Alis (earlier) | 20.8 (48) | 69% (~33 ms) | 16.5% (~8 ms) | ~15% (~7 ms) | `tasks/todo.md:985` (file `glm_sample.txt`) |

Reading it: GPU-wait is the prize (host serialization to hide/kill); encode +
python are the host cost a fused-layer/`mx.compile` campaign would attack. The
Ultra evidence lines to look for: `async_eval` branch ~5008 samples at depth 26,
`condition_variable::wait` 2948 at depth 28–34, encode `gemv` 146 / `binary_op`
128 at depth 28, python tower ~1005 at depth 20 (`tasks/ultra_speed.md §1`).

---

## 2. Stream-free T1/T256 decode probe

`scripts/t1t256_probe.py`. Send the SAME prompt three times, non-streaming:
**prime** (`max_tokens=1`, warms the prefix cache), **T1** (`max_tokens=1`,
cached), **T256** (`max_tokens=256`, cached). Then:

```
decode_tok_s = (completion_256 − completion_1) / (wall_256 − wall_1)
             = (c256 − 1) / (t256 − t1)          # 255 tokens on the Ultra config
```

**Why this beats streaming timings.** Prefill, TLS handshake, request queueing,
sampler warmup and detokenization all appear in **both** T1 and T256, so they
subtract out — what remains is pure steady-state decode. `usage.completion_tokens`
is authoritative (no chunk-counting). Streamed timings, by contrast, are
corrupted by the reasoning-channel and one-chunk artifacts in §3.

```sh
.venv/bin/python scripts/t1t256_probe.py --base-url http://127.0.0.1:8000 \
    --model Nemotron-3-Ultra-oQNVFP4-dq8 --prompt-tokens 5000
# [5k] prompt=5007 prime=42.00s T1c=3.10s T256c=22.60s -> decode=13.07 tok/s
```

`--base-url` is **required with no default** — a missing default is the guard
against accidentally benchmarking a server someone else is live-testing. This is
the probe of record: the Ultra ladder (7.68 → 13.08 tok/s) was measured entirely
with it (`tasks/todo.md:1070`).

---

## 3. Why streaming TTFT lies — the `reasoning_content` trap

A client that computes TTFT / tok-s from `delta.content` alone **lies** on this
stack. Two independent mechanisms corrupt streamed timings:

1. **Reasoning channel.** Thinking models emit their reasoning into
   `delta.reasoning_content` first, switching to `delta.content` only for the
   final answer. A content-only counter sees seconds of phantom **"dead air"**
   before its first token. This was chased as an omlx serving bug and **refuted**:
   the tokens were streaming the whole time, in the reasoning channel (the
   channel + server behavior belong to `omlx.md`). Evidence that the channel is
   the normal thinking path: `tasks/todo.md:677`. Diagnose with `scripts/cadence_probe.py`.
2. **One-SSE-chunk artifact.** Nemotron-Ultra once delivered an entire 128-token
   answer as a **single** content chunk at completion (1 chunk @18.75 s) while
   sibling Super streamed 8 chunks at 0.134 s cadence — inflating a client-side
   number to a bogus "48 tok/s @5k" (`tasks/todo.md:1051`, `:1055`). **RESOLVED 2026-07-05 —
   same reasoning_content root cause as (1); not a serving defect.**

**The fix, used by every campaign probe:** count BOTH channels
(`delta.get("content") or delta.get("reasoning_content")`) and never derive
decode rate from stream chunk timestamps — use §2.

```sh
.venv/bin/python scripts/cadence_probe.py --base-url http://127.0.0.1:8000 \
    --model <thinking-model> --max-tokens 128
# [cadence] reasoning: chunks=63 first@0.48s last@8.90s ...
# [cadence] content:   chunks=1  first@8.95s last@8.95s ...
# NOTE: content first-token 8.95s >> reasoning 0.48s -> ~8.5s false dead air.
```

---

## 4. Engagement evidence — expected-vs-actual counters

A one-shot "`[X] ENGAGED`" log line is **insufficient**: it proves the code ran
once, not that the fast path fired on every layer/every step. The house rule
(`laws.md`; origin `tasks/lessons.md:17`): every gated fast path (shape / dtype
/ flag gate) ships a **per-layer expected-vs-actual counter** visible live, and
you **grep it before benching**. This law exists because an fp16-dtype gate let a
"bit-exact, engaged" M3 kernel fall back **silently** on the bf16 live model
(`fused_none=57/57` every step) — the very slope it was built to fix
(`tasks/lessons.md:17`, `tasks/todo.md:462`).

| counter (grep target) | meaning | source |
|---|---|---|
| `[ULTRA-DQ8] mamba expected=96 actual=96` | all 96 mamba mods quantized; **hard-fail on mismatch** | `tasks/ultra_speed.md:368` |
| `sorted_routes=48/48` | sorted-route path fired on all 48 MoE layers | `tasks/ultra_speed.md:433` |
| `[ULTRA-CHUNK] eval_chunk=27 chunks=4/4` | intra-token async_eval chunking engaged | `tasks/ultra_speed.md:446` |
| `M3CENSUS ... fused_hit=57/57` / `fused_topk=57/57` | M3 fused index/topk kernels engaged every layer | `tasks/todo.md:473`, `:584` |
| `pack_full=56 pack_qkv=3 pack_none=1` | packed-projection tiers match config prediction | `tasks/todo.md:604` |
| `[GLM-DKO] ENGAGED` / `BAIL` | GLM fused indexer scores fired vs bailed | `tasks/todo.md:494` |

Grep recipe (server log; port 8000 / tmux "omlx" are the user's — read logs, do
not drive the server):

```sh
grep -E 'ULTRA-DQ8|sorted_routes=|ULTRA-CHUNK|M3CENSUS|GLM-DKO' <server-log>
# every counter must read N/N (e.g. 96/96, 48/48, 57/57) BEFORE trusting an A/B.
```

Trap to remember: a fast-path gate can sit **upstream** of its own bail log, so a
`mask is None` (or dtype) skip fires with **no** log line at all — census the call
site, not just the kernel body (`tasks/todo.md:520`).

---

## 5. Reproducible-timing hygiene

### Golden env (what the vars do: `mlx.md`)

```sh
export MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000   # Ultra/GLM golden
```

Every timing on the big models runs under this env. The cost of forgetting it:
the **K1 kernel bench read 13.35× ideal** on its first run — confirmed a pure
harness artifact from a **missing golden env**, not a real cost
(`tasks/ultra_speed.md §"v2.2 changelog"`). (M3 shipped with `=500`; 500→2000 was
flat, golden 4000/4000 stands — `tasks/ultra_speed.md:516`.)

### Standalone (non-server) MLX scripts MUST set the wired limit

Any script that loads weights and calls `model(x, cache)` **outside** the server
must call `mx.set_wired_limit(506*1024**3)` before the forward:

```python
import mlx.core as mx
mx.set_wired_limit(506 * 1024 ** 3)   # 506 GB == the iogpu sysctl cap on this box
```

Without it, an isolated eager decode measured **~3100 ms vs ~48 ms in-loop — a
65× illusion** (`GLM52_MTP_FORAY.md:264`): unwired weights trigger a GPU
page-fault storm on every weight read (root cause `GLM52_MTP_FORAY.md:388`,
`tasks/todo.md:1020`). The omlx server raises the wired limit at startup
(`process_memory_enforcer`); a fresh main thread does not. The M3 EAGLE Gate-1
harness hit the identical trap. There is no hidden 65× to harvest — it is purely
the missing wired limit.

### Fresh-nonce discipline (prefix cache: `kv-cache.md`)

The prefix cache lives in **RAM and on SSD and survives server restarts**
(`tasks/todo.md:174`). A repeated identical prompt therefore serves a cached
prefill and **poisons repeat-prompt timings** (an exact-repeat 5k prompt went
from 40 s to 3.0 s TTFT once its entry landed — `tasks/todo.md:1075`,
`:508`). So:

- Every fresh-prefill timing prompt carries a **unique nonce** (the probes
  auto-generate `f"probe-{time.time_ns()}-..."`).
- An **exact-repeat cache-HIT test is a separate, deliberate** measurement — pass
  a fixed `--nonce` and run twice (that is how SF-1's 13× TTFT win was proven).

---

## 6. Ledger-keeping discipline

Every measured leg lands in `tasks/todo.md` / `tasks/ultra_speed.md` with three
fields, so a later reader can re-run it exactly and trust the number:

1. **Launch line** — the full env + model, e.g.
   `OMLX_ULTRA_DQ8_MAMBA=1 OMLX_ULTRA_DQ8_MOEDENSE=1 OMLX_ULTRA_DQ8_ATTN=1
   OMLX_ULTRA_DQ8_LMHEAD=1` at golden env (`tasks/todo.md:1077`).
2. **Engagement evidence** — the expected-vs-actual counters read N/N (§4),
   named in the leg ("all engagement-verified"), not assumed.
3. **Numbers** — tok/s at fixed context points (short / 6k / 20k), the probe used,
   and any resident-memory delta. Ladder-leg format:

```
leg1 +DQ8 mamba 96/96: 10.38 tok/s, -17.7GB     (stream-free T1/T256, short==5k)
leg2 +moedense 192/192: 12.52, -26.1GB
leg4 +lmhead 1/1: 13.07-13.08, 305.07GB
```
(`tasks/todo.md:1070`). A leg without all three fields is not evidence — it is a
number you cannot defend or reproduce.

**Name the harness (Law 17).** Every citation carries a 4th field: which script produced it
(`verify_econ`, `t1t256_probe`, `puzzle_decode_decomp`, a campaign bench...). Never compare or
borrow a number across harnesses — a conflated `verify_econ` t1=15.5ms briefly corrupted the
EXP-095 fusion verdict ("+19%" vs the real +2.9%) until traced back to its source script.

---

## 7. Methods banked, not yet ported to `scripts/`

The campaign measured DECODE (§2) and decode-M1 kernels (`kernel_parity_bench.py`). These
**prefill / long-ctx / serving-A-B** methods exist only as raw probes (vendored verbatim to
`scripts/archive/`, sanitize first — `scripts/archive/README.md`); port one before the
GLM-NVFP4 / batch≥2 / mxfp8-KV campaigns, which pay or hurt on exactly these axes:

- **Prefill per-component decomposition + prefill-TTFT A/B** — `scripts/archive/prefill_attn.py`
  (per-layer indexer/topk/sparse_mla vs MoE at long-ctx prefill), `bench_int8_prefill.py` (server
  prefill-TTFT A/B across context sizes). Prefill is otherwise UNMEASURED here.
- **Long-ctx correctness** — `scripts/archive/niah_bench.py` / `niah600.py` (multi-needle NIAH
  16k–128k, 3 depths, + perf); `quick_reason.py` (chain / aggregate / latest-state / transitive /
  temporal, auto-scored). `acc_bench_serial.py` covers only SHORT gsm8k/mmlu/arc — any long-ctx or
  KV-quant work must prove retrieval + reasoning survive at length.
- **Serving token-identity sha256 A/B** — `scripts/archive/ov_capture.py` (fixed prompt+nonce,
  greedy temp0, `sha256(full_output)`, writes text for diff); `mtp_identity_probe.py`. The "did my
  change alter ANY output bit over the LIVE server?" gate every kill-switch A/B wants (offline
  tensor bit-parity is not the same check).
- **Cross-request cache-contamination + negative control** — `scripts/archive/ov_reuse_niah.py`
  (A seeds a bucket, a SHORTER B reuses it and must retrieve ITS OWN needles — retrieval-based, so
  nondeterminism-immune) + `compile_spikes/spike_phaseb_reuse.py` (offline reseed-on-vs-off negative
  control). Runnable derivative of Law 7; also in `kv-cache.md §7`.

---

### Sibling cross-references (do not duplicate their content)
- `laws.md` — the in-stream / live-engagement / no-standalone-ceiling laws in full.
- `mlx.md` — what `MLX_MAX_OPS_PER_BUFFER` / `MAX_ACTIVE_TASKS` do; `set_wired_limit` mechanics.
- `omlx.md` — the `reasoning_content` channel and SSE serving behavior.
- `kv-cache.md` — prefix cache (RAM + SSD) internals and the SF-1 commit-lag fix.
- `scripts/kernel_parity_bench.py` — the standalone in-stream kernel A/B harness this file's laws govern.
