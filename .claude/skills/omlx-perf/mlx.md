> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# MLX 0.31.x operational facts

Operational truths about MLX (Apple's array framework) that the omlx speed campaign
depended on. Each is sourced to MLX source in this repo's venv
(`.venv/lib/python3.12/site-packages/mlx/…`) and/or a campaign ledger, and where cheap,
re-checked live with `.venv/bin/python`. **Never run `uv run` / `uv sync`** — use
`.venv/bin/python` from the repo root (see "uv-sync wheel-revert trap" below).

MLX paths below are relative to `MLX=.venv/lib/python3.12/site-packages/mlx`.

---

## Scheduler & the MAX_ACTIVE_TASKS=10 cap

MLX runs one worker thread per stream and tracks outstanding GPU tasks in a single counter
`Scheduler::n_active_tasks_`, incremented in `notify_new_task`, decremented in
`notify_task_completion`; `wait_for_one()` blocks a caller until the count drops
(`$MLX/include/mlx/scheduler.h:80-113`). There is **no hardcoded cap in the shipped headers** —
the counter is just a counter there.

The cap `MAX_ACTIVE_TASKS = 10` is a **compile-time constant in `transforms.cpp`** (compiled
into `libmlx.dylib`, not exposed in headers or as a string). Verified against MLX v0.31.2
source: `eval_impl` blocks any caller — **including `mx.async_eval`** — once 10 tasks are
outstanding (`tasks/todo.md` §HOST-SERIALIZATION PROFILING, ~L958-960).

**What hitting it looks like:** the decode thread parks in `cond_wait` inside `mx.async_eval`
(measured 52% of decode-thread time; `tasks/todo.md` ~L954). omlx's `BatchGenerator`
one-ahead `async_eval` exists to overlap host with GPU but **cannot run ahead** past the cap,
so no overlap materializes (`tasks/todo.md` ~L960). In a `/usr/bin/sample` profile this
shows up as thousands of backpressure-wait samples.

**The 10→64 experiment (definitive, DEAD lever):** a patched `libmlx.dylib` raising the cap
10→64 killed the backpressure wait **entirely (8550→0 samples)** yet **tok/s stayed flat**
(28.27→28.05 short-context; *worse* at 16k) and GLM output was byte-identical across
stock/cap10/cap64 (`tasks/overlap_levers.md` ~L96-98). Verdict: the wait was already
overlapped / off the critical path; the real wall is main-thread Python graph-build (~95%
busy) + encode + GPU. Production ships **stock MLX**, not the patched wheel. Full experiment:
see `experiments/` (sibling).

---

## Command-buffer limits: MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER

Both are read **once, C-side, via `get_var` into a `static` at first use**
(`$MLX/include/mlx/utils.h:154-163`) — so **set them in the ENV before the process starts**;
mutating `os.environ` after `import mlx` does nothing. They are paired on the Metal device as
`(max_ops_per_buffer_, max_mb_per_buffer_)` (`$MLX/include/mlx/backend/metal/device.h:152-153,226-227`).

| Env var | Bounds | Fleet golden (final) | M3 early (superseded) |
|---|---|---|---|
| `MLX_MAX_OPS_PER_BUFFER` | # ops batched into one Metal command buffer before force-commit | `4000` | `500` |
| `MLX_MAX_MB_PER_BUFFER`  | estimated **MB of a command buffer** before force-commit | `4000` | (default → now `4000`) |

**What the MB accounting counts:** the campaign found empirically that **MB=4000 (4GB) ≈ one
M3 decoder layer's weights → MLX commits roughly per-layer**; raising it batches more ops per
buffer (fewer commits) but grows peak resident memory (more live intermediates) — MB=8000 gave
+49GB peak, MB=50000 breached the memory cap (`tasks/overlap_levers.md` ~L90-93). So MB bounds
a buffer's memory footprint, not just its op count.

**Golden values `4000/4000`** are the fleet standard, **M3 included as of 2026-07-04**
(`tasks/ultra_speed.md` §baseline, L5; `memory/omlx-glm52-decode-opts.md:183`). M3 originally
shipped `MLX_MAX_OPS_PER_BUFFER=500` (+~1% over the arch default, `tasks/todo.md` ~L564-566) but
golden `4000/4000` supersedes it — the MB=4000 cap also prevents the M3 nvfp4 cmd-buffer timeout
(`models/minimax-m3.md` §4).

**Beyond golden = DEAD lever:** resizing 500→2000 was flat on M3 (`tasks/todo.md` ~L961);
golden 4000/4000 stands as the sole survivor (`tasks/ultra_speed.md` ~L515).

**Missing the golden env is a HARNESS ARTIFACT that inflates benches:** the K1 gather_qmm
bench's first run reported **13.35×** inflation purely from a missing golden env; under
golden it was 1.87× (`tasks/ultra_speed.md` ~L18-21). **Always set the golden env for any
standalone quantized-matmul timing** (the parity bench script warns if it is unset).

---

## mx.compile for decode: VERDICT DEAD (three attempts)

`mx.compile` is **bit-identical and fully traceable** for the decode path — but its batch-1
tok/s win is **dead** because the MAX_ACTIVE_TASKS backpressure it would relieve is already
overlapped (see above). **Three separate attempts, all flat:** GLM foray §10b **1.06×**;
M3 A/B flat; GLM A/B flat (`tasks/ultra_speed.md` ~L513). Offline every spike PASSED and a
real-fs5 parity gate was bit-identical, but the standalone step-time was only **1.09×
(48.3→44.3 ms/tok)** and did not even reach the production MTP path
(`tasks/compile_spikes/PHASE_B_STATUS.md` §Step-time preview). Full spike record: `experiments/`.

**Mechanism rules (learn these before ever compiling a decode step)** —
sourced to `tasks/compile_spikes/PHASE_A_SUMMARY.md` (spikes A–E) unless noted:

| Rule | Consequence | Fix |
|---|---|---|
| Cache-keys on Python **scalar arg VALUES** | passing per-step `total_len`/offset as a Python `int` **re-traces every token** → kills the host win **and leaks** (unbounded compile-cache growth) | pass those scalars as `mx.array` inputs (constant shape) ⇒ exactly **1 trace per shape bucket** (Finding 1) |
| **Side effects run once PER TRACE**, not per call | census/telemetry counters inside a compiled fn undercount 50→1 (fired 1× over 50 calls) | keep counters on the **uncompiled** fallback path (Finding 3) |
| **Shape buckets recompile** | a new cache-capacity bucket = a fresh `mx.compile` (~11ms, 43µs/token amortized at 60 layers) | rebucket on 256-growth; `shapeless=True` matches on count/ndim/dtype (`tasks/todo.md` ~L555) |
| **State must flow via `inputs=`/`outputs=`** | a mutating KV cache works: `slice_update(buf, upd, start_indices=off.reshape(1))` with `off` an `mx.array` (Spike C, parity 4.8e-7) | carry KV+offset as compiled state; offset an `mx.array` |
| Primitives don't fuse; **surrounding elementwise does** | Spike E: 4931→3007 ops (39% fewer), `QuantizedMatmul` count unchanged, Broadcast/Reshape/Multiply collapse | lazy eval alone does **not** fuse — only compile does |

`single-request offline PASS ≠ multi-request-live correct`: a per-bucket KV buffer reseeded
only on growth leaked a prior request's KV into a shorter next request — caught only by a
cross-request temp0 token-identity gate (`tasks/lessons.md` ~L90-96).

---

## Quantization modes matrix

From `mx.quantize.__doc__` (the `quantize-modes` table) — re-print with
`.venv/bin/python -c "import mlx.core as mx; print(mx.quantize.__doc__)"`:

| mode | group_size | bits | scale type | bias | tuple len |
|---|---|---|---|---|---|
| `affine` | **32, 64\*, 128** | 2,3,4\*,5,6,8 | same as input | **yes** | 3 (w,scales,biases) |
| `mxfp4`  | **32\*** | 4\* | e8m0 | no | 2 (w,scales) |
| `mxfp8`  | **32\*** | 8\* | e8m0 | no | 2 |
| `nvfp4`  | **16\*** | 4\* | e4m3 | no | 2 |

`*` = default when unspecified. `mx.dequantize`/`gather_qmm` must be called with the **same**
`(mode, group_size, bits)`. Last dim of `w` must be divisible by `group_size`.

**Live-verify the constraints** (tiny, no weights — encouraged before trusting a container):
```
.venv/bin/python -c "import mlx.core as mx; w=mx.random.normal((64,128)).astype(mx.bfloat16)
for gs in (16,32,64,128):
 try: mx.quantize(w,group_size=gs,bits=4,mode='affine'); print(f'affine gs{gs}: OK')
 except Exception as e: print(f'affine gs{gs}: REJECT {str(e)[:50]}')"
# affine gs16: REJECT [quantize] The requested group size 16 is not supported.
# affine gs32/64/128: OK      nvfp4 requires gs16 (gs32 -> REJECT)
```
This is why the "exact-container" idea `nvfp4→affine5-gs16` was dropped: **affine has no
gs16** (`tasks/ultra_speed.md` ~L96-98).

**Determinism:** `mx.quantize` is **deterministic given identical inputs + config** — the
converter's bit-parity gate relies on it, asserting `mx.array_equal` on `(weight, scales,
biases)` produced by two independent call paths (converter vs `nn.QuantizedLinear.from_linear`)
over both f32 and bf16 inputs (`tests/test_nemotron_ultra_dq8_convert.py:115-142`,
`TestBitParity`). If a future MLX changes rounding on one path only, this test desyncs the
baked vs load-time checkpoints — do not delete it.

---

## Kernel parity: nvfp4 ≈ affine4 ≈ mxfp4 at fat shapes, but not small ones

Measured with the M1 bench (`scripts/kernel_parity_bench.py`, sanitized from the campaign's
`m1_kernel_bench.py`; 200 iters, M=1, Ultra's exact expert shapes E=512 top-22):

| matrix | nvfp4 gs16 | affine4 gs64 | mxfp4 gs32 | bandwidth-ideal |
|---|---|---|---|---|
| fc1 (129.8MB) | 408.2µs | 404.7µs | 395.1µs | 158.5µs |
| fc2 (129.8MB) | 416.5µs | 423.6µs | 401.3µs | 158.5µs |
| **layer total** | **824.7µs** | 828.3µs | 796.4µs | **316.9µs** |

**Finding 1 — no mode gap: all three within ±4%** at Ultra's fat latent-2048 shapes
(`tasks/ultra_speed.md` ~L241-248). ⇒ requant to any other 4-bit container buys nothing.

**But the gap is real at SMALL shapes:** Nemotron-Super oQNVFP4 decode ran **0.58×** its
affine sibling because nvfp4-gs16 does **4× the scale reads** of affine-gs64, and at
Super's small MoE shapes scale traffic dominates (`tasks/oqnvfp4_nemotron.md` ~L324,L331).
The tax vanishes at Ultra's fat shapes. Rule of thumb: **gs16's scale-read tax bites when
the weight tile is small relative to its scales.**

**Finding 2 — absolute efficiency in ISOLATION looks poor (~38% of peak)** but that is a
harness trap: eval-per-op standalone timings inflate; only **~50-60% of a naive delta
survives in-stream pipelining** (`tasks/ultra_speed.md` ~L252-257, `tasks/lessons.md` ~L1-13).
Always attribute in-stream (`mx.depends`-chained) before trusting an isolated number.
Full experiment: `experiments/` (sibling).

---

## gather_qmm contracts (the batched-MoE workhorse)

Signature: `mx.gather_qmm(x, w, scales, biases=None, *, rhs_indices, transpose=True,
group_size, bits, mode, sorted_indices=False)`. Ground truth = how omlx calls it
(`omlx/patches/glm_moe_dsa/switch_layers.py:88-103` `QuantizedSwitchLinear.__call__`;
`omlx/patches/nemotron_ultra_decode/moe_fastpath.py`).

**Shapes / output axis (verified live, this venv):** for a single decode row
`x=[1,1,IN]` with `rhs_indices=[1,TOP]`, output is **`[1, TOP, 1, OUT]`** — the **gathered
experts land at axis 1** (the TOP/route axis). Callers then `y.squeeze(-2)` →
`(y*scores[...,None]).sum(axis=-2)` to weight-sum over routes
(`moe_fastpath.py:150-155`). Reproduce:
```
.venv/bin/python -c "import mlx.core as mx; w=(mx.random.normal((8,10,32))*0.02).astype(mx.bfloat16)
wq,sc=mx.quantize(w,bits=4,group_size=16,mode='nvfp4'); x=mx.random.normal((1,1,32)).astype(mx.bfloat16)
idx=mx.random.randint(0,8,(1,3)).astype(mx.uint32)
print(mx.gather_qmm(x,wq,sc,rhs_indices=idx,transpose=True,group_size=16,bits=4,mode='nvfp4').shape)"
# (1, 3, 1, 10)
```

**The `(M,TOP)` broadcast trap:** at batch ≥2 `rhs_indices` must be **`(M, 1, TOP)`**, not
`(M, TOP)` — the flat form raises `[broadcast_shapes] Shapes (M) and (M,TOP) cannot be
broadcast` (`tasks/ultra_speed.md` ~L58-59,L259). Batch-1 (`[1,TOP]`) is the only measured
regime; batch≥2 is future work.

**`sorted_indices=True`:** may pick a faster kernel when `rhs_indices` is ascending. It is
**correctness-neutral at B=1**: with a single decode row there is one query broadcasting
against the gathered experts, so permuting `(inds, scores)` jointly is exact — top-22 indices
are distinct, so the sorted precondition holds (`tasks/ultra_speed.md` ~L416-427). Empirically
the sorted-flag output equals the unsorted-then-reordered output and is **bit-identical to
passing pre-sorted indices without the flag** (campaign probe `sorted_parity_realdims.py`:
`flag-vs-noflag bit-identical, finite`). ⚠ Only safe at B=1 — batching changes float
accumulation order.

**`do_sort` threshold = 64:** omlx sorts routes only when `indices.size >= 64`
(`switch_layers.py:217,277`); batch-1 top-22 (size 22) always runs **unsorted** through the
stock kernel. The Ultra fast path manually engages the sorted path below 64 for B=1
(`moe_fastpath.py:140-155`, gated `inds.size < 64 and _row_count(inds) == 1`).

---

## QuantizedLinear.from_linear

`nn.QuantizedLinear.from_linear(linear, group_size, bits, mode)`
(`$MLX/nn/layers/quantized.py:281-302`):
- **Quantizes** `linear.weight` via `mx.quantize` → sets `ql.weight, ql.scales, ql.biases`
  (`biases=None` for non-affine modes). The original high-precision weight is **not retained**.
- **Copies** `linear.bias` **only if present** (`if "bias" in linear_layer`).
- Copies **nothing else** (no name, no dtype coercion). Reads `linear.weight` directly, so
  feed it a layer whose `.weight` is already the tensor/dtype you want quantized.

`SwitchLinear.to_quantized(...)` is the MoE analogue (`switch_layers.py:144-162`), same pattern.

---

## uv-sync wheel-revert trap

`uv run` (and `uv sync`) **re-sync the venv against `uv.lock` on EVERY invocation** — a
pip-installed custom wheel (e.g. a patched `mlx-metal`) is **silently replaced with the
locked PyPI version** at the next `uv run omlx serve`. Symptom: the patch is present in
site-packages right after install, **gone at serve time; mtime snaps back**
(`tasks/lessons.md` ~L78-88).

**Detection recipe** (run BEFORE trusting a patched run):
```
# 1. grep a patch-marker string in the LOADED lib (not just site-packages on disk)
strings .venv/lib/python3.12/site-packages/mlx/lib/libmlx.dylib | grep -i "<your-marker>"
# 2. check mtime hasn't reverted
ls -l --time-style=full .venv/lib/python3.12/site-packages/mlx/lib/libmlx.dylib
# 3. confirm the version
.venv/bin/python -c "import mlx.core as mx; print(mx.__version__)"   # expect 0.31.2
```
**Fix:** `uv run --no-sync …`, or install into the synced env and never `uv run` afterward;
never edit `uv.lock` for experiments. This is the third incarnation of the house live-path
law (fp16-gate kernels, spy-vs-compiled-region, now uv-sync revert): **verify the artifact is
ENGAGED in the live process, not merely installed.** The rebuild recipe is in `env-setup.md`
(sibling).

---

## MLX 0.32.0 — side-venv only, NOT production (EXP-093/094/098)

Side venv `.venv-mlx032` (mlx 0.32.0 + same git-pinned mlx-lm @2ed2231 + transformers 5.12.1), A/B only.
**Install trap:** repo `pyproject.toml` `[tool.uv] override-dependencies` FORCES mlx==0.31.2 on any uv
install run from INSIDE the repo cwd, even targeting the side venv — install from outside the repo.
**transformers must stay 5.12.x** (5.13 breaks tokenizer_utils vs our git-pinned mlx-lm).
**qmv_wide** (new): batches M∈[2,8) in one dispatch; OLD qmv re-read the full weight per vector
(batch-2 paid 2× bytes). Puzzle in_proj (affine8) measured: M=4 **1.61×**, M=8 **1.65×**, M=1 1.18×
(codegen). Model-level batched decode: B=2 1.07× / B=4 **1.23×** / B=8 **1.32×** aggregate. Verify
L-curve compresses: L2/L1 1.38→1.21, L3/L1 1.69→1.40 (why spec-decode economics differ per venv).
**Single-stream production verdict: FLAT** (one-ahead 18.31→18.46ms; eager worse but pipeline hides
it) — 0.32 is the batch≥2 enabler, not a single-stream upgrade. Adoption cost: nanobind 2.13 → GLM
native ext REBUILD + fleet re-verify. Unexploited in 0.32: fused SDPA vector kernel for asymmetric
Q/V head dims (192,128) = MLA geometry (GLM bench candidate); MLX_SDPA_BLOCKS env; ST_F8_E8M0 dtype.

## UNVERIFIED / limits of this file
- `MAX_ACTIVE_TASKS=10` and the MB-per-buffer accounting live in **compiled** `transforms.cpp`
  / metal backend `.cpp` — not in the venv's shipped headers. The values here are from the
  campaign ledger's verification against MLX v0.31.2 source, not re-derivable from `.venv`
  alone. Re-confirm against upstream MLX source on any version bump.
