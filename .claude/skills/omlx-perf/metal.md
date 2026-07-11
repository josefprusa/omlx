> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Writing & Judging Custom Metal Kernels (`mx.fast.metal_kernel`)

How to write a JIT Metal kernel for omlx, and — harder — how to decide one is
worth writing. Distilled from the three SHIPPED precedents. `mx.fast.metal_kernel`
is **pure Python JIT** (no native build, no ABI pin — that only applies to the
`csrc/*.metal` native primitives, see Nanobind below). Ground-truth files:

| file | fuses | engaged when | kill switch |
|---|---|---|---|
| `omlx/patches/glm_moe_dsa/decode_kernels.py` | GLM-5.2 s==1 indexer scores; flash-decode sparse-MLA (2-kernel split-K); `mh_qmm_m1` (per-head qmm via `gather_qmm`) | decode L==1; flash only `K>=98304` | `OMLX_GLM_DISABLE_DECODE_OPT=1` |
| `omlx/patches/nemotron_h_puzzle/fused_decode.py` (Puzzle pool A) | mamba glue: conv-step (concat+depthwise-conv4+silu+split+state-shift) + gated-norm (swiglu+grouped-RMS+weight) | decode B=L=1, fingerprint-gated | `OMLX_PUZZLE_DISABLE_FUSED_MAMBA=1` |
| `.../minimax_m3_vl/fused_index.py` | M3 idx_q·idx_k + 128-block max-pool one kernel; fused NaN-clean+top-16+sort; SwiGLU-OAI+ts; flash sparse SDPA | decode + small-L (2≤L≤8) verify | `OMLX_M3_DISABLE_FUSED_INDEX=1` |
| `.../minimax_m3_vl/fused_flash_v2.py` | split-K sparse flash decode (pass1 partials → pass2 merge) | opt-in only | `OMLX_M3_ENABLE_FLASH_SPARSE_V2=1` (default OFF) |

Sources: file docstrings (`decode_kernels.py:1-18`, `fused_index.py:1-8`,
`fused_flash_v2.py:1-9`).

## Anatomy — the recurring skeleton

Every kernel follows the same 4 parts: a `_SRC` body string, a lazy cached
builder, a gated Python wrapper returning `Optional`, and a reshape back.

```python
import mlx.core as mx
from typing import Optional

# 1) _SRC is the kernel BODY only — NO signature. MLX synthesizes the function
#    prototype from input_names/output_names + template. Each named input is in
#    scope as `const device T*`, each output as `device T*`. Metal builtins
#    (thread_position_in_threadgroup, threadgroup_position_in_grid,
#    threads/threadgroups_per_grid) are available unprefixed.
_SRC = """
    uint tid  = thread_position_in_threadgroup.x;     // 0..255
    uint blk  = threadgroup_position_in_grid.y;       // grid.y = this TG's index
    uint lane = tid & 31u;                            // simd lane 0..31
    uint sg   = tid >> 5u;                            // simdgroup 0..7
    // ... math in float accumulators ...
    if (lane == 0) out[blk] = (T)acc;                 // T comes from template
"""

_kernel = None                                        # build ONCE, cache global
def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="my_kernel",                         # MUST be unique per source
            input_names=["x", "params"],              # bind to _SRC symbols
            output_names=["out"],
            source=_SRC,
        )
    return _kernel

def my_op(x: mx.array, n: int) -> Optional[mx.array]:
    if x.dtype not in (mx.float16, mx.bfloat16) or x.ndim != 4:
        return None                                   # GATE → caller falls back
    (o,) = _get_kernel()(
        inputs=[x, mx.array([n], dtype=mx.uint32)],   # scalars = tiny mx.arrays
        template=[("T", x.dtype)],                    # one cached kernel PER dtype
        grid=(256, n, 1),                             # TOTAL threads, NOT TGs
        threadgroup=(256, 1, 1),                      # 256 = 8 simdgroups × 32
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )
    return o.reshape(1, 1, 1, n)
```

Rules mined from all three files:
- **Source = body only.** MLX builds the signature; symbols are the
  `input_names`/`output_names` (`decode_kernels.py:110-116`, `:143-150`).
- **Build once, cache in a module global** via a lazy `_get_*()` — never build
  per call (`decode_kernels.py:104-116`; `fused_index.py:703-712`).
- **`grid` is TOTAL threads** (x,y,z), not threadgroup count (differs from CUDA).
  `grid=(256, nblocks, 1)` + `threadgroup=(256,1,1)` ⇒ `threadgroups_per_grid.y
  == nblocks`, indexed by `threadgroup_position_in_grid.y`
  (`decode_kernels.py:152-153`; `fused_index.py:68,71`).
- **Scalars go in as tiny arrays** (`mx.array([...], dtype=mx.uint32/float32)`) —
  e.g. `kdim`, `scale`, `params` (`decode_kernels.py:148-150`; `fused_index.py:905`).
- **`template=[("T", dtype)]`** makes MLX generate & cache a distinct kernel per
  dtype from the signature. For dtype-generic device loads use
  `const device auto*` (works for fp16 AND bf16) (`fused_index.py:44-46,204`).
- Wrapper **returns `Optional`; `None` = unsupported → caller uses the mx chain**
  (`decode_kernels.py:126-140,332-346`; `fused_index.py:663-679`).

## Grid / threadgroup sizing (the patterns actually used)

- **256 threads/TG = 8 simdgroups × 32 lanes.** `lane = tid & 31`,
  `sg = tid >> 5` (`decode_kernels.py:65-67,186-187`; `fused_index.py:373-375`).
- **One TG per head** (attention): `grid.y = num_heads`; lane covers `D/32` dims
  of the head dot (`PER_LANE = D/32` = 4 for D=128, 16 for D=512), reduced by
  `simd_sum` (`decode_kernels.py:83-94,179`; `fused_index.py:411-417`).
- **Split-K** for long contexts: add `grid.z = n_splits`; each TG handles a
  key-chunk, emits partial `(m, l, acc)`; a second merge kernel combines them
  (`decode_kernels.py:361-372` grid `(256,8,16)`; `fused_flash_v2.py:242,254`
  pass1 `(256,H,8)` → pass2 `(D,H,1)`).
- **Online softmax in fp32** accumulators; stage K/V tiles in `threadgroup`
  memory with barriers between load and compute
  (`decode_kernels.py:190-262`; `fused_index.py:381-440`).
- **ILP**: process 4 slots per simdgroup iteration so the dots + `simd_sum`s
  overlap; keep the softmax update sequential per slot (`fused_index.py:399-435`).

## dtype gating & the fp16-vs-bf16 incident (READ THIS)

A gate that is wrong-by-dtype silently disables the kernel in production.

**Incident (2026-07-03):** `fused_index.py` originally required **fp16**; live M3
runs **bf16** (`torch_dtype`). Result: **100% silent fallback**
(`fused_none=57/57` every step) executing the full-cache fp32 `astype`+matmul
glue — the entire anomalous decode slope. Standalone verification with
`set_dtype(float16)` reported "bit-exact, engages" — true **only offline**
(`lessons.md` §"fused kernel dead-on-arrival in live serving (dtype gate)").

**Fix:** dtype-generic `const device auto*` device loads support fp16 AND bf16;
MLX caches a kernel per dtype (`fused_index.py:44-46`). Every current wrapper
gates `dtype in (mx.float16, mx.bfloat16)`, not `== float16`
(`decode_kernels.py:137`; `fused_index.py:597,676`).

**LAW:** any gated fast-path (shape/dtype/flag) MUST ship a live engagement
counter, and standalone repros MUST copy the LIVE dtype (read `config.json`
`torch_dtype`, never assume fp16) (`lessons.md`). → cross-ref `laws.md`
(live-path law: gated fast paths need live engagement counters).

## Fallback paths & engagement counters

- **`_log_once(tag)`**: a module `set` logs the first ENGAGE and first BAIL per
  tag once, zero cost after — the antidote to silent-death
  (`decode_kernels.py:37-47`; wrapper logs `indexer-scores ENGAGED dtype=...` /
  `BAIL q=.../k=...` at `:139,142`).
- **M3 census**: `OMLX_M3_DEBUG_PATH=N` prints per-step `fused=X/Y` /
  `fused_none=X/Y` counters — grep them LIVE before benching (`lessons.md`).
- **Kill switches** are env-gated helpers: `decode_opt_enabled()`
  (`decode_kernels.py:28-34`), `enabled()`/`topk_enabled()`/`flash_enabled()`
  (`fused_index.py:642-649,819-828`). Opt-in experiments default OFF
  (`flash_v2_enabled()` `fused_flash_v2.py:190-193`).

## Per-instance patching

Kernels are module-level functions returning `Optional`; the model patch calls
them and falls back on `None`. **Patch per-INSTANCE** (`mixer.__class__ =
_SubClass` swap on the target model's layers) — NEVER rebind a class method at
module scope: it silently alters other models sharing the process
(`lessons.md` §2026-07-05 "MODEL PATCH THAT COEXISTS"; `ultra_speed.md:381-382`).
→ cross-ref `omlx.md` (per-instance swap convention & patch registration).

## Iteration history — bench PER STEP or you ship the wrong kernel

The GLM/M3 sparse-MLA decode kernel took **5 architectures**; measured µs vs the
MLX gather+SDPA chain (`todo.md:43-44,88-96`):

| ver | approach | µs | verdict |
|---|---|---|---|
| v1 | initial | 114 | — |
| v2 | — | 108 | — |
| v3 | — | 78 | **SHIPPED for K≥98304** (`decode_kernels.py:169`) |
| v4 | read-once all-64-heads/1024-thread TG, 2-phase TG-mem | **430** | **REGRESSION** (barrier+TG-mem coord cost > the SLC re-read it removed) |
| v5 | faithful MLX `sdpa_vector` clone + in-kernel topk gather + fused pe | 92-94 | correct (2e-4) but 0.92× at 131k → CLOSED |

**v4 regressed 5.5× vs v3** — without a bench per step you cannot tell v3 from
v4. **Root cause proven:** materialize-once-then-sequential-SDPA beats ANY
in-kernel random-gather on M3 Ultra; MLX's compact 2.4MB gather buffer stays
SLC-hot (re-read ~3TB/s). Crossover ~64k; the kernel only wins ≥98k, hence the
`FLASH_DECODE_MIN_K` gate (`todo.md:29-33,44-46,88-96`;
`decode_kernels.py:164-171`). A losing prototype is DATA; try the free/native
alternative first — `gather_qmm` swap beat a hand-written multi-head qmv
(`lessons.md`; `mh_qmm_m1` `decode_kernels.py:390-413`).

## Dispatch floors — will the kernel even clear its own launch cost?

Measured in-session 2026-07-05 via `ultra_probe_dispatch.py` (probe spec
`ultra_speed.md:281-284`; registry EXP-056 in `experiments/ultra-day.md`). **PROVISIONAL — the
probe's raw stdout was never archived to a readable ledger; re-measure on a quiet box before any
load-bearing use:**

| op | host floor |
|---|---|
| binary op | **~3.17µs** |
| `mx.fast.metal_kernel` call | **~6.16µs** |
| skinny `mx.gather_qmm` | **~7.23µs** |

Earlier lumped M3 figure: **~8.6µs/dispatch** incl. lazy-graph + encode
(`todo.md:541,983`). **Economics:** fusing N ops saves ~(N−1)×floor; a kernel
must clear its OWN ~6µs floor to matter at decode. A kernel replacing 7 glue ops
nets ~7×(3-6µs) − 6µs. This is why M3's fused index/topk (replaces ~8
dispatches/layer) pays but a single-op replacement rarely does
(`fused_index.py:82-85` docstring).

## Command-buffer mechanics — a fragmented chain destroys benches

- **Golden env: `MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000`**
  (`ultra_speed.md:4-5`; `SHELF.md`). Arch default commits every 50 ops / 50MB →
  a fragmented command chain (`todo.md:551`). Mechanics live in `mlx.md`
  (cross-ref: op-chain batching & encoder commit points); the Metal-side view is
  that a fragmented chain adds an encode/commit between fused ops, erasing the
  saving.
- **Proof — the K1 bench:** first run WITHOUT golden env measured **13.35× ideal**
  (confirmed harness artifact); WITH golden env it dropped to **1.87× ideal**
  (`ultra_speed.md:20-24`). Always benchmark a kernel under the golden env or the
  numbers are fiction. `MLX_MAX_OPS_PER_BUFFER 500→2000` is flat on M3; 4000/4000
  stands, M3 included (M3 FINAL = golden 4000/4000, `models/minimax-m3.md` §4;
  `ultra_speed.md:515-516`; `todo.md:961`).

## Box physics — the envelopes any kernel lives inside

- **819GB/s nominal.** Whole-model token reads hit **~74% of nominal** (Ultra:
  79GB/token, ceiling 10.36-10.48 tok/s; live 7.68) (`ultra_speed.md:108-109`; the 7.68 datum at `ultra_speed.md:4`).
- **Fat matmuls run ~96% of ceiling** — MLX qmv = 789GB/s at 8192×6144
  (`todo.md:883-884`; `ultra_speed.md:199`). **Do NOT hand-rewrite dense matmul/qmv.**
- **In-stream op chains ~58-60% of nominal**: K1's dense-equal-bytes control ran
  **1.71× ideal ⇒ ~58% utilization** on `mx.depends`-chained ops
  (`ultra_speed.md:20-24`). Isolated timings LOOK worse (~38% of peak) but that is
  harness inflation — only ~50-60% of naive deltas survive pipelining, and this
  trap produced the retracted "MoE at 60% bandwidth" claim
  (`ultra_speed.md:252-256`; `lessons.md`). **Bench in-stream, not standalone.**
- **Wired memory:** default Apple Metal cap **464GB**; `sudo sysctl
  iogpu.wired_limit_mb=518144` raises it to **506GB** (`todo.md:227,244`). The
  server raises the Metal wired limit 0→506GB at startup and runs a
  process-memory enforcer at a **balanced-tier ceiling ~489–496GB (boot-dependent)**
  (493.4GB `serve_211919.log:44`; 496.3GB `~/.omlx/logs/server.log.2026-07-05:5748` —
  `min(static 506, dynamic, metal_cap)`, the dynamic term shifts with memory pressure);
  ~6GB physical headroom over
  512GB is the macOS floor. A STANDALONE kernel repro must call
  `mx.set_wired_limit(506*1024**3)` first or hit a GPU page-fault storm (the ~65×
  standalone-vs-server slowdown, `GLM52_MTP_FORAY.md:388-394`). Preflight checks →
  cross-ref `preflight.md`.

## The diffuse-overhead wall & the mega-kernel case

After every targeted lever, a **~1.7× diffuse in-stream overhead** remains — the
K1 dense-equal-bytes control's 1.71×-of-ideal IS this floor
(`ultra_speed.md:20-28`). Same animal across models:
- GLM: **~32ms/token** diffuse residual in the L=2 verify (attention/indexer/
  norms/cache glue × 78 layers of eager MLX dispatch), NOT any quantized matmul;
  `mx.compile` of the full forward = only 1.06× → dead as a lever
  (`GLM52_MTP_FORAY.md:342-360`).
- Ultra: expert-SPECIFIC excess is only **2.41ms/token** (unsorted 1.87× − dense
  control 1.71× = 50.1µs/layer × 48) — a fused expert-MLP kernel can capture
  ~2-4ms, nowhere near a DQ8 bytes-cut (`ultra_speed.md:24-26,314-316`).

The only structural attack on the diffuse floor is a **fused decoder-layer
"mega-kernel"** (one kernel/layer, eliminating inter-kernel dispatch/dataflow) —
a months-class rewrite. Its economics → cross-ref `future-campaigns.md`.

## Nanobind / ABI note (JIT path is exempt)

`mx.fast.metal_kernel` needs **no native build**. Only the `csrc/*.metal` native
primitives require MLX built against **nanobind 2.12.0 (ABI v19)** — mlx-metal
0.31.2 ships v19; an ext built against 2.13.0 (v20) crashes or silently falls
back (`memory omlx-glm52-native-kernels.md:14-21`). Rebuild recipe →
cross-ref `env-setup.md`.

## Decision tree — should I write a Metal kernel?

1. **Is the target a single dense matmul/qmv?** → NO. MLX runs ~96% of ceiling
   (`todo.md:884`). Try `gather_qmm`/`quantized_matmul` shape tricks first
   (`lessons.md`; `mh_qmm_m1`).
2. **Does it fuse ≥~3-4 glue ops?** If it replaces N dispatches, expected saving
   ~(N−1)×3-6µs − one 6µs launch. <3 ops rarely clears the floor.
3. **Bench IN-STREAM under golden env** (`mx.depends`-chained, 4000/4000). If the
   isolated win vanishes in-stream, it was harness inflation (~50-60% survives).
4. **Is the byte traffic already minimal?** If bandwidth-bound and bytes can't
   drop, a kernel can't help — cut bytes (quant) instead (`ultra_speed.md` DQ8).
5. **Iterate with a bench PER architecture** — v4 regressed 5.5× vs v3 silently.
6. **Ship with:** an `Optional` gate (`dtype in (fp16,bf16)`), a `_log_once`
   engagement counter, a kill-switch env var, and a per-instance patch.

## Pipelined-decode stability is its own gate (Law 18, EXP-095)

A kernel can PASS parity and run clean under EAGER decode and still be unsafe live: Puzzle pool C
(fused expert chain, `nemotron_h_puzzle/pool_c.py`) parity-PASSED everywhere, then intermittently
faulted the Metal command buffer ONLY under async one-ahead decode
(kIOGPUCommandBufferCallbackErrorInnocentVictim, exit 134). Root cause undiagnosed; banked unwired.
**Rule:** every kernel destined for the production decode path gets a sustained PIPELINED soak
before being called safe — eager-only testing is insufficient.
**Also EXP-095:** pool B replaced `MoEGate` with a single-threadgroup 512×4096 gemv kernel — engaged
40/40, ran **7.3× slower**. Two architectures (GLM, NemotronH) now agree: never hand-roll a dense
gemv/router against MLX's own.
