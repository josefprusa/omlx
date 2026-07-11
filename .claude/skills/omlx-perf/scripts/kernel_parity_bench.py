#!/usr/bin/env python3
# > Verified 2026-07-05 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.
"""kernel_parity_bench.py — standalone quantized-matmul (gather_qmm) parity + throughput bench.

Compares quantization containers (nvfp4-gs16, affine4-gs64, mxfp4-gs32, affine8-gs64,
mxfp8-gs32, affine3-gs64) on the SAME synthetic MoE expert weights at configurable shapes.
Reports per-mode microseconds/layer, achieved GB/s, % of a bandwidth-ideal dense read, and a
"within +-X%" verdict across modes. Reproduces the campaign's "M1" kernel-parity experiment
(sanitized from jobs/.../tmp/m1_kernel_bench.py; see mlx.md section "Kernel parity" and
.claude/skills/omlx-perf/experiments/) ENTIRELY OFFLINE — NO server, NO model weights loaded,
NO HTTP. Peak resident ~10GB at Ultra defaults (synthetic weights, freed between shapes).

WHY THE mx.set_wired_limit CALL BELOW IS MANDATORY (do NOT delete it):
  A standalone bench that allocates a multi-GB working set WITHOUT raising the Metal wired
  limit lets the GPU page memory in/out -> a page-fault storm that inflated one campaign
  standalone bench by ~65x. It is the same class of harness artifact as the K1 first run's
  "13.35x" (that one from a MISSING golden buffer env). Both make quantized-matmul timings
  meaningless. Sources: tasks/todo.md line ~1021 ("missing mx.set_wired_limit => page-fault
  storm; Gate-1 hit the same"); tasks/ultra_speed.md line ~21 ("first run's 13.35x was
  confirmed harness artifact (missing golden env)").

COMPANION REQUIREMENT — run under the golden command-buffer env so timings track production:
      MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000   (Ultra golden; see mlx.md)
  These are read ONCE at process start, so set them in the ENV, not after import.

USAGE (from repo root; NEVER `uv run` — it resyncs the venv, see mlx.md "uv-sync trap"):
  MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000 \
    .venv/bin/python .claude/skills/omlx-perf/scripts/kernel_parity_bench.py

  # custom shapes / modes / batch rows:
  ... kernel_parity_bench.py --experts 512 --top 22 --fc1 5120x2048 --fc2 2048x5120 \
       --modes nvfp4,affine4,mxfp4 --M 1 --iters 200 --warm 30

  # to elicit nvfp4-gs16's scale-read tax (the Nemotron-Super "0.58x" regime), shrink the
  # latent + expert count — small fat-free shapes make gs16's 4x scale reads bite
  # (oqnvfp4_nemotron.md); at Ultra's fat 2048-latent shapes all modes land within +-4%.
  ... --experts 128 --top 8 --fc1 1536x2048 --fc2 2048x1536

EXPECTED OUTPUT (Ultra defaults, M=1; absolute us are this box +- spread, ratios are stable):
  [setup] wired limit -> 506.0 GB (max recommended 506.0 GB)
  [env]   MLX_MAX_OPS_PER_BUFFER=4000  MLX_MAX_MB_PER_BUFFER=4000  (golden: OK)
  --- M=1  top-22 of 512 experts  (peak 819.0 GB/s) ---
  fc1    nvfp4=  408.2us  affine4=  404.7us  mxfp4=  395.1us    ideal= 158.5us
  fc2    nvfp4=  416.5us  affine4=  423.6us  mxfp4=  401.3us    ideal= 158.5us
  layer  nvfp4=  824.7us  affine4=  828.3us  mxfp4=  796.4us    ideal= 316.9us (~38% peak)
  spread across modes: 4.0%   VERDICT: within +-5% (no mode gap at these shapes)
  finite: nvfp4=OK affine4=OK mxfp4=OK
  done
"""
from __future__ import annotations

import argparse
import os
import time

import mlx.core as mx

# name -> mx.quantize / mx.gather_qmm kwargs. Extend freely; every entry is a valid
# (mode, group_size, bits) triple per MLX 0.31.2's quantize-modes table (see mlx.md).
MODES: dict[str, dict] = {
    "nvfp4": dict(bits=4, group_size=16, mode="nvfp4"),
    "affine4": dict(bits=4, group_size=64, mode="affine"),
    "affine8": dict(bits=8, group_size=64, mode="affine"),
    "affine3": dict(bits=3, group_size=64, mode="affine"),
    "mxfp4": dict(bits=4, group_size=32, mode="mxfp4"),
    "mxfp8": dict(bits=8, group_size=32, mode="mxfp8"),
}


def _parse_shape(s: str) -> tuple[int, int]:
    """'OUTxIN' -> (out, in). gather_qmm w is [experts, out, in], x @ w.T."""
    out, inn = s.lower().split("x")
    return int(out), int(inn)


def _set_wired_limit(wired_gb: float | None) -> None:
    # macOS 15+ only. Clamp strictly below max_recommended_working_set_size (MLX warns
    # otherwise). Best-effort: a bench must not crash on a smaller / older box.
    try:
        info = mx.device_info()  # mx.metal.device_info is deprecated in 0.31.2
        max_ws = int(info.get("max_recommended_working_set_size", 0))
    except Exception:
        max_ws = 0
    want = int(wired_gb * 1024**3) if wired_gb else max_ws
    if max_ws:
        want = min(want, max_ws)
    if want <= 0:
        print("[setup] WARNING: could not set wired limit (no device info); timings may be noisy")
        return
    try:
        mx.set_wired_limit(want)
        print(f"[setup] wired limit -> {want/1024**3:.1f} GB (max recommended {max_ws/1024**3:.1f} GB)")
    except Exception as ex:  # pre-macOS-15, or value rejected
        print(f"[setup] WARNING: set_wired_limit failed ({type(ex).__name__}: {ex}); timings may be noisy")


def _bytes_ideal(out: int, inn: int, top: int, q: dict) -> float:
    """Lower-bound bytes moved for a top-`top` gather over one [experts,out,inn] matrix:
    only the `top` gathered expert tiles' packed weights + their per-group scales are read."""
    elems = out * inn * top
    wbytes = elems * q["bits"] / 8.0
    # one scale (+ maybe bias) per group_size elements; ~2-4 bytes each. Small vs weights.
    n_groups = elems / q["group_size"]
    sbytes = n_groups * (4.0 if q["mode"] == "affine" else 1.0)
    return wbytes + sbytes


def _bench(fn, iters: int, warm: int) -> float:
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds/call


def _make_fn(x, wq, scales, biases, idx, q):
    kw = dict(transpose=True, group_size=q["group_size"], bits=q["bits"], mode=q["mode"])
    if biases is not None:
        return lambda: mx.gather_qmm(x, wq, scales, biases, rhs_indices=idx, **kw)
    return lambda: mx.gather_qmm(x, wq, scales, rhs_indices=idx, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experts", type=int, default=512, help="number of experts E (default 512, Ultra)")
    ap.add_argument("--top", type=int, default=22, help="top-k routed experts per token (default 22)")
    ap.add_argument("--fc1", type=_parse_shape, default=(5120, 2048), help="fc1 OUTxIN (default 5120x2048)")
    ap.add_argument("--fc2", type=_parse_shape, default=(2048, 5120), help="fc2 OUTxIN (default 2048x5120)")
    ap.add_argument("--modes", default="nvfp4,affine4,mxfp4", help="comma list from: " + ",".join(MODES))
    ap.add_argument("--M", type=int, nargs="+", default=[1], help="batch rows to sweep (default: 1)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warm", type=int, default=30)
    ap.add_argument("--peak-gbps", type=float, default=819.0, help="memory peak GB/s for %%-of-peak (default 819)")
    ap.add_argument("--wired-gb", type=float, default=None, help="wired limit GB (default: max recommended)")
    ap.add_argument("--tol", type=float, default=5.0, help="spread%% threshold for the no-gap VERDICT (default 5)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    _set_wired_limit(args.wired_gb)
    # MLX reads these ONCE C-side at process start; report whether the golden env is present.
    o = os.environ.get("MLX_MAX_OPS_PER_BUFFER", "<unset>")
    mb = os.environ.get("MLX_MAX_MB_PER_BUFFER", "<unset>")
    golden = "OK" if (o == "4000" and mb == "4000") else "NOT golden -> timings may not track production"
    print(f"[env]   MLX_MAX_OPS_PER_BUFFER={o}  MLX_MAX_MB_PER_BUFFER={mb}  ({golden})")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m!r}; choose from {list(MODES)}")
    shapes = {"fc1": args.fc1, "fc2": args.fc2}
    mx.random.seed(args.seed)

    for M in args.M:
        # M=1 (the ONLY measured/production regime) uses rhs_indices [1,TOP]. For M>1 use the
        # ledger-corrected [M,1,TOP] shape (ultra_speed.md ~L259): the flat [M,TOP] form errors
        # with a broadcast_shapes ValueError -- that is the documented batch trap, not a bug here.
        # Batch>=2 is NOT a validated campaign path; the graceful ERR cells report any mismatch.
        idx_shape = (M, args.top) if M == 1 else (M, 1, args.top)
        idx = mx.random.randint(0, args.experts, idx_shape).astype(mx.uint32)
        print(f"--- M={M}  top-{args.top} of {args.experts} experts  (peak {args.peak_gbps:.1f} GB/s) ---")
        layer_us = {m: 0.0 for m in modes}
        finite = {m: True for m in modes}
        for name, (out, inn) in shapes.items():
            w = (mx.random.normal((args.experts, out, inn)) * 0.02).astype(mx.bfloat16)
            x = mx.random.normal((M, 1, inn)).astype(mx.bfloat16)
            cells = []
            for m in modes:
                q = MODES[m]
                try:
                    parts = mx.quantize(w, **q)
                    wq, scales = parts[0], parts[1]
                    biases = parts[2] if len(parts) > 2 else None
                    fn = _make_fn(x, wq, scales, biases, idx, q)
                    out_arr = fn()
                    mx.eval(out_arr)  # shape/validity check
                    finite[m] = finite[m] and bool(mx.isfinite(out_arr).all().item())
                    us = _bench(fn, args.iters, args.warm)
                    layer_us[m] += us
                    cells.append(f"{m}={us:8.1f}us")
                except Exception as ex:  # unsupported (mode,gs,bits) or shape lands here
                    cells.append(f"{m}=ERR({type(ex).__name__}:{str(ex)[:32]})")
                    finite[m] = False
            ideal_us = _bytes_ideal(out, inn, args.top, MODES[modes[0]]) / (args.peak_gbps * 1e9) * 1e6
            print(f"{name:6s} " + "  ".join(cells) + f"    ideal={ideal_us:8.1f}us")
            del w
            mx.clear_cache()

        # layer verdict + spread across modes (only modes that timed cleanly)
        timed = {m: v for m, v in layer_us.items() if v > 0 and finite.get(m)}
        if len(timed) >= 2:
            lo, hi = min(timed.values()), max(timed.values())
            spread = (hi - lo) / lo * 100.0
            ideal_layer = sum(
                _bytes_ideal(o, i, args.top, MODES[modes[0]]) for (o, i) in shapes.values()
            ) / (args.peak_gbps * 1e9) * 1e6
            pct_peak = ideal_layer / hi * 100.0
            cells = "  ".join(f"{m}={v:8.1f}us" for m, v in timed.items())
            verdict = f"within +-{args.tol:.0f}%" if spread <= args.tol else f"MODE GAP {spread:.1f}% > {args.tol:.0f}%"
            print(f"layer  {cells}    ideal={ideal_layer:8.1f}us (~{pct_peak:.0f}% peak)")
            print(f"spread across modes: {spread:.1f}%   VERDICT: {verdict}")
        print("finite: " + " ".join(f"{m}={'OK' if finite.get(m) else 'NO'}" for m in modes))
    print("done")


if __name__ == "__main__":
    main()
