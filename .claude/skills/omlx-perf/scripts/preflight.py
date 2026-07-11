#!/usr/bin/env python3
# > Verified 2026-07-05 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.
#
# omlx-perf preflight — SERVERLESS pre-serve / pre-bench sanity checks.
# Stdlib only. NO network, NO tmux, NO HTTP to localhost:8000, NO model loads,
# NO reads of any apikey file. Reads local files + `sysctl`/`ps` only. Safe to
# run while the production server is live (it never touches the server).
#
# Each check prints PASS / WARN / FAIL <item>: <detail> and a one-line antidote
# on non-PASS. Exit code 1 if any FAIL, else 0. WARN never fails the run.
#
# Companion prose (WHAT/WHY/HOW per item): preflight.md
# Trap antidotes cited inline point into gotchas.md.
#
# Usage:  .venv/bin/python .claude/skills/omlx-perf/scripts/preflight.py

import json
import os
import shutil
import subprocess
import sys

HOME = os.path.expanduser("~")
MODELS_DIR = os.path.join(HOME, ".omlx", "models")
SETTINGS = os.path.join(HOME, ".omlx", "model_settings.json")

# Golden env — final campaign value (tasks/ultra_speed.md:5, overlap_levers.md:90).
GOLDEN_ENV = {"MLX_MAX_OPS_PER_BUFFER": "4000", "MLX_MAX_MB_PER_BUFFER": "4000"}
# Metal wired-limit sysctl target (raises cap 464->506GB; tasks/todo.md:227,243).
EXPECTED_WIRED_MB = 518144
EXPECTED_MLX = "0.31.2"
EXPECTED_NANOBIND = "2.12.0"  # GLM native-kernel ABI pin (commit 3c224ed)

# Kill/disable switches: presence means an optimization is turned OFF -> WARN.
# (grep OMLX_ omlx/patches/ omlx/scheduler.py — ground truth 2026-07-05)
DISABLE_SWITCHES = {
    "OMLX_GLM_DISABLE_DECODE_OPT", "OMLX_DISABLE_DSV3_DECODE_OPT",
    "OMLX_M3_DISABLE_FUSED_INDEX", "OMLX_M3_DISABLE_FUSED_TOPK",
    "OMLX_M3_DISABLE_FUSED_POSITIONS", "OMLX_M3_DISABLE_FUSED_SWIGLU_TS",
    "OMLX_M3_DISABLE_PACKED_PROJ", "OMLX_M3_DISABLE_NVFP4_TS",
    "OMLX_M3_DISABLE_SMALLL_VERIFY", "OMLX_NEMO_DISABLE_NVFP4_TS",
    "OMLX_ULTRA_DISABLE_TSPRE", "OMLX_ULTRA_DISABLE_SORTED_ROUTES",
    "OMLX_DISABLE_EARLY_INDEX_PUBLISH",
}
# Informational switches: presence is a deliberate tuning/telemetry choice.
INFO_SWITCHES = {
    "OMLX_ULTRA_DQ8_MAMBA", "OMLX_ULTRA_DQ8_MOEDENSE", "OMLX_ULTRA_DQ8_ATTN",
    "OMLX_ULTRA_DQ8_LMHEAD", "OMLX_TRANSIENT_CLAMP_K", "OMLX_M3_DEBUG_PATH",
    "OMLX_M3_SPARSE_MIN_K", "OMLX_M3_COMPACT_MAX_DENSITY", "OMLX_EAGLE3_MXFP8",
    "OMLX_M3_ENABLE_FLASH_SPARSE", "OMLX_M3_ENABLE_FLASH_SPARSE_V2",
    "OMLX_M3_ROUTE_TRACE", "OMLX_GLM_FLASH_DECODE_MIN_K", "OMLX_GLM_TOPK_DUMP",
    "OMLX_MTP_DRAFT_K", "OMLX_M3_COMPILE",
}

# Known serving footprints on THIS box (GB, cited). Used for the disk table.
# Sizes are DISK footprints of the served checkpoint.
KNOWN_MODELS_GB = [
    ("Nemotron-3-Ultra-oQNVFP4-dq8", 327, "tasks/todo.md:1096 (327GB disk / 305 resident)"),
    ("Kimi-K2.7-Code-*-VLM", 434, "tasks/todo.md:255 (434GiB)"),
    ("MiniMax-M3-oQNVFP4-fs*", 246, "tasks/todo.md:769 (-fs 246G)"),
    ("MiniMax-M3-oQ4", 228, "tasks/todo.md:299 (228GB)"),
]

_fails = 0
_warns = 0


def _p(status, item, detail="", antidote=""):
    global _fails, _warns
    line = f"{status:4} {item}"
    if detail:
        line += f": {detail}"
    print(line)
    if antidote:
        print(f"       -> {antidote}")
    if status == "FAIL":
        _fails += 1
    elif status == "WARN":
        _warns += 1


def _hdr(n, title):
    print(f"\n[{n}] {title}")


def _run(cmd, timeout=20):
    """Run a local command, return (rc, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:  # noqa: BLE001
        return 255, "", str(e)


def find_venv_python():
    """Locate the repo .venv python without importing anything heavy."""
    env = os.environ.get("OMLX_PERF_VENV")
    if env and os.path.exists(env):
        return env
    # Walk up from this script and from cwd looking for .venv/bin/python.
    seeds = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for seed in seeds:
        d = seed
        for _ in range(8):
            cand = os.path.join(d, ".venv", "bin", "python")
            if os.path.exists(cand):
                return cand
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd
    return None


# ---------------------------------------------------------------- check 1
def check_golden_env():
    _hdr(1, "Golden MLX buffer env (dispatch-batching)")
    ok = True
    for k, want in GOLDEN_ENV.items():
        got = os.environ.get(k)
        if got != want:
            ok = False
            _p("FAIL", k, f"is {got!r}, expected {want!r}")
    if ok:
        _p("PASS", "MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000")
    else:
        _p("", "antidote",
           antidote="export MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000 "
                    "in the serve/bench line (gotchas.md 'golden buffer env'; "
                    "tasks/ultra_speed.md:5).")


# ---------------------------------------------------------------- check 2
def check_omlx_switches():
    _hdr(2, "Per-model OMLX_* switch inventory")
    set_disable = sorted(k for k in DISABLE_SWITCHES if os.environ.get(k))
    set_info = sorted(k for k in INFO_SWITCHES if os.environ.get(k))
    # Any OMLX_* not in either known set -> surface it too.
    known = DISABLE_SWITCHES | INFO_SWITCHES
    set_unknown = sorted(k for k in os.environ if k.startswith("OMLX_") and k not in known)
    if not (set_disable or set_info or set_unknown):
        _p("PASS", "no OMLX_* switches set (stock optimization paths)")
        return
    for k in set_info:
        _p("INFO", f"{k}={os.environ[k]}", "deliberate tuning/telemetry")
    for k in set_unknown:
        _p("INFO", f"{k}={os.environ[k]}", "not in known inventory — verify intent")
    for k in set_disable:
        _p("WARN", f"{k}={os.environ[k]}", "a shipped optimization is DISABLED",
           antidote="unset it unless you are A/B-ing that lever (kill switches "
                    "default OFF in production). See gotchas.md 'kill switch left on'.")


# ---------------------------------------------------------------- check 3
def check_wired_limit():
    _hdr(3, "Metal wired-limit sysctl")
    rc, out, _ = _run(["sysctl", "-n", "iogpu.wired_limit_mb"])
    if rc != 0 or not out:
        _p("WARN", "iogpu.wired_limit_mb", "unreadable (kernel default ~ 0/unset)",
           antidote=f"sudo sysctl iogpu.wired_limit_mb={EXPECTED_WIRED_MB} "
                    "(raises Metal cap 464->506GB; tasks/todo.md:243).")
        return
    try:
        val = int(out)
    except ValueError:
        _p("WARN", "iogpu.wired_limit_mb", f"non-integer: {out!r}")
        return
    if val >= EXPECTED_WIRED_MB:
        _p("PASS", "iogpu.wired_limit_mb", f"{val} (>= {EXPECTED_WIRED_MB}, ~506GB cap)")
    else:
        _p("WARN", "iogpu.wired_limit_mb", f"{val} < {EXPECTED_WIRED_MB}",
           antidote=f"sudo sysctl iogpu.wired_limit_mb={EXPECTED_WIRED_MB} for long-ctx "
                    "KV headroom (default caps Metal ~464GB; tasks/todo.md:227).")


# ---------------------------------------------------------------- check 4
def check_mlx_provenance():
    _hdr(4, "MLX/nanobind version + wheel provenance")
    py = find_venv_python()
    if not py:
        _p("WARN", ".venv/bin/python", "not found; cannot verify installed MLX",
           antidote="run this via the repo's .venv/bin/python (never bare `uv run`).")
        return
    probe = (
        "import importlib.metadata as m\n"
        "import mlx.core as mx\n"
        "print(mx.__version__)\n"
        "try:\n"
        "    print(m.version('nanobind'))\n"
        "except Exception:\n"
        "    print('none')\n"
        "print(mx.__file__)\n"
    )
    rc, out, err = _run([py, "-c", probe], timeout=60)
    if rc != 0:
        _p("WARN", "import mlx.core", f"failed via {py}: {err.splitlines()[-1] if err else rc}")
        return
    lines = out.splitlines()
    mlx_ver = lines[0] if lines else "?"
    nb_ver = lines[1] if len(lines) > 1 else "?"
    mlx_path = lines[2] if len(lines) > 2 else "?"
    if mlx_ver == EXPECTED_MLX:
        _p("PASS", "mlx", f"{mlx_ver} (stock; {mlx_path})")
    else:
        _p("WARN", "mlx", f"{mlx_ver}, expected {EXPECTED_MLX}",
           antidote="`uv sync`/bare `uv run` silently reverts pinned wheels — use "
                    "`uv run --no-sync`; re-verify (gotchas.md 'uv sync reverts wheel').")
    if nb_ver == EXPECTED_NANOBIND:
        _p("PASS", "nanobind", f"{nb_ver} (GLM native-kernel ABI pin)")
    else:
        _p("WARN", "nanobind", f"{nb_ver}, expected {EXPECTED_NANOBIND}",
           antidote="GLM Metal kernels need nanobind 2.12.0 ABI — rebuild ext if changed "
                    "(commit 3c224ed).")


# ---------------------------------------------------------------- check 5
def check_disk_free():
    _hdr(5, "Disk free on ~/.omlx/models volume")
    target = MODELS_DIR if os.path.exists(MODELS_DIR) else HOME
    try:
        total, used, free = shutil.disk_usage(target)
    except Exception as e:  # noqa: BLE001
        _p("WARN", "disk_usage", f"unreadable: {e}")
        return
    free_gb = free // (1024 ** 3)
    _p("INFO", "free", f"{free_gb} GB free on the internal volume")
    biggest = max(KNOWN_MODELS_GB, key=lambda t: t[1])
    for name, size, cite in KNOWN_MODELS_GB:
        fits = "fits" if free_gb >= size else "WILL NOT FIT"
        print(f"       {name:34} {size:>4} GB  [{fits}]  ({cite})")
    if free_gb < biggest[1]:
        _p("WARN", "headroom", f"< largest known model ({biggest[0]} {biggest[1]}GB)",
           antidote="a fresh download/convert needs headroom; HF `usedStorage` is repo-wide "
                    "(all revisions) and a mid-run `du` under-reports — size from safetensors "
                    "shard bytes / the converter Summary line (gotchas.md 'HF usedStorage', "
                    "'du mid-convert'; conversion.md du pacing).")


# ---------------------------------------------------------------- check 6
def check_no_usb_serving():
    _hdr(6, "No USB/NFS-volume serving (omlx#2098)")
    if not os.path.isdir(MODELS_DIR):
        _p("WARN", MODELS_DIR, "not found")
        return
    external = []
    seen = set()
    # depth 1 + depth 2 (namespace dirs like avlp12/, unigilby/ hold real models)
    def scan(d, depth):
        try:
            entries = list(os.scandir(d))
        except OSError:
            return
        for e in entries:
            try:
                real = os.path.realpath(e.path)
            except OSError:
                continue
            if real.startswith("/Volumes/"):
                if real not in seen:
                    seen.add(real)
                    external.append((e.path.replace(HOME, "~"), real))
            elif depth < 2 and (e.is_dir() and not e.is_symlink()):
                scan(e.path, depth + 1)
    scan(MODELS_DIR, 1)
    if not external:
        _p("PASS", "all model paths resolve to internal SSD")
        return
    for link, real in external:
        _p("FAIL", link, f"-> {real}")
    _p("", "antidote",
       antidote="serving from USB/NFS -> Metal GPU command-buffer TIMEOUT (omlx#2098). "
                "COPY the model onto the internal SSD before SERVING it (symlinks are fine "
                "as cold storage for models you do NOT load). See gotchas.md 'USB serving'.")


# ---------------------------------------------------------------- check 7
def check_model_settings():
    _hdr(7, "model_settings force_sampling / temperature audit")
    if not os.path.exists(SETTINGS):
        _p("WARN", SETTINGS, "not found")
        return
    try:
        with open(SETTINGS) as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        _p("WARN", SETTINGS, f"unparseable: {e}")
        return
    models = data.get("models", {})
    flagged = []
    for name, cfg in models.items():
        if not isinstance(cfg, dict):
            continue
        fs = cfg.get("force_sampling")
        temp = cfg.get("temperature")
        if fs is True:
            flagged.append((name, temp))
    if not flagged:
        _p("PASS", "no model forces sampling (temp-0 gates honored)")
        return
    for name, temp in flagged:
        _p("WARN", name, f"force_sampling=true (temperature={temp})",
           antidote="force_sampling OVERRIDES a caller's temp=0 -> any temp-0 quality gate "
                    "or spec-decode/EAGLE temp0 engagement guard is silently defeated. Lift "
                    "it for those benches (tasks/eagle_temp1.md:9; gotchas.md 'force_sampling').")


# ---------------------------------------------------------------- check 8
def check_quiet_box():
    _hdr(8, "Quiet box (no competing GPU-heavy processes)")
    rc, out, _ = _run(["ps", "-Ao", "pid=,rss=,comm="])
    if rc != 0 or not out:
        _p("WARN", "ps", "unreadable; cannot assess box quietness")
        return
    heavy = []
    for ln in out.splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss, comm = parts
        try:
            rss_gb = int(rss) / (1024 ** 2)  # rss is KB
        except ValueError:
            continue
        low = comm.lower()
        if rss_gb >= 4.0 and any(t in low for t in ("python", "uv", "omlx", "mlx")):
            heavy.append((rss_gb, pid, comm))
    heavy.sort(reverse=True)
    if not heavy:
        _p("PASS", "no >4GB python/mlx processes besides the OS")
        return
    for rss_gb, pid, comm in heavy[:6]:
        print(f"       {rss_gb:6.1f} GB  pid {pid:>7}  {comm}")
    if len(heavy) > 1:
        _p("WARN", "multiple heavy processes",
           f"{len(heavy)} GPU-capable processes >4GB (server + ?)",
           antidote="a second bench/convert/model-load contends for 819GB/s bandwidth and "
                    "skews tok/s. Run benches solo (gotchas.md 'noisy box').")
    else:
        _p("INFO", "one heavy process", "likely the production server — expected")


def main():
    print("omlx-perf preflight — serverless checks (no server contact, no model load)")
    check_golden_env()
    check_omlx_switches()
    check_wired_limit()
    check_mlx_provenance()
    check_disk_free()
    check_no_usb_serving()
    check_model_settings()
    check_quiet_box()
    print(f"\nSUMMARY: {_fails} FAIL, {_warns} WARN")
    if _fails:
        print("One or more FAIL checks — read the antidotes above before serving/benching.")
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
