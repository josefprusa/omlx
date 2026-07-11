#!/usr/bin/env python3
"""Verify that every legacy path maps to exactly one feature-ledger group."""

from __future__ import annotations

import subprocess
import sys

FORK_POINT = "14338c3e37260a43b027b7317a0f85a60cf85c35"
LEGACY_HEAD = "1dded076eba287386e92e557ad781932c97630dd"
ARCHIVE = "3f2edd6051ecdb58395d3c63118dd56e51c66aee"


def git(*args: str) -> list[str]:
    output = subprocess.check_output(["git", *args], text=True)
    return [line for line in output.splitlines() if line]


def groups(path: str) -> list[str]:
    matches: list[str] = []

    if path.startswith((".claude/", "tasks/")) or path == "GLM52_MTP_FORAY.md":
        matches.append("R-001")
    elif path.startswith("omlx/patches/nemotron_h_puzzle/") or path.startswith(
        "omlx/tools/oq_puzzle_"
    ) or path.startswith(("tests/test_nemotron_h_puzzle", "tests/test_oq_puzzle")):
        matches.append("E-004")
    elif path.startswith("omlx/patches/hy_v3/") or path == "omlx/patches/mlx_lm_mtp/hy_v3_model.py" or path.startswith(
        ("tests/test_hy_v3_",)
    ):
        matches.append("U-003")
    elif "nemotron" in path:
        matches.append("E-003")
    elif "minimax" in path or "eagle3_minimax" in path:
        matches.append("E-002")
    elif "glm" in path or "int8_mla" in path or path == "omlx/patches/dsv3_decode_opts.py":
        matches.append("E-001")
    elif path.startswith("omlx/patches/mlx_lm_mtp/") or path == "omlx/speculative/vlm_mtp.py" or path.startswith(
        ("tests/test_mlx_lm_mtp", "tests/test_vlm_mtp")
    ):
        matches.append("U-002")
    elif path.startswith("omlx/tools/oqnvfp4") or path == "omlx/tools/__init__.py" or path.startswith(
        ("tests/test_oqnvfp4",)
    ) or path == "docs/oQ_Quantization.md":
        matches.append("E-005")
    elif path in {"omlx/cache/type_handlers.py", "omlx/cache/type_registry.py"}:
        matches.append("C-003")
    elif path in {"omlx/engine_pool.py", "tests/test_engine_pool.py"}:
        matches.append("U-001")
    elif path in {
        "omlx/cache/prefix_cache.py",
        "omlx/memory_monitor.py",
        "omlx/prefill_transient_tracker.py",
        "omlx/process_memory_enforcer.py",
        "omlx/scheduler.py",
        "tests/test_prefill_gate_logging.py",
        "tests/test_prefill_oom_graceful.py",
        "tests/test_scheduler.py",
        "tests/test_scheduler_prefill_memory_guard.py",
    }:
        matches.append("C-001")
    elif path in {
        "README.md",
        "omlx/admin/static/omlx_preset.json",
        "omlx/api/openai_models.py",
        "omlx/api/thinking.py",
        "omlx/engine/batched.py",
        "omlx/engine/vlm.py",
        "omlx/model_profiles.py",
        "omlx/model_settings.py",
        "omlx/models/vlm.py",
        "omlx/server.py",
        "omlx/settings.py",
        "omlx/utils/model_loading.py",
        "pyproject.toml",
    }:
        matches.append("C-002")

    return matches


def main() -> int:
    committed = git("diff", "--name-only", f"{FORK_POINT}..{LEGACY_HEAD}")
    checkpoint = git("diff-tree", "--no-commit-id", "--name-only", "-r", ARCHIVE)
    inventories = {"committed": committed, "checkpoint": checkpoint}
    failures: list[str] = []

    for source, paths in inventories.items():
        for path in paths:
            assigned = groups(path)
            if len(assigned) != 1:
                failures.append(f"{source}: {path}: {assigned or 'UNMAPPED'}")

    print(f"committed_paths={len(committed)}")
    print(f"checkpoint_paths={len(checkpoint)}")
    print(f"union_paths={len(set(committed) | set(checkpoint))}")
    if failures:
        print("coverage=FAIL")
        print("\n".join(failures))
        return 1
    print("coverage=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
