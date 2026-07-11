# Reproducible Environments

## Procedure

1. Read current `pyproject.toml` before reusing commands.
2. Create a clean environment with `uv sync --dev`.
3. Record Python, omlx, MLX, mlx-metal, and mlx-lm versions plus Git SHA.
4. Verify imports for `omlx`, `mlx`, and `mlx_lm`.
5. Do not use results from a legacy environment as reconstruction proof.

## Current Baseline

At `d5fcb22a`, upstream has no tracked `uv.lock`; `uv sync --dev` generated an ignored local lock and installed Python 3.13.11, omlx 0.5.0, MLX 0.32.0, and mlx-lm 0.31.3. Imports passed.

## Provenance

Validated in the clean reconstruction worktree on 2026-07-11.

Decay condition: rerun after any dependency, Python, build-system, or lockfile change.
