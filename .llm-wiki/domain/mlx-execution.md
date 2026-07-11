# MLX Execution

## Current Contract

The upstream environment pins `mlx==0.32.0`, `mlx-metal==0.32.0`, and mlx-lm commit `ab1806e8f5d6aa035973af194a1b9198ab4754dc` (package version 0.31.3).

Native extension availability is runtime evidence, not implied by Python import success. Performance claims also require proof that the intended native path engaged.

## Source Anchors

- `pyproject.toml`
- `omlx/custom_kernels/glm_moe_dsa/fast.py`
- `omlx/custom_kernels/minimax_m3/fast.py`
- `omlx/custom_kernels/qwen35_prefill/fast.py`
- `omlx/engine_core.py` function `_init_mlx_thread`

Updated from the clean `uv sync --dev` environment on 2026-07-11 at upstream `d5fcb22a`.

Decay condition: recheck whenever MLX, mlx-metal, mlx-lm, or the custom-kernel ABI changes.
