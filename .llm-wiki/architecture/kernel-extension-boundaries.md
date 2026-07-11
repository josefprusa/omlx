# Kernel Extension Boundaries

## Build Boundary

`setup.py` builds native extensions only when `--with-custom-kernel` or `OMLX_WITH_CUSTOM_KERNEL` enables them. The current extensions are GLM MoE DSA, MiniMax M3, and Qwen3.5 prefill.

## Runtime Boundary

The `fast.py` wrappers under `omlx/custom_kernels/glm_moe_dsa/`, `omlx/custom_kernels/minimax_m3/`, and `omlx/custom_kernels/qwen35_prefill/` probe import and ABI availability before exposing native symbols. Callers must retain a correct fallback when a binary or symbol is absent.

## Dependency Contract

`pyproject.toml` pins MLX 0.32.0 and nanobind 2.13.0 because bundled binaries are ABI-coupled to them.

## Provenance

Updated from `setup.py`, `pyproject.toml`, and current custom-kernel fast wrappers on 2026-07-11 at upstream `d5fcb22a`.

Decay condition: recheck on any MLX, nanobind, deployment-target, extension-name, or exported-symbol change.
