# Repo Overview

omlx is an MLX-based inference server for Apple Silicon with OpenAI-compatible APIs, continuous batching, cache persistence, model compatibility patches, speculative decoding, quantization, and optional native kernels.

## Map

- HTTP and engine path: [[architecture/serving-runtime]]
- Scheduler and caches: [[architecture/scheduler-cache-lifecycle]]
- Model loading and patches: [[architecture/model-patch-system]]
- Native extensions: [[architecture/kernel-extension-boundaries]]
- Performance domains: [[summaries/performance-work-map]]

Current repository files and tests are authoritative when they disagree with this wiki.

Updated from `README.md`, `pyproject.toml`, and the current source tree on 2026-07-11 at upstream `d5fcb22a`.

Decay condition: recheck on major package, serving architecture, or product-scope changes.
