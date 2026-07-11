# Speculative Decoding and MTP

## Current Paths

- Native text MTP patching and verify cycles live under `omlx/patches/mlx_lm_mtp/`.
- VLM assistant MTP lives in `omlx/speculative/vlm_mtp.py` and scheduler integration.
- DFlash is a separate speculative path.
- `omlx/model_settings.py` class `ModelSettings` validates incompatible combinations.

## Invariants

Draft state is speculative; accepted tokens, sampler semantics, target-cache position, and rollback must match the standard path. Engagement must be observable, and disabled mode must reproduce the non-speculative path.

An MTP sidecar must match the loaded target model's exact expert geometry. On
2026-07-11 the production GLM-5.2 setting was rejected because the runtime
expected `down_proj.weight` shape `(256, 6144, 256)` while the graft supplied
`(256, 6144, 192)`. Disable MTP until the sidecar and target quant are rebuilt
from the same model geometry; do not relax strict loading.

## Provenance

Updated from current MTP, VLM MTP, DFlash, scheduler, and settings code on 2026-07-11 at upstream `d5fcb22a`.

Decay condition: recheck when batch ownership, acceptance sampling, rollback APIs, or speculative-path exclusivity changes.
