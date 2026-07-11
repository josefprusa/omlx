# Model Patch System

## Dispatch

In `omlx/utils/model_loading.py`, function `maybe_apply_pre_load_patches` reads model configuration and installs model-specific compatibility patches before loading. `load_text_model` is the text load boundary, and `apply_post_load_transforms` owns transformations that require a loaded model.

## Rules

- Gate patches by explicit model/config evidence.
- Keep patch application idempotent.
- Preserve an unaffected-model fallback.
- Test both engagement and non-engagement.
- Treat sanitize correctness separately from decode-time feature enablement.

## Provenance

Updated from `omlx/utils/model_loading.py` and `omlx/patches/` on 2026-07-11 at upstream `d5fcb22a`.

Decay condition: recheck when pinned MLX model loaders absorb a patch or config dispatch changes.
