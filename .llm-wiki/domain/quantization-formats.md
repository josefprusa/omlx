# Quantization Formats

## Ownership

- `omlx/oq.py` owns oQ planning, validation, conversion, calibration, sensitivity, and streaming quantization.
- `omlx/utils/model_loading.py` expands per-layer quantization keys and dispatches custom loaders.
- `omlx/model_settings.py` owns runtime quantization-related controls such as TurboQuant KV.

## Rule

A format is not supported merely because tensors load. Conversion metadata, tensor paths, runtime dispatch, numerical parity, and serialization round trips must agree.

## Provenance

Updated from `omlx/oq.py`, `omlx/utils/model_loading.py`, and `omlx/model_settings.py` on 2026-07-11 at upstream `d5fcb22a`.

Decay condition: recheck when quantization metadata schemas, tensor naming, or loader dispatch changes.
