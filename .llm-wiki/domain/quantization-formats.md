# Quantization Formats

## Ownership

- `omlx/oq.py` owns oQ planning, validation, conversion, calibration, sensitivity, and streaming quantization.
- `omlx/utils/model_loading.py` expands per-layer quantization keys and dispatches custom loaders.
- `omlx/model_settings.py` owns runtime quantization-related controls such as TurboQuant KV.

## Rule

A format is not supported merely because tensors load. Conversion metadata, tensor paths, runtime dispatch, numerical parity, and serialization round trips must agree.

For int8 MLA-KV, support additionally requires all of the following:

- dense storage below the configured start threshold;
- a raw packed/scales/biases triple after threshold crossing;
- dense rope keys and indexer caches;
- native int8 hot/SSD block persistence;
- mixed legacy-fp16 and native-int8 chain restoration;
- cross-mode restoration when int8 is enabled or disabled; and
- live `[INT8KV] ENGAGED` evidence from the serving engine.

## Provenance

Updated from `omlx/oq.py`, `omlx/utils/model_loading.py`, and `omlx/model_settings.py` on 2026-07-11 at upstream `d5fcb22a`. Production int8 MLA-KV proof is commit `4e1ee51f` on the GLM experiment branch.

Decay condition: recheck when quantization metadata schemas, tensor naming, or loader dispatch changes.
