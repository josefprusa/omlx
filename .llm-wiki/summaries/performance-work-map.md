# Performance Work Map

## Durable Areas

- Serving and admission: [[architecture/serving-runtime]]
- Scheduler and persistence: [[architecture/scheduler-cache-lifecycle]]
- Native kernels and ABI: [[architecture/kernel-extension-boundaries]]
- Model compatibility patches: [[architecture/model-patch-system]]
- Sparse attention: [[domain/sparse-mla-dsa]]
- Speculative decoding: [[domain/speculative-decoding-mtp]]
- Quantization: [[domain/quantization-formats]]

## Validation

- Environment: [[operations/reproducible-environments]]
- Numerical and behavioral parity: [[operations/parity-validation]]
- Performance evidence: [[operations/benchmark-protocol]]
- Promotion: [[decisions/experiment-promotion]]

## Provenance

Seeded from the current upstream source tree and approved rebuild topology on 2026-07-11 at `d5fcb22a`.

Decay condition: update when a new performance domain becomes durable or an existing domain is removed.
