# Benchmark Protocol

## Minimum Protocol

- Use the same target Mac, model, weights, prompt, context, sampler, environment, and server path for baseline and candidate.
- Record Git SHA, dependency versions, model identity, settings, warmups, repetitions, and raw samples.
- Use at least 12 measured repetitions and report medians.
- Measure short and long context at batch 1 unless the feature targets another workload.
- Prove the intended optimized path engaged.
- Classify costs as every-token, every-N-tokens, admission-time, or cleanup-only.

A speed feature requires at least 2% end-to-end median improvement unless its approved objective is memory capacity rather than speed.

## Provenance

Approved in the upstream rebuild plan on 2026-07-11.

Decay condition: revisit when a stable benchmark harness or statistically stronger project-wide gate replaces this protocol.
