# Parity Validation

## Procedure

1. State the reference implementation and exact Git/dependency versions.
2. Test engagement and fallback separately.
3. Compare shapes, dtypes, cache positions, metadata, and deterministic outputs.
4. Exercise boundary cases, rollback, disabled mode, and failure exits.
5. Keep existing tolerances unless a quality change is explicitly approved.
6. Run focused tests, then the broader relevant suite.

## Evidence Rule

A helper-level comparison cannot prove a serving-path claim. Verification scope must cover the claimed behavior.

For converted checkpoints, use two independent gates:

1. A header-level census proves the exact expected tensor names, shapes,
   quantization triples, and absence of dropped feature residue.
2. Deterministic byte parity recomputes representative tensors from the source
   across module families and layer depths. It must compare packed weights,
   scales, and biases, not only dequantized outputs.

Finish with a strict real-artifact load and a deterministic generation probe.
Strict loading alone proves tree compatibility, not conversion correctness.

Run the final probe through `BatchedEngine` or `VLMBatchedEngine`, not only a
library-level loader. Record load time, prompt/completion token counts, first
token latency, decode rate, peak MLX memory, output, finish reason, and required
feature-engagement logs. A short prompt cannot certify a context-gated feature;
cross its real threshold in a separate rail or mark that rail unverified.

## Provenance

Approved in the upstream rebuild plan and aligned with current tests on 2026-07-11. Revalidated by the 88-layer Puzzle artifact census and 12-source byte-parity samples in Nemotron experiment commit `ce739d7c`.

Decay condition: revisit when test infrastructure, sampler semantics, or accepted numerical tolerances change.
