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

## Provenance

Approved in the upstream rebuild plan and aligned with current tests on 2026-07-11.

Decay condition: revisit when test infrastructure, sampler semantics, or accepted numerical tolerances change.
