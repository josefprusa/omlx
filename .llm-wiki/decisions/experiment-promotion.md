# Experiment Promotion

## Decision

Experimental branches are evidence generators, not merge units. Promote only small reviewed commits whose behavior is unique relative to current upstream.

## Required Evidence

- Focused tests for changed behavior and fallback.
- Numerical parity or a predeclared quality budget.
- Target-Mac end-to-end measurements for performance claims.
- Observable optimized-path engagement.
- A kill switch for lossy or risky optimizations.

## Provenance

Approved in the upstream rebuild plan on 2026-07-11.

Decay condition: revisit when CI gains stable hardware benchmarks or upstream adopts a formal experimental-feature process.
