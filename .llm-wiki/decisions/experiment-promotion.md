# Experiment Promotion

## Decision

Experimental branches are evidence generators, not merge units. Promote only small reviewed commits whose behavior is unique relative to current upstream.

## Required Evidence

- Focused tests for changed behavior and fallback.
- Numerical parity or a predeclared quality budget.
- Target-Mac end-to-end measurements for performance claims.
- Observable optimized-path engagement.
- A kill switch for lossy or risky optimizations.

Recurrent and deep models require an end-to-end deterministic identity rail.
Small per-layer errors can accumulate across recurrent state even when isolated
kernel comparisons pass ordinary tolerances. If greedy output changes, reject
the optimization unless an explicit lossy-feature quality budget was approved
before measurement. Do not benchmark first and excuse divergence afterward.

## Provenance

Approved in the upstream rebuild plan on 2026-07-11. Revalidated when the Puzzle Mamba fusion passed isolated fp16/bf16 parity but changed real-model greedy output on MLX 0.32; the uncommitted fusion was removed.

Decay condition: revisit when CI gains stable hardware benchmarks or upstream adopts a formal experimental-feature process.
