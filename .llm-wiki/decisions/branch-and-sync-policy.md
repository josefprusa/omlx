# Branch and Sync Policy

## Topology

- **main**: exact upstream baseline.
- **knowledge/base**: wiki and root agent guidance.
- **research/perf-notes**: research material and migration ledger.
- **rebuild/core**: verified production candidates.
- **experiment branches**: isolated model and performance experiments.
- **archive/pre-rebuild-20260711**: local recovery only, never publish.

## Owner Waiver

The owner explicitly allows agents to access and push `main` because this is a pet project. GitHub branch protection is intentionally not required. Policy still requires exact-SHA verification before any baseline sync and forbids custom commits on `main`.

## Provenance

Approved by the repository owner on 2026-07-11; initial exact sync target was `d5fcb22a`.

Decay condition: revisit before adding collaborators, automation with broader credentials, releases, or external production users.
