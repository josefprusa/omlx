# Upstream Canonical Policy

## Decision

jundot/omlx main is the canonical baseline. The fork's `main` tracks it exactly; custom production work belongs on reviewed branches and is reconstructed semantically against current upstream.

## Consequences

- Adopt equivalent upstream behavior instead of retaining a local duplicate.
- Do not bulk-copy legacy implementation files.
- Preserve rejected and research-only work outside production branches.
- Prepare minimal upstream contribution branches from current upstream.

## Provenance

Approved by the repository owner during the upstream rebuild on 2026-07-11. Fork and upstream were verified equal at `d5fcb22a`.

Decay condition: revisit if the fork stops tracking jundot/omlx or becomes an independently released product.
