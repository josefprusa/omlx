# Upstream Rebuild and Verified Knowledge Migration Plan

Status: executing

Last reviewed: 2026-07-11

## Objective

Move `josefprusa/omlx:main` to the exact current head of `jundot/omlx:main`, preserve every piece of existing work, and reconstruct the valuable behavior on a clean upstream base. Keep upstream implementations canonical, isolate experimental performance work, and progressively distill verified durable knowledge from `.claude/skills/omlx-perf/` into a repo-local Project LLM Wiki.

This plan is active execution state and therefore lives under `tasks/`. The wiki contains only durable, validated knowledge.

## Non-Negotiable Rules

1. Never reset, clean, rebase, or switch away from the current dirty checkout before the archive branch is committed and the external backups are reverified.
2. Never publish the raw archive branch to the public fork.
3. `origin/main` must be an exact fast-forward to the selected upstream commit. Do not use a PR merge, squash, or rebase merge to synchronize it.
4. Keep `main` as an exact upstream baseline. The owner explicitly waived branch protection and allows agent access because this is a pet project.
5. Do not copy old implementation files wholesale. Compare behavior against current upstream and port only unique, still-valid behavior.
6. Treat current code and tests as authoritative over wiki notes, research notes, memory, and diagrams.
7. Do not put active tasks, complete transcripts, raw logs, secrets, credentials, dumps, or unverified hypotheses in `.llm-wiki/`.
8. Experimental code cannot enter `rebuild/core` without focused tests, numerical parity where applicable, and target-Mac measurements.
9. Do not relax existing numerical tolerances or quality gates without explicit user approval.
10. Use Exa frequently for research-heavy claims. Prefer primary arXiv papers and official implementations; repository facts still come from the live source tree.

## Authoritative Starting Evidence

Observed on 2026-07-11. Recheck all remote values immediately before execution.

- Current checkout: `glm5.2-native-kernels-v0.4.5` at `1dded076eba287386e92e557ad781932c97630dd`.
- Current branch fork point with cached upstream: `14338c3e37260a43b027b7317a0f85a60cf85c35`.
- Divergence from cached `upstream/main`: 23 upstream-only commits and 5 local-only commits.
- Dirty work before this plan was added: 31 modified tracked files and 24 untracked files.
- Current archive inventory: those legacy paths plus this plan, for 31 modified tracked files and 25 untracked files.
- Fork main: `3d3f6663ae4addf9b837c5f4e4e50ea4f44a3964`.
- Live upstream main: `d5fcb22a87c3b46ab6dd91016fbbbdb1e624f374`.
- Fork `main` currently has no applied branch rules.
- `.venv`: MLX 0.31.2, mlx-lm 0.31.3.
- `.venv-mlx032`: MLX 0.32.0, mlx-lm 0.31.3. Neither environment is authoritative for current upstream.
- Project LLM Wiki skills are installed from audited commit `32202ae373719d359bb1d89265d65ecb59221461`.

Verified backups:

- `/Users/josefprusa/Backups/omlx/20260711-113229/omlx-worktree-complete.tar.gz`
  - SHA-256: `9b92768c92bb13fbebad262b38764f40df2923adf4199bd77816d1e3ec15ca83`
- `/Users/josefprusa/Backups/omlx/20260711-113229/omlx-history.bundle`
  - SHA-256: `e8c793fb310497c5fd7023f7ed777621f1ddb67c6ea33653c3fa67e5fe53846f`
  - `git bundle verify` reports a complete history.

## Branch Topology

```text
origin/main                          exact upstream baseline, no custom commits
└── knowledge/base                  .llm-wiki and root AGENTS guidance only
    ├── research/perf-notes         research skills, tasks, scripts, feature ledger
    ├── rebuild/core                production-ready reconstructed behavior
    ├── experiment/glm52-kernels-mtp
    ├── experiment/nemotron-puzzle-oq
    └── experiment/hy3-mtp-delta   create only if the audit proves a unique delta

archive/pre-rebuild-20260711        immutable local recovery branch, never pushed
```

`knowledge/base` is the shared custom base so every reconstruction branch receives the same curated wiki. Full research material remains isolated on `research/perf-notes`; only the durable wiki is shared with code branches.

## Phase 0: Freeze and Preserve the Existing State

### 0.1 Reverify the external backups

```sh
shasum -a 256 /Users/josefprusa/Backups/omlx/20260711-113229/omlx-worktree-complete.tar.gz
shasum -a 256 /Users/josefprusa/Backups/omlx/20260711-113229/omlx-history.bundle
git bundle verify /Users/josefprusa/Backups/omlx/20260711-113229/omlx-history.bundle
```

The hashes must match the values above. Stop on any mismatch.

### 0.2 Scan the pending archive content

Inspect all tracked modifications and untracked files for credentials, private paths, model tokens, `.env` material, PEM files, and large generated artifacts. The GitHub PAT must remain only in macOS Keychain; `.git/agent-github/` and `.claude/settings.local.json` must remain ignored.

Do not add an ignored file merely to make the archive look complete. The external worktree archive is the recovery source for intentionally ignored local files.

### 0.3 Create the immutable local archive branch

```sh
git switch -c archive/pre-rebuild-20260711
git add -A
git diff --cached --check
git commit -m "archive: checkpoint pre-upstream rebuild"
```

Acceptance checks:

- `git status --short` is empty after the commit.
- The archive commit contains all 31 modified tracked paths and all 25 untracked paths in the current inventory, unless the secret scan explicitly excluded and recorded a path.
- `git show --name-status --format= archive/pre-rebuild-20260711` is reconciled against the pre-commit `git status --short --untracked-files=all` inventory; no path is silently omitted.
- `glm5.2-native-kernels-v0.4.5` remains at `1dded076...`.
- `archive/pre-rebuild-20260711` is not pushed to any remote.

## Phase 1: Pin and Synchronize the Upstream Baseline

### 1.1 Fetch without touching the archived worktree contents

```sh
git fetch --prune upstream
git fetch --prune origin
BASE_SHA=$(git rev-parse upstream/main)
```

Record `BASE_SHA`, current `origin/main`, the date, and the upstream subject in the execution log inside `tasks/upstream_rebuild_feature_ledger.md`.

### 1.2 Prove the update is a fast-forward

```sh
git merge-base --is-ancestor origin/main "$BASE_SHA"
```

Stop if this exits nonzero. Do not force-push and do not invent a merge strategy.

### 1.3 Synchronize the fork exactly

```sh
git push origin "$BASE_SHA":refs/heads/main
```

Verify with read-only remote queries:

```sh
git ls-remote origin refs/heads/main
git ls-remote upstream refs/heads/main
```

Both must report the same SHA as `BASE_SHA`.

### 1.4 Record the owner protection waiver

The repository owner explicitly allows agents to access and push `main` because this is a pet project. GitHub branch protection is not required. Record the waiver in `decisions/branch-and-sync-policy.md`, keep custom commits off `main`, and verify exact remote SHAs before every upstream baseline sync.

## Phase 2: Create the Clean Reconstruction Worktree

Use the fixed sibling path `/Users/josefprusa/Projects/Temp/omlx-rebuild`, which was absent during planning.

```sh
git worktree add -b knowledge/base /Users/josefprusa/Projects/Temp/omlx-rebuild "$BASE_SHA"
cd /Users/josefprusa/Projects/Temp/omlx-rebuild
```

Authentication rules for the linked worktree:

- Git HTTPS uses the common repository config and the absolute credential helper already stored in the original Git directory.
- Invoke GitHub API operations through:

```sh
COMMON_GIT_DIR=$(git rev-parse --git-common-dir)
"$COMMON_GIT_DIR/agent-github/gh-axi" repo view
```

- Do not assume `.git` is a directory in the linked worktree.
- Keychain access may require narrowly scoped host escalation in sandboxed agent sessions.

## Phase 3: Build a Fresh Authoritative Environment

Read the new upstream `pyproject.toml` and `uv.lock` before creating the environment. Use the repository's current documented `uv` command, expected to be:

```sh
uv sync --dev
```

Then record:

```sh
uv run python -c 'import importlib.metadata as m; print(m.version("mlx")); print(m.version("mlx-lm"))'
```

Acceptance checks:

- Installed versions match the selected upstream lock/configuration.
- Imports for `omlx`, `mlx`, and the pinned `mlx_lm` source succeed.
- No test result from `.venv` or `.venv-mlx032` is used as proof for the reconstruction.

## Phase 4: Initialize the Shared Project LLM Wiki

Run the installed Project LLM Wiki helper from `knowledge/base`.

```sh
export WIKI_TOOL="$HOME/.codex/skills/project-llm-wiki/scripts/project_wiki.py"
uv run --no-project python "$WIKI_TOOL" init --dry-run
uv run --no-project python "$WIKI_TOOL" init
```

The dry run must show no conflicts. The applied init creates `.llm-wiki/` and a marker-managed section in root `AGENTS.md`.

Create and index these durable pages:

### Architecture

- `architecture/serving-runtime.md`
- `architecture/scheduler-cache-lifecycle.md`
- `architecture/kernel-extension-boundaries.md`
- `architecture/model-patch-system.md`

### Domain

- `domain/mlx-execution.md`
- `domain/quantization-formats.md`
- `domain/sparse-mla-dsa.md`
- `domain/speculative-decoding-mtp.md`

### Decisions

- `decisions/upstream-canonical-policy.md`
- `decisions/branch-and-sync-policy.md`
- `decisions/experiment-promotion.md`

### Operations

- `operations/reproducible-environments.md`
- `operations/parity-validation.md`
- `operations/benchmark-protocol.md`

### Summary

- `summaries/performance-work-map.md`

Initial pages contain only structure, current source anchors, and already reverified facts. Do not bulk-copy existing skill prose during initialization.

Run:

```sh
export WIKI_TOOL="$HOME/.codex/skills/project-llm-wiki/scripts/project_wiki.py"
uv run --no-project python "$WIKI_TOOL" lint
```

Commit the clean wiki foundation:

```sh
git add AGENTS.md .llm-wiki
git diff --cached --check
git commit -m "docs(wiki): initialize verified project knowledge"
git push -u origin knowledge/base
```

## Phase 5: Create the Research Branch and Feature Ledger

```sh
git switch -c research/perf-notes knowledge/base
```

Create `tasks/upstream_rebuild_feature_ledger.md`. It is active migration state and must not live in `.llm-wiki/`.

Every row in the ledger has these fields:

| Field | Required content |
| --- | --- |
| ID | Stable capability identifier |
| Capability | User-visible or internal behavior, not merely a filename |
| Legacy sources | Current commit(s), files, and tests |
| Upstream state | Exact upstream files, commits, or PRs covering it |
| Classification | `adopt-upstream`, `core-candidate`, `experiment`, `research-only`, or `obsolete` |
| Evidence | Source lines, test output, benchmark artifact, paper, or official implementation |
| Missing proof | Concrete gap preventing promotion |
| Target branch | One branch from the topology above |
| Wiki target | Existing or proposed durable page |
| Decision | Keep, port minimally, replace with upstream, or reject |

The ledger must cover every changed or untracked path through a capability grouping. No path may be silently dropped.

Minimum audit groups:

1. GLM-5.2 decode kernels, sparse MLA, quantization, and native ABI work.
2. GLM/Lightning MTP and speculative verification behavior.
3. Scheduler early prefix-index publication, rollback, freshness, and transient-memory clamp.
4. Prefix cache, paged SSD cache, and eviction/reconstruction behavior.
5. Engine-pool admission and failed-load memory recovery.
6. Hy3 model, parser, and MTP deltas.
7. Nemotron Puzzle, oQ conversion, gates, and parity tools.
8. MiniMax sparse/decode work.
9. Model profiles, settings, server integration, and model loading.
10. `.claude/skills/omlx-perf/`, task notes, benchmark scripts, and rejected experiments.

Known upstream-canonical areas include current Hy3 support, Lightning MTP, engine-pool recovery, MLX 0.32/nanobind ABI work, GLM sharded/indexer improvements, and prefix-cache eviction recovery. Verify live code before relying on this list.

## Phase 6: Verified Skill-to-Wiki Porting Protocol

`.claude/skills/omlx-perf/` is source material and an operational playbook. It is not copied wholesale into the wiki.

For each non-trivial verified work unit:

1. Read `.llm-wiki/index.md`, then only relevant linked pages.
2. Identify at most three candidate durable learnings.
3. Verify repository claims against current code and focused tests.
4. Verify performance claims on the target Mac with a recorded baseline and environment.
5. Use Exa for research-heavy claims; prefer primary arXiv papers and official implementations.
6. Present each candidate for human approval with:
   - category;
   - target page;
   - concise learning;
   - evidence;
   - validation performed;
   - decay condition describing when it must be rechecked.
7. After approval, update an existing page first. Create a new page only when no existing page can own the concept.
8. Add concise provenance: source title or repo anchor and date.
9. Append only a concise change record to `.llm-wiki/log.md`.
10. Run wiki lint.
11. Shorten duplicated explanatory prose in `omlx-perf` only after the wiki page exists and the skill remains operationally self-contained.

Use the helper for approved ingestion:

```sh
export WIKI_TOOL="$HOME/.codex/skills/project-llm-wiki/scripts/project_wiki.py"
uv run --no-project python "$WIKI_TOOL" ingest \
  --file CURATED_NOTE \
  --title TITLE \
  --target-page PAGE \
  --key-idea KEY_IDEA
uv run --no-project python "$WIKI_TOOL" lint
```

Never ingest full task files, full logs, complete chat transcripts, unreviewed paper text, private model data, secrets, or execution checkpoints. Curated raw preservation is off by default.

Approved wiki changes are committed on `knowledge/base`. Before the next implementation batch, active branches incorporate the new wiki-only commits from `knowledge/base`.

## Phase 7: Reconstruct the Branches

### 7.1 Research branch

Port `.claude/skills/omlx-perf/`, task documents, and useful benchmark scripts to `research/perf-notes`. Preserve negative results and rejected ideas; they prevent repeated dead-end work. Active status remains in `tasks/`, not the wiki.

### 7.2 Core branch

```sh
git switch -c rebuild/core knowledge/base
```

Only unique production behavior with complete proof enters this branch. Scheduler/cache work remains a candidate until all rollback, kill-switch, disabled-mode, and concurrency tests pass against current upstream.

### 7.3 GLM experiment

```sh
git switch -c experiment/glm52-kernels-mtp knowledge/base
```

Rebuild unique capabilities against upstream's current kernel ABI, GLM head/indexer layout, and Lightning MTP implementation. Do not restore removed kernel files or stale nanobind/MLX assumptions.

### 7.4 Nemotron experiment

```sh
git switch -c experiment/nemotron-puzzle-oq knowledge/base
```

Port Puzzle model patches, conversion tooling, offline gates, and parity tests. Keep it isolated until conversion correctness, numerical parity, and target-Mac performance are proven.

### 7.5 Conditional Hy3 branch

Create `experiment/hy3-mtp-delta` only if the ledger proves current upstream lacks a behavior that still passes against the latest model and MTP contracts. Otherwise classify the local implementation as replaced by upstream and preserve only the durable reasoning.

## Phase 8: Verification Gates

### Common gates for every branch

- `git diff --check` passes.
- Focused tests covering every changed behavior pass.
- The broader relevant suite passes.
- `project-wiki lint` passes.
- No token, credential, private model data, local absolute scratch path, or large generated artifact is staged.
- Engagement/fallback logging proves the intended optimized path actually ran.

### Core scheduler/cache gate

At minimum run focused coverage in:

- `tests/test_scheduler.py`
- `tests/test_paged_ssd_cache.py`
- current upstream cache reconstruction/eviction tests

Required semantics:

- failed persistence retracts published hash visibility, not only request bookkeeping;
- every store-cache exit path is covered;
- the early-publish kill switch reproduces upstream behavior;
- `OMLX_TRANSIENT_CLAMP_K=0` reproduces old behavior exactly;
- only tracker-derived transient terms are clamped;
- fixed abort margins and the existing safety multiplier remain unchanged;
- concurrency/call-order tests use focused mocks where real MLX storage is unnecessary.

### GLM/MTP experimental gate

At minimum run current upstream equivalents of:

- GLM native patch tests;
- custom-kernel ABI/import probes;
- MTP cache/rollback/acceptance tests;
- model-loading and settings tests.

Numerical and serving requirements:

- existing test tolerances are not weakened;
- greedy end-to-end output remains identical for deterministic validation prompts unless the feature is explicitly lossy;
- lossy features require explicit quality acceptance and a kill switch;
- measure short and long context, batch 1, with identical model, prompt, environment, and sampler;
- use warmups plus at least 12 measured repetitions and report medians;
- a speed feature needs at least 2% end-to-end median improvement, unless its approved purpose is memory capacity rather than speed.

### Nemotron/oQ experimental gate

- conversion metadata, tensor shapes, and serialization round trips are correct;
- offline gate and model patch tests pass;
- parity covers representative routing, fused decode, and conversion cases;
- quality does not regress beyond the predeclared acceptance bar;
- target-Mac measurements use the same baseline protocol as GLM.

### Full-suite gate

Before publishing a production candidate:

```sh
uv run pytest -q
export WIKI_TOOL="$HOME/.codex/skills/project-llm-wiki/scripts/project_wiki.py"
uv run --no-project python "$WIKI_TOOL" lint
```

If Metal access is unavailable in the agent sandbox, rerun only the required hardware checks with scoped host approval. Absence of a hardware result is not a pass.

## Phase 9: Promotion and Publication

1. Publish only reviewed, secret-scanned reconstruction branches.
2. Keep `origin/main` as the upstream baseline.
3. Use `knowledge/base` as the base for internal research and experiment PRs.
4. Promote small verified commits into `rebuild/core`; never merge an entire experimental branch blindly.
5. For upstream contribution, create a minimal topic branch from the latest upstream main containing only the contribution and its tests, then open a PR to `jundot/omlx:main`.
6. Do not include `.claude` research archives, task state, unrelated wiki changes, or experiment history in upstream PRs.

## Stop Conditions

Stop and ask the user before continuing if any of these occurs:

- backup checksum or bundle verification fails;
- the fork update is not a fast-forward;
- secret scanning finds ambiguous credentials or private data;
- the live upstream dependency contract cannot be reproduced;
- a proposed port requires weakening tests or numerical tolerances;
- an experiment changes output quality without an explicit lossy-feature approval;
- the fork's `main` contains a custom commit or cannot be synchronized by exact fast-forward;
- a wiki ingest would contain active state, raw logs, full transcripts, secrets, or unverified claims;
- the feature ledger cannot account for every current changed/untracked path.

## Completion Criteria

The rebuild is complete only when all of the following are true:

- external backup hashes and Git bundle verification are recorded and valid;
- the immutable local archive branch exists and was not published;
- fork `main` exactly matches the pinned upstream `BASE_SHA`;
- the owner-approved branch-protection waiver is recorded and fork `main` remains an exact upstream baseline;
- the clean worktree and fresh dependency environment are reproducible;
- `knowledge/base` contains a clean, linting `.llm-wiki/` and root guidance;
- the feature ledger accounts for all legacy committed, modified, and untracked work;
- every capability is classified with evidence and a target branch;
- verified durable skill knowledge is incrementally represented in the wiki with provenance and decay conditions;
- research-only and obsolete work is preserved without entering production code;
- core candidates pass focused and full relevant tests;
- experiments meet their parity, quality, and benchmark gates before promotion;
- only reviewed, secret-scanned branches are pushed;
- no required evidence is missing or merely inferred.
