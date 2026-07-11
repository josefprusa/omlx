# Upstream Rebuild Feature Ledger

Status: active migration evidence

Baseline: `jundot/omlx@d5fcb22a87c3b46ab6dd91016fbbbdb1e624f374`

Legacy fork point: `14338c3e37260a43b027b7317a0f85a60cf85c35`

Legacy committed head: `1dded076eba287386e92e557ad781932c97630dd`

Immutable checkpoint: `3f2edd6051ecdb58395d3c63118dd56e51c66aee`

Knowledge base: `debb34d64adbd39aa5bae386f8a427c37dfa3f33`

## Coverage Contract

Run:

```sh
.venv/bin/python tasks/check_upstream_rebuild_ledger.py
```

The checker derives the five-commit inventory and checkpoint inventory directly from Git. Current proof:

```text
committed_paths=166
checkpoint_paths=56
union_paths=195
coverage=PASS
```

Every legacy path maps to exactly one capability ID below. The checker fails if a path is unmapped or assigned more than once.

## Classification Summary

| ID | Capability | Legacy sources | Upstream state | Classification | Evidence | Missing proof | Target branch | Wiki target | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-001 | Performance research corpus, skills, task notes, and probes | `.claude/**`, `tasks/**`, `GLM52_MTP_FORAY.md`; introduced and expanded across the five local commits and checkpoint | Upstream intentionally has no equivalent private research corpus | `research-only` | Git inventory; individual claims remain untrusted until reverified | Per-note provenance, current code anchors, target-Mac reruns | `research/perf-notes` | Existing architecture/domain/operations pages, proposal-first | Preserve the corpus on the research branch; port only approved verified learnings to the wiki |
| U-001 | Engine-pool admission and failed-load memory recovery | Local commit `5a26eb1f`; `omlx/engine_pool.py`, `tests/test_engine_pool.py` | Upstream contains `a46e6fa6`, `58e0a6c6`, `9c815b74`, and `73e9a53b` covering failed-load reclaim, tests, back-to-back swaps, and scoped residue retry | `adopt-upstream` | Upstream Git history and current `EnginePool` tests | Focused comparison proving no local edge case remains unique | `research/perf-notes` | `architecture/serving-runtime.md` | Replace local implementation with upstream; preserve only any proven missing test case |
| E-001 | GLM-5.2 sparse MLA, int8 MLA-KV, decode kernels, routed-expert quantization, native ABI, and GLM MTP glue | Commits `603f47fe`, `e43872f6`, `3c224edd`, `1dded076`; GLM-named kernel, patch, tool, and test paths plus int8 MLA paths | Upstream has baseline GLM sparse kernels (`5b5d9228`), DeepSeek/GLM native work (`3d3f6663`), tensor-sharded 32-head support (`1011ec12`), MLX 0.32/nanobind 2.13 ABI checks (`2ce529d4`, `1c616249`), Lightning MTP (`8a9b1972`), and verify-row routing fix (`94450351`). Legacy `int8_mla_kv.py`, `decode_kernels.py`, and `nvfp4_ts.py` are absent | `experiment` | Current source comparison; [GLM-5 report](https://arxiv.org/abs/2602.15763) confirms DSA and MTP as model-level mechanisms, not this implementation | Current-ABI reconstruction, kernel parity, cache round trips, engagement logs, end-to-end quality, target-Mac short/long-context benchmarks | `experiment/glm52-kernels-mtp` | `domain/sparse-mla-dsa.md`, `domain/speculative-decoding-mtp.md`, `architecture/kernel-extension-boundaries.md` | Adopt upstream baseline; reconstruct only absent capabilities one at a time |
| U-002 | Generic native MTP, rollback, adaptive depth, and VLM MTP | `omlx/patches/mlx_lm_mtp/**`, `omlx/speculative/vlm_mtp.py`, MTP tests, excluding model-specific files claimed by other groups | Upstream Lightning MTP `8a9b1972`, atomic rollback history, per-model eligibility, late-join handling, and adaptive measured-cost depth `6342b4d9` supersede the old generic baseline | `adopt-upstream` | Current upstream MTP source/tests; [AdaEAGLE](https://arxiv.org/abs/2412.18910) supports adaptive draft structures as a research direction but does not validate omlx code | Diff each legacy behavior against current tests; prove any claimed unique fallback or statistic | `research/perf-notes` | `domain/speculative-decoding-mtp.md` | Keep current upstream generic MTP; route only model-specific missing deltas to their experiment |
| E-002 | MiniMax M3 fused index, compiled decode, fused flash, and EAGLE3 experiments | MiniMax-named patch/vendor paths and `eagle3_minimax.py` | Current upstream retains MiniMax compatibility, sparse patching, and a native top-k kernel, but legacy `compiled_decode.py`, `fused_flash_v2.py`, `fused_index.py`, and EAGLE3 module are absent | `experiment` | Current tree presence/absence audit | Current model compatibility, numerical parity, serving integration, quality, and target-Mac benefit | `research/perf-notes` | `domain/sparse-mla-dsa.md`, `domain/speculative-decoding-mtp.md` | Preserve for research; do not create a production branch until evidence justifies one |
| E-003 | Nemotron Super/Ultra DQ8, NVFP4 transform, MoE fast path, and conversion tooling | Nemotron-named paths except Puzzle-specific paths | No equivalent implementation is present in current upstream | `experiment` | Current tree presence/absence audit; legacy tests and offline gates are archived evidence only | Current model/config compatibility, conversion round trip, parity, quality, memory, and target-Mac benchmark | `experiment/nemotron-puzzle-oq` | `domain/quantization-formats.md`, `operations/parity-validation.md` | Rebuild shared conversion/parity pieces only if they still apply to current checkpoints |
| E-004 | Nemotron H Puzzle model, fused router/decode, pool C, oQ conversion, gates, and parity probes | `omlx/patches/nemotron_h_puzzle/**`, `omlx/tools/oq_puzzle_*`, Puzzle tests; added in checkpoint | Entire implementation is absent upstream | `experiment` | Immutable checkpoint source and tests; current tree absence | Model load, conversion metadata, representative router/pool/decode parity, quality bar, and target-Mac measurements | `experiment/nemotron-puzzle-oq` | `domain/quantization-formats.md`, `operations/parity-validation.md`, `operations/benchmark-protocol.md` | Reimplement against current loaders and settings; no wholesale file restore |
| U-003 | Tencent Hy3 base support, tool parsing, and experimental MTP extension | Hy3 patch and tests in checkpoint | Upstream commit `5ad89be0` already supplies base Hy3 model and tool parser support. Legacy `mlx_lm_mtp/hy_v3_model.py` and Hy3 MTP tests remain absent | `adopt-upstream` | Upstream Hy3 commit and current paths; source diff shows 1,186 added legacy MTP/test lines beyond upstream base | Prove the MTP extension matches the latest Hy3 config, cache, sampler, and tool-call contracts | `experiment/hy3-mtp-delta` only if proof succeeds | `architecture/model-patch-system.md`, `domain/speculative-decoding-mtp.md` | Adopt upstream Hy3. Create the conditional branch only for a verified unique MTP delta |
| E-005 | Shared oQ/NVFP4 streaming conversion and offline gates not owned by a model group | `omlx/tools/oqnvfp4_convert.py`, `omlx/tools/__init__.py`, generic oQ documentation/tests | Current upstream has a much newer `omlx/oq.py` and oQe support, but not the legacy standalone conversion module | `experiment` | Current source comparison; upstream oQ/oQe history | Demonstrate a capability missing from current `omlx/oq.py`, then conversion parity and round trip | `research/perf-notes` until a model experiment consumes it | `domain/quantization-formats.md` | Prefer current oQ APIs; extract only a proven missing primitive |
| C-003 | Cache type registry and handler support for experimental cache states | `omlx/cache/type_handlers.py`, `omlx/cache/type_registry.py` | Current upstream has evolved generic cache handlers but lacks the legacy int8 MLA-specific state owner | `core-candidate` | Source comparison and legacy cache tests | Reconstruct only alongside a consuming feature; prove serialization, compatibility signature, reconstruction, and eviction behavior | `experiment/glm52-kernels-mtp` | `architecture/scheduler-cache-lifecycle.md` | Keep out of core until a verified cache type needs it |
| C-001 | Scheduler/cache early publication, failure rollback, transient clamp, restore profiling, and prefill-memory changes | Scheduler, prefix-cache, memory tracker/enforcer, and scheduler/prefill tests across commit `1dded076` and checkpoint | Upstream includes per-engine prefill streams (`851de00e`), cache reconstruction and eviction fixes through `d5fcb22a`, but not the complete legacy early-publish/clamp experiment | `core-candidate` | Current source diff; legacy tests; upstream cache history | Kill-switch exactness, every store exit, hash-visibility retraction, clamp-zero exactness, tracker-only clamp, concurrency call order, focused and broader cache suites | `rebuild/core` | `architecture/scheduler-cache-lifecycle.md` | Reimplement the two narrow scheduler fixes against current code; keep restore telemetry research-only unless requested |
| C-002 | Cross-cutting settings, profiles, server/load wiring, API schemas, documentation, and dependency edits | Explicit integration paths in the checker, including `model_settings.py`, `server.py`, `model_loading.py`, API files, engines, README, preset, and `pyproject.toml` | Current upstream already contains Hy3, Lightning MTP, MLX 0.32, ABI guards, newer settings, and server behavior; legacy files mix multiple features and cannot be transplanted safely | `core-candidate` | File-by-file current source comparison | For each experiment, prove the minimum wiring still missing and test both enabled and disabled behavior | Owning feature branch; ledger preservation lives on `research/perf-notes` | `architecture/model-patch-system.md`, `architecture/serving-runtime.md` | Do not port this group as a unit; add the smallest current-tree integration change with its owning feature |

## Executable Path Mapping

`tasks/check_upstream_rebuild_ledger.py` is authoritative for coverage. Its ordered mapping is intentionally coarse at the path level and semantic at the capability level:

| ID | Path ownership rule |
| --- | --- |
| R-001 | `.claude/`, `tasks/`, and `GLM52_MTP_FORAY.md` |
| E-004 | Nemotron Puzzle patch, conversion, gate, and test paths |
| U-003 | Hy3 patch, Hy3 MTP model, and Hy3 tests |
| E-003 | Other Nemotron paths |
| E-002 | MiniMax and MiniMax EAGLE3 paths |
| E-001 | GLM, int8 MLA, and DSV3 decode optimization paths |
| U-002 | Remaining native/VLM MTP paths and tests |
| E-005 | Remaining standalone oQ/NVFP4 paths and documentation |
| C-003 | Cache type handler and registry paths |
| U-001 | Engine pool implementation and tests |
| C-001 | Scheduler, prefix cache, prefill memory, and associated tests |
| C-002 | Remaining explicit cross-cutting integration paths |

## Immediate Reconstruction Order

1. Commit and publish the research corpus and this ledger without production code.
2. Reconstruct C-001 as the first core candidate because its scope is narrow and its rollback semantics are testable without model downloads.
3. Audit E-001 against current GLM ABI and tests before porting any kernel code.
4. Rebuild E-004 conversion and parity gates before attempting fused runtime paths.
5. Audit U-003 and create the conditional Hy3 branch only if MTP remains genuinely absent and compatible.
6. Leave E-002, E-003, and E-005 research-only until a current-model proof justifies implementation work.

## Update Rule

Each reconstructed capability appends exact commits, test commands, benchmark artifacts, and its final keep/reject decision here. Durable conclusions move to `.llm-wiki/` only after human approval and wiki lint.
