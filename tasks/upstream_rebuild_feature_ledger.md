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

## Reconstruction Results

Verified on 2026-07-11 against baseline `d5fcb22a87c3b46ab6dd91016fbbbdb1e624f374`,
MLX 0.32.0, and mlx-lm 0.31.3.

### C-001 Scheduler and Cache

- Branch: `rebuild/core`
- Commit: `38093a76`
- Kept: early prefix-index publication with selective rollback and
  `OMLX_DISABLE_EARLY_INDEX_PUBLISH=1`; tracker-only transient clamp with
  `OMLX_TRANSIENT_CLAMP_K=0` restoring the upstream max-of-three behavior.
- Proof: 15 focused early-publish/clamp tests, including reader-before-persist,
  mixed persisted/unpersisted prefixes, executor call order, and submit failure.
- Broader gate:
  `.venv/bin/python -m pytest tests/test_scheduler.py tests/test_paged_cache.py tests/test_paged_ssd_cache.py tests/test_prefix_cache.py tests/test_prefix_cache_rotating_tip_strip.py tests/test_prefix_cache_v4_block_storage.py tests/test_prefix_divergence_probe.py tests/test_store_cache_gate.py tests/test_hot_cache.py -q`
  produced `586 passed`.
- Decision: keep on `rebuild/core`; no experimental restore telemetry was ported.

### E-001 GLM

- Branch: `experiment/glm52-kernels-mtp`
- Commit: `20b39c00`
- Kept: optional NVFP4 global tensor scales for routed GLM experts. The current
  `SwitchGLU` receives two guarded scale points instead of copying its entire
  call method. The runtime scale fold matches prescaled-weight references in
  both sorted and unsorted paths and has `OMLX_GLM_DISABLE_NVFP4_TS=1` for
  attribution.
- Proof: `10` focused tests; the broader GLM, MTP, loading, and settings gate
  produced `229 passed, 4 skipped`. NVIDIA's NVFP4 specification independently
  confirms the required block-scale times global-FP32-scale reconstruction.
- Missing gate: the converted GLM oQNVFP4 runtime artifact is no longer present,
  so no current real-checkpoint quality or target-Mac timing result exists.
- Rejected: legacy `decode_kernels.py`. Current upstream native sparse MLA
  covers fp16/bf16 and 32/64 heads, and already supplies native weighted sum and
  q8 V-up. The old helper is shape-locked and has no current ABI benchmark.
- Rejected: int8 MLA-KV promotion. The archived production dossier records the
  feature as built but disabled because fp16 KV fits the deployed GLM model.
- Decision: keep the scale fold experimental; do not promote any GLM commit to
  core without a regenerated artifact and the full quality/benchmark gate.

### E-003 and E-004 Nemotron

- Branch: `experiment/nemotron-puzzle-oq`
- Commits: `7adaf3b9`, `ce739d7c`, `90380bb0`
- Kept: heterogeneous Puzzle model construction from per-layer
  `block_configs`; deterministic oQ48 converter; exact offline census and byte
  parity gate; NemotronH NVFP4 expert tensor-scale restoration for the baked
  Ultra artifact.
- Puzzle proof: real 47 GB artifact loaded in `6.67s`, bound all `88` layers,
  exposed the expected layer-1 expert shape `(512, 1280, 128)` and top-k `4`,
  and produced coherent output. The real offline gate covered all `88` layers
  and `12` distributed byte-parity samples. Related gate: `107 passed`.
- Ultra proof: real 327 GB baked artifact loaded in `46.31s`; all `108` layers,
  `512`-element `fc1_ts`/`fc2_ts` sidecars, and the baked mamba, MoE-dense,
  attention, and lm-head affine8 families were present. The runtime scale fold
  engaged and greedy generation produced coherent output. Related gate:
  `110 passed`.
- Rejected: Puzzle fused router and fused expert pools, based on archived target
  measurements showing a `7.3x` loss or unsafe command-buffer behavior.
- Rejected after current rerun: Puzzle fused Mamba. Synthetic fp16/bf16 kernel
  parity passed, but the real MLX 0.32 identity rail failed. One layer differed
  by about `9.8e-4`; recurrent accumulation changed greedy output. All fusion
  code was removed before commit.
- Rejected: load-time Ultra DQ8 and sorted-route runtime patches. The deployed
  artifact is already fully baked; recreating those transforms adds no serving
  capability. The converter/runtime sidecar contract is the retained primitive.
- Decision: keep the three isolated experiment commits. Puzzle base support and
  conversion are promotion candidates after final branch review; Ultra scale
  support remains tied to the marked artifact format.

### U-003 Hy3

- Current upstream base model and tool parser remain canonical.
- The archived MTP delta has synthetic contract tests but no current sidecar
  load, target-Mac speed result, or quality result.
- Decision: do not create `experiment/hy3-mtp-delta`; preserve it as research
  until a real current-model proof establishes a benefit.

## Production Model Verification

Verified on 2026-07-11 through the direct omlx serving engines with MLX 0.32.0.
These are smoke and engagement measurements, not controlled old-versus-new
benchmarks.

### MiniMax M3 oQNVFP4 fused

- Artifact: `MiniMax-M3-oQNVFP4-fused`, 230 GiB on disk.
- Initial result: strict load failed on 114 unbound `gate_up_ts`/`down_ts`
  tensors. This refuted the earlier unit-test-only compatibility verdict.
- Fix: `rebuild/core` commit `9c036e42` restores the per-expert NVFP4 global
  tensor-scale fold and adds one-shot sparse-decode engagement telemetry.
- Short serving result: strict VLM load `35.431s`, peak `235.323 GiB`, correct
  output, `28.351 tok/s` over 36 measured decode tokens.
- 5k result: `5,191` prompt tokens, sparse MSA engaged with 16 blocks of 128,
  peak `295.125 GiB`, `21.621 tok/s` over 31 measured decode tokens.
- Regression proof: MiniMax/VLM focused gate `107 passed`; the later full GLM
  branch suite also covered this commit.

### Nemotron Puzzle oQ48

- Artifact: `NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-oQ48`, 44 GiB on disk.
- Branch head during proof: `experiment/nemotron-puzzle-oq@0639a650`.
- Serving result: strict batched-engine load `7.285s`, peak `44.659 GiB`,
  correct output, `60.353 tok/s` over 51 measured decode tokens.
- This confirms the actual serving wrapper, tokenizer, scheduler, and teardown;
  it supplements the earlier direct artifact and byte-parity gates.

### GLM-5.2 Alis 4.5bpw with int8 MLA-KV

- Artifact: `GLM-5.2-Alis-MLX-Dynamic-4.5bpw`, 395 GiB on disk.
- Branch: `experiment/glm52-kernels-mtp`; focused replay commits `dba30e24`
  and `16c8d406`, finalized for current cache/scheduler contracts by `4e1ee51f`.
- Kept: thresholded `Int8MLAKVCache`, native int8 block persistence,
  cross-mode and mixed-era restores, q8 native sparse MLA, scheduler restore
  conversion, and settings/profile wiring.
- Serving proof with MTP disabled and int8 forced at token 1: native kernels
  loaded, `[INT8KV] ENGAGED` logged, strict load `71.193s`, peak
  `399.155 GiB`, coherent output containing the correct calculation, and
  `21.756 tok/s` over 63 measured decode tokens.
- Tests: broad cache/scheduler gate `736 passed`; full repository gate with
  native kernels built produced `6,491 passed, 23 skipped, 67 deselected`.
- Blocker: the persisted production setting has MTP enabled, but its graft
  fails strict load because runtime expects `(256, 6144, 256)` while the shard
  supplies `(256, 6144, 192)`. Int8 is independently verified; MTP is not.
- Missing rail: two attempts to repeat the smoke beyond the exact production
  `int8_mla_kv_start=4096` threshold were killed by macOS with exit 137 during
  the 395 GiB model materialization, before generation. The threshold behavior,
  native persistence, and real start=1 engagement are tested, but the current
  real 4k-context leg remains unverified.
