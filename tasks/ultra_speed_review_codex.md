Overall verdict: APPROVE-WITH-CHANGES / no-go as written. The plan's core diagnosis is right: the checkpoint is mostly dense bf16 traffic plus nvfp4 routed experts, not a 55B-at-4.5bpw decode workload, and the source geometry supports a ~10.4 tok/s bandwidth ceiling before byte-cutting. The plan is not implementation-ready because P0-1's post-load quantization hook contract is overstated, the transient memory peak is not budgeted, P0-2a's batch/ragged cache gate is incomplete, and several microbench GO gates need to be expressed as weighted end-to-end token deltas rather than single-shape speed ratios.

## Plan-Item Verdicts

### Headline arithmetic / section 2: APPROVE-WITH-CHANGES
The geometry is source-backed: config has hidden_size=8192, 108 layers, vocab_size=131072, mamba/attention/MoE dimensions, top-22 experts, and omlx_moe_nvfp4_ts enabled (`~/.omlx/models/unigilby/Nemotron-3-Ultra-oQNVFP4/config.json`:15,19-127,129-154,175,1146-1147), and `nemotron_h.py` builds mamba projection_size=35072, GQA q/k/v/o linears, latent/shared MoE, and lm_head exactly as assumed (`.venv/lib/python3.12/site-packages/mlx_lm/models/nemotron_h.py`:91-132,237-265,372-424,509-523). Recomputed pure weight reads are 78.123 GB/token, not 78.7 GB; adding f32 SSM state r+w plus conv/KV gives ~79.013 GB/token, so the ceiling claim remains materially correct at ~10.4 tok/s.

### P0-1 load-time DQ8: APPROVE-WITH-CHANGES
The hook exists and is called from `BatchedEngine` after `mlx_lm.load` (`omlx/utils/model_loading.py`:513-537, `omlx/engine/batched.py`:277-288), but today it returns immediately when `model_settings is None` and only handles IndexCache, so an env-only `OMLX_ULTRA_DQ8=1` contract is not present. `nn.quantize` replaces leaf modules with `QuantizedLinear` via `model.update_modules`, not parent classes (`.venv/.../mlx/nn/layers/quantized.py`:22-95), so it should not break the `_TsFoldMoE` parent class swap if `switch_mlp.*` is excluded, but it runs after `mlx_lm.load_model` has already `mx.eval`'d all parameters (`.venv/.../mlx_lm/utils.py`:428-432), so the old bf16 dense pool and new quantized buffers coexist transiently.

### P0-2a mamba glue fusion: APPROVE-WITH-CHANGES
The target is real: mamba `_conv` does mask handling, concat, cache update, conv1d, and silu before split/SSM (`nemotron_h.py`:134-234). The gate must include `cache.left_padding is None or all-zero` and `mask is None or all-true`, not only `B==1` and `cache.lengths is None`, because `mlx_lm.generate._make_cache` sets `ArraysCache.left_padding` for batch-aware caches (`.venv/.../mlx_lm/generate.py`:838-861) and `ArraysCache.make_mask` uses left_padding/lengths (`.venv/.../mlx_lm/models/cache.py`:691-699).

### P0-2b MoE routing / epilogue trim: APPROVE-WITH-CHANGES
The ts-fold math and source hook are real: the current patch multiplies scores by `fc1_ts[inds]**2 * fc2_ts[inds]` in fp32 and applies a per-instance `_TsFoldMoE` class swap during sanitize (`omlx/patches/nemotron_h_nvfp4_ts.py`:52-68,82-117). Precomputing `combined_ts` is low-risk; fused top-22 is higher risk because it must match `group_expert_select`'s `sigmoid -> argpartition -> take -> normalize -> scale` path (`nemotron_h.py`:313-344), including ties/near-ties and ordering, so require real-gate-distribution parity and fallback counters, not only random vectors.

### P0-2c attention qkv pack: APPROVE-WITH-CHANGES
The shapes are correct: q/o are 8192->8192 and k/v are 8192->256 (`nemotron_h.py`:237-265), so an 8192->8704 packed projection is valid. The payoff is small (~24 dispatches/token per plan), so this should be gated behind M4 plus a qkv-specific in-stream probe; post-DQ8 packing of quantized rows is plausible only when all three modules share mode/group/bits and copied scales/biases are preserved exactly.

### P0-3 chunked intra-token async_eval: APPROVE-WITH-CHANGES
The scheduler does not currently run competing mid-forward evals: upstream `GenerationBatch._step` schedules one-ahead `mx.async_eval` then blocks on current tokens/logprobs (`.venv/.../mlx_lm/generate.py`:1320-1378), and oMLX's cache materialization is post-step and interval-gated (`omlx/scheduler.py`:9642-9664). The risk is not obvious deadlock; it is partially materialized cache/state if a later layer errors or fallback/recovery fires, so the required test set must include abort/retry, cache-corruption recovery, prefix-cache store/reuse, and SSM/KV state identity across long greedy decode.

### P1-1 expert nvfp4 -> affine4 requant: APPROVE-WITH-CHANGES
The real active expert shape matches M1: `SwitchMLP` uses fc1 `[512,5120,2048]` and fc2 `[512,2048,5120]` from `SwitchLinear(input_dims, output_dims, num_experts)` (`.venv/.../mlx_lm/models/switch_layers.py`:202-231), and the checkpoint has 96 nvfp4 fc1/fc2 entries (`config.json`:176-658). The `affine4 >=1.25x` single-shape gate is too weak by itself; require an in-stream projected token saving threshold after sorted/unsorted rhs_indices and M={1,2,4}, plus full quality regate because this is lossy-on-lossy requant.

### P1-2 mamba decode megakernel v2: APPROVE-WITH-CHANGES
The stateful target is real: `ssm_update` uses the Metal `ssm_update_kernel` only for `seq_len == 1`, non-null state, GPU, and Metal availability; otherwise it falls back to `ssm_attn` (`.venv/.../mlx_lm/models/ssm.py`:217-261). Keep this behind M2, but add explicit tests for state dtype preservation, `left_padding`/`lengths` fallback, first-token state-none fallback, and 1000-step recurrent drift.

### P1-3 DQ8 checkpoint productization: APPROVE
This is the clean way to remove P0-1's boot-time quantization cost and much of its transient peak: upstream loading quantizes before `load_weights` when quantization config is present (`.venv/.../mlx_lm/utils.py`:359-380,428-432), while post-load DQ8 must allocate from already materialized bf16 weights. Promote this to the preferred path if P0-1 wins but load-time peak or restart latency is unacceptable.

### P2-1 MAX_ACTIVE_TASKS cap wheel: APPROVE-WITH-CHANGES
Keep this as explicit lead-signoff work. The plan's risk framing is consistent with local lessons: `uv run` can silently revert patched wheels and live engagement must be verified (`tasks/lessons.md`:78-88), and the scheduler/MLX task-cap mechanism is a process-wide runtime change, not a per-model patch.

### P2-2 speculative decode economics re-check: APPROVE-WITH-CHANGES
It is correctly kept as "probe only / no build"; current oMLX does not apply native MTP to `nemotron_h` because `_is_mtp_compatible` only admits Qwen, DeepSeek-V4, and GLM-DSA families (`omlx/utils/model_loading.py`:386-397,456-472), and stock Nemotron sanitize drops `mtp.*` weights (`nemotron_h.py`:538-540). Add a mamba/SSM M={2,4} sequence probe, not only dense/expert matmul probes, because recurrent state update semantics are the hardest part of any verify path.

### P2-3 dense 6-bit / mixed push: REJECT
As a near-term workstream this is not justified: the plan itself estimates only ~5 ms beyond DQ8 while introducing much higher quality risk (`tasks/ultra_speed.md`:292-294). Reopen only after DQ8 per-module ablations show large quality margin and after P0-2/P0-3 actual live deltas are known.

### M0 dense microbench: APPROVE-WITH-CHANGES
The listed shapes match the real linears (`tasks/ultra_speed.md`:113-121; `nemotron_h.py`:115-132,250-265,401-405,514), but the GO gate wording is backwards/ambiguous: it should require `bf16_time / q_time >= 1.6` or `q_time <= 0.625 * bf16_time`, not "quantized mode >=1.6x bf16 time." Gate on weighted full-model token delta, and report the router gate separately because P0-1 intends to keep it bf16.

### M1 experts microbench: APPROVE-WITH-CHANGES
The fc1/fc2 shapes and top-22 rhs_indices are representative of the real `SwitchMLP` decode path (`switch_layers.py`:202-231; `nemotron_h.py`:385-416). Add an in-stream total-token projection and require sorted/unsorted cases to match the live `do_sort = indices.size >= 64` behavior, which is false for batch-1 top-22 and true for larger batches (`switch_layers.py`:220-231).

### M2 SSM microbench: APPROVE-WITH-CHANGES
The state shape `[1,256,64,128]` and f32 state assumption are supported by config and kernel dtype plumbing (`config.json`:129-133; `ssm.py`:78-97). Add first-decode-token/state-none fallback, `left_padding`/`lengths` mask cases, and a separate graph-build-only custom-kernel call cost, because `ssm_update` takes a different path when state is absent or L>1 (`ssm.py`:230-249).

### M3 route microbench: APPROVE-WITH-CHANGES
The chain is representative: gate matmul calls compiled `group_expert_select`, latent projections, switch_mlp, weighted sum, and shared expert (`nemotron_h.py`:360-424). Add a real-gate-logit corpus or recorded distribution, tie/near-tie stress, and output parity after reordering, because random gate vectors under-test argpartition edge cases.

### M4 dispatch microbench: APPROVE-WITH-CHANGES
The host-dispatch floor is useful and aligns with the local lesson to measure in-stream deltas (`tasks/lessons.md`:3-13). Add at least one `mx.fast.metal_kernel`, one `mx.quantized_matmul`, and one `mx.gather_qmm` no-op/skinny-op variant so the ROI multiplier is not derived only from tiny binary ops.

### M5 overlap microbench: APPROVE-WITH-CHANGES
This is a valid mechanism check for P0-3, but a synthetic 1800-op chain with ~40MB/op is not sufficient to predict live win because real decode mixes large GEMV, gather_qmm, SSM custom kernels, cache updates, and one-ahead sampling (`generate.py`:1320-1378). Treat M5 as a falsifier only: a negative result kills P0-3, but a positive result still requires the planned live A/B.

## Required Changes

1. Fix P0-1's hook contract: specify whether DQ8 is env-only even when `model_settings is None`, where the model config/fingerprint is read, and how the hook avoids silently doing nothing under the current `apply_post_load_transforms` early return (`omlx/utils/model_loading.py`:513-537).
2. Add a P0-1 transient-memory budget and mitigation. Recomputed target bf16 pool excluding router gate is 65.263 GB; new mxfp8/affine8 buffers add ~33.651/~34.671 GB before old buffers are released, and `QuantizedLinear.from_linear` constructs/quantizes a replacement module before `update_modules` installs it (`quantized.py`:238-255,280-302).
3. Change P0-2a's fast-path gate to reject nonzero `left_padding`, non-null `lengths`, nontrivial masks, B>1, and non-bf16 unless explicitly supported; add census fields for each fallback reason.
4. Split P0-2b into low-risk ts-precombine and high-risk fused top22. Require parity on real gate distributions, near-tie/tie cases, final MoE output, and fallback counters before any live A/B.
5. Rewrite the M0 gate as weighted model-level token saving, with clear ratio direction and per-module class results. Keep router gate excluded from P0-1 saving math unless the plan changes its quality stance.
6. Add cross-request correctness tests for every stateful optimization: request A then shorter request B, prefix-cache hit after store, cache extraction/store with state arrays, abort/retry, and batch-2/ragged fallback.
7. Promote P1-3 DQ8 checkpoint productization as the preferred production path if P0-1 wins; load-time DQ8 should be treated as a probe and emergency deployment path, not the final shape.
8. Make engagement logging stronger: log expected and actual per-layer counts plus fallback counts, not only "engaged once"; the local live-path lesson explicitly requires live-visible counters (`tasks/lessons.md`:17-26).

## Arithmetic Corrections

- Section 2 total `~78.7GB -> 78.123GB pure weight reads -> plan rounded in state/cache traffic without separating it`. Component recomputation from config/source: mamba in_proj 27.581743 GB, mamba out_proj 12.884902 GB, MoE shared 16.106127 GB, MoE latent 3.221225 GB, router gate 0.402653 GB, routed experts 12.457083 GB, attention q/k/v/o 3.321889 GB, lm_head 2.147484 GB.
- Section 2 SSM/cache row `~0.6GB -> ~0.890GB -> f32 SSM state r+w alone is 256*64*128*4*2*48 = 0.805306 GB; conv state r+w adds ~0.010617 GB and KV@6k for 12 attention layers adds ~0.073728 GB`.
- Headline ceiling `~78.7GB / 96.1ms / 10.4 tok/s -> pure weights 78.123GB / 95.388ms / 10.48 tok/s, or weights+state/cache 79.013GB / 96.478ms / 10.36 tok/s -> not materially wrong`.
- Dense/expert split `65.7GB dense vs 12.5GB expert -> 65.666GB non-expert weights including router gate, 65.263GB P0-1 DQ8 target excluding router gate, 12.457GB active routed experts -> split is correct within rounding but P0-1 target arithmetic should exclude the gate`.
- P0-1 DQ8 new total `48.1GB -> ~47.401GB mxfp8 incl state/cache or ~48.421GB affine8 incl state/cache -> plan is close for affine8 but slightly high for mxfp8; pure-weight totals are ~46.511GB / ~47.531GB respectively`.
- P0-1 resident `334->~303GB -> plausible final resident drop, but missing transient peak -> post-load path can transiently hold 334GB resident plus ~34-35GB compressed replacements and allocator scratch before old bf16 buffers are reclaimable`.

## Gaps / Missed Or Mis-Prioritized Levers

- The cheapest immediate lever is not another kernel: it is making DQ8 production as an offline checkpoint sooner if M0 passes. That avoids the current hook-contract problem, boot-time quantization latency, and much of the transient peak.
- P0-1 should be staged by module class for quality: mamba in/out first, then shared/latent/attention, then lm_head last. `lm_head` saves only ~1.1GB vs bf16 but touches final logits directly, so it deserves an isolated quality/latency ablation.
- P0-2a must account for `left_padding=[0]` as the common batch-aware singleton state. Engaging on all-zero left padding may be valid, but the plan must say so and assert it; nonzero left padding must fall back.
- P0-3 needs store-cache/prefix-cache tests because oMLX materializes cache arrays on the owner thread before background cache storage (`omlx/scheduler.py`:8949-9026). Mid-forward async eval changes when those arrays become concrete, which should be value-identical but is not covered by a 100-step greedy-only synthetic test.
- The microbench suite lacks a small end-to-end synthetic Nemotron forward that composes DQ8 + fused route + fused conv + chunked eval together. Component probes will miss interaction bugs, especially stale cache state and dead fast-path engagement.
- P2 speculative economics needs a mamba sequence probe. Dense linear M-scaling is not enough because Nemotron decode contains 48 recurrent mamba layers whose L>1 path switches from `ssm_update_kernel` to `ssm_attn` (`ssm.py`:230-249).
