# Sparse MLA and DSA

## Current Implementations

- GLM/DeepSeek patch logic: `omlx/patches/glm_moe_dsa/`.
- GLM/DeepSeek native kernels: `omlx/custom_kernels/glm_moe_dsa/`.
- MiniMax sparse attention patch: `omlx/patches/minimax_m3_sparse_attention.py`.
- MiniMax native top-k kernel: `omlx/custom_kernels/minimax_m3/`.

## Rule

Separate index selection, attention computation, native availability, and model routing when testing or benchmarking. A fast isolated kernel does not prove end-to-end decode improvement.

MiniMax M3 oQNVFP4 artifacts may carry per-expert `gate_up_ts` and `down_ts`
sidecars. Strict loading and correct generation require binding those tensors and
applying the gate/up scales before SwiGLU and the down scale before expert
accumulation. The production fused artifact also requires a prompt beyond the
sparse density crossover to prove MSA engagement; the 2026-07-11 proof engaged
at 5,191 KV tokens.

## Provenance

Updated from current sparse patch and kernel modules on 2026-07-11 at upstream `d5fcb22a`. MiniMax tensor-scale and real sparse engagement proof is commit `9c036e42` on the core rebuild branch.

Decay condition: recheck when indexer layout, attention cache shape, top-k policy, or native symbol contracts change.
