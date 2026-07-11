# LEVER #1 Phase B — REFACTOR INVENTORY (M3 real decode step) — compile-builder — 2026-07-04

Read-only inventory of `.../minimax_m3_vl/language.py` (2472 lines) + `fused_index.py`
+ mlx_lm `KVCache`. Scope = the **L=1 single-sequence decode** path (the compile target).
Nothing edited. All refs are `language.py:LINE` unless noted.

## TL;DR headline counts
- **Per-step scalars into graph construction:** 1 root value (`cache.offset`, a python int)
  fans out to **~13 graph-entry sites** across rope, sparse index builders, fused-kernel
  params, and 2 KV-cache buffer classes. All trace back to one fix: carry offset as `mx.array`.
- **Census / side-effect touchpoints on decode path:** **16** `_m3c()` sites + 1 step counter
  + 1 route-trace append (already compile-guarded in-tree) + 2 once-flag log side-effects.
  ALL are env-gated OFF in production → dead branches under compile. None block; keep on fallback.
- **Per-step VALUE branches (bucket-key inventory):** **~10** live branches; only **3 real
  bucket axes**: (cap → growth), (total_len vs sparse-crossover), (L: 1=decode / 2–8=MTP verify).
  The rest are config/env constants fixed at load → collapse to one path per bucket.
- **Seam:** `MiniMaxM3Model.__call__` layer loop (2082–2085). Recommend a dedicated compiled
  decode-step over the 60-layer stack, NOT compiling the branchy attention dispatcher in place.

---

## (1) Per-step python-int/float scalars crossing into graph construction

Root cause: mlx_lm `KVCache.offset` is a **python int** (`.venv/.../mlx_lm/models/cache.py:331`,
incremented `self.offset += ...` at :352). `MiniMaxM3KVCache.offset`/`index_offset` mirror it
(language.py:300-306, 360). Everything below is derived from that int and enters the graph.

| site | scalar | current form | must become |
|---|---|---|---|
| 1604 | `offset = cache.offset` | python int | `mx.array` (root state, `inputs=/outputs=`) |
| 1605,1608,1610 | `rope_offset` | python int (int path) / `mx.array` (position_ids path) | `mx.array` always — **rope already accepts array offset** via the position_ids path (proven live), so no blocker |
| 1714,1715,1725,1726,1729,1730 | `self.rope(..., offset=rope_offset)` | int | pass the `mx.array` offset (6 call sites, same var) |
| 1611 | `sparse_q_start = cache.index_offset` | python int | `mx.array` (or derive from offset state) |
| 1054,1160,1161 | `_build_sparse_decode_indices(q_start)`; `cur_block=q_start//blk`, `local_start=max(cur_block-…,0)` | python ints → `mx.where` mask consts | mx.array offset; compute cur_block/local_start as array ops (mx-fallback path only) |
| 1088–1092 | `fused_topk_blocks(total_len,…)` | `total_len` python int → **baked** `mx.array([nb,cur_block,init,local_start,emit])` inside fused_index.py:962-968 | pass total_len/cur_block/local_start as an `mx.array` **input** to the kernel (Spike B fix). **This is the #1 refactor.** |
| 1562,1566 | `mx.array(total_len,…)`, `mx.array(total_len-1,…)` | total_len python int baked | array offset-derived (masked path only) |
| 890 | `_sparse_query_positions`: `mx.arange(q_start, q_start+L)` | q_start python int | array offset (masked path only) |
| 1745,1747 | `int(os.environ["OMLX_M3_SPARSE_MIN_K"])` then `if idx_keys.shape[2] > max(...)` | **per-call env read** + shape branch | hoist env read to load-time constant; keep threshold compare as a host bucket decision |
| cache buffers | mlx_lm `KVCache.update_and_fetch` in-place `self.keys[...,prev:offset,:]=keys` (cache.py:280,352) + M3 index `self.index_keys[...,prev:index_offset,:]=keys` (language.py:361) | python-slice assign w/ python-int offset | `mx.slice_update(buf, k, start_indices=off.reshape(1), axes=[2])` (Spike C) |

Trace-time constants (safe as-is; retrace once, never change): `float(self.scale)` (1081,1204),
`float(alpha/limit/beta)` (749-751), `k`/`routed_scaling_factor`/`scoring_func` in the already-
`@mx.compile`d `_minimax_moe_select` (139-162), block_size/topk/init/local (config).

Shape-derived (bucket keys, not per-token landmines by themselves): `total_len=idx_keys.shape[2]`
(1058), `num_blocks`/`pad` (1066-1067), `k_eff=min(topk,num_blocks)` (1170). These are constant
*within a cap bucket* but note `total_len` grows every token → any site that BAKES it (rows above)
still thrashes; sites that only use it for shape math are fine.

## (2) Census / `_m3c` / route-trace side-effects on the decode path

All gated by module-level `_M3_DBG_EVERY` (int from `OMLX_M3_DEBUG_PATH`, default **0** → off) or
`_M3_ROUTE_TRACE_ENABLED`/`_M3_ROUTE_ACTIVE` (default off). When off, the `if` is a python constant
`False` → dead-code-eliminated at trace → zero effect. When ON, they fire **once per trace, not per
token** (Spike B/C finding) → undercount. Keep census on the uncompiled fallback path.

- `_m3c()` call sites (16): 756, 759, 879, 1084, 1097, 1106, 1207, 1221, 1224, 1496, 1519, 1530,
  1754, 1780, 1812, 1840.
- Step counter: `_M3C["step"] += 1` + logging at **2065-2079** — sits in `MiniMaxM3Model.__call__`
  ABOVE the layer loop → naturally outside the per-step compiled region. No action if seam is the loop body.
- Route trace: `_M3_ROUTE_TRACE.append(...)` at **159**, INSIDE the `@mx.compile`d
  `_minimax_moe_select`. **The codebase already documents the exact hazard** (comments 124-131:
  "the append below runs at mx.compile trace time and captures graph placeholders… harness calls
  `mx.disable_compile()` first"). Direct in-tree confirmation of the Spike B/C mechanism.
- Once-flag log side-effects: `_warn_nvfp4_ts_disabled()` (1986, only on the numerically-wrong
  bypass path) and `MiniMaxAttention._dbg_done` setattr+log (1660-1670, env-gated). Both harmless
  (fire once) but must not be relied on inside compile.

## (3) Per-step VALUE branches (bucket-key inventory)

Branches whose truth depends on per-step VALUES (offset/total_len) or runtime flags — these decide
how many compiled buckets exist. Shape-only branches (L, K, head_dim, num_blocks) are noted as such.

**Real bucket axes (structural — different compiled graph per value):**
- **Cap / cache growth** — mlx_lm `KVCache.step=256` (cache.py:326) + M3 index growth
  (language.py:348-356). Buffer shape changes every 256 tokens → recompile. Bucket key = cap. (Spike C/D: ~11ms/bucket.)
- **Sparse-vs-dense crossover** — `if idx_keys.shape[2] > max(blk*topk, OMLX_M3_SPARSE_MIN_K)` (1747).
  Below → dense `scaled_dot_product_attention` (1858); above → fused sparse path. Two distinct graphs.
  Bucket key = (total_len ≷ threshold). In steady long-context decode it's latched True → one bucket.
- **L regime** — `L==1` decode vs `2<=L<=8` MTP verify (1059, 1194, 1330, 1480, 1755). Different
  kernels (single vs `_multi`). Separate buckets; MTP verify is its own compile target.

**Value branches that latch to a constant in steady L=1 decode (collapse to one path/bucket):**
- 1062 `if not explicit_positions and q_start+1 != total_len: return None` — latched False (contiguous decode).
- 1488-1494 `use_maskless_compact = topk_all_valid and mask is None and q_positions is None and
  q_start+L==total_len and local_blocks>0` — latched True in steady decode → maskless path.
- 1750/1802/1824 `compact_candidate` (from `_can_use_sparse_decode_attention`, 1314) — latched True.
- 1810 `if decode_indices is None` — latched False when fused index path engages.
- 1524 `if fused_positions is not None` — latched by `OMLX_M3_DISABLE_FUSED_POSITIONS` (const env).
- 1613-1652 regime setup (position_ids/padding/mask normalization) — resolve to a fixed branch given
  (B=1, L=1, position_ids=None, zero/no padding).

**Config/const branches (fixed at load, never per-step):** 723 `do_sort=indices.size>=64` (decode top-k
< 64 → False), 728 `self.training` (False), 732/1977 `nvfp4_ts` flags, 1882-1898 shared-expert mode,
1946 `e_score_correction_bias is not None`, 1966 `pack_shared_expert`, 1992 `shared_experts is not None`,
2007 `is_moe_layer` (3 dense + 57 MoE layers → two layer templates).

**Host predicates that must STAY in python (path selectors, not tensor ops):**
`_can_use_sparse_decode_attention` (1314), `_can_use_msa_prefill_attention` (1337). They return bools
from shapes/density; run once on the host to pick which compiled bucket to dispatch.

## (4) The seam — where a compiled decode step splices in

**Layer loop:** `MiniMaxM3Model.__call__`, lines **2082-2085**:
```
for idx, (layer, c) in enumerate(zip(self.layers, cache)):
    h = layer(h, mask, c, position_ids=position_ids)
```
Each `layer` = `MiniMaxDecoderLayer` (2020: `h = x + self_attn(norm(x)…); return h + mlp(norm(h))`).
The MoE block (1941), PackedSwitchGLU (719), decoder layer (2020), RMSNorm are **clean** — every
branch is a config/shape constant in L=1 decode. The ONLY branchy code is `MiniMaxAttention.__call__`
(1596-1862): ~15 python branches, a per-call env read (1745), and **5 data-dependent `return`s**
(1783, 1799, 1843, 1862 + verify 1783). Do NOT wrap that dispatcher in `mx.compile` as-is.

**Recommended seam = a dedicated compiled decode-step over the whole 60-layer stack.** The host
wrapper (uncompiled) checks the regime once (L==1, above crossover, no padding, position_ids None),
then calls the compiled step; anything else falls through to today's eager dispatcher.

Compiled `decode_step(input_id_or_h, offset_arr)` — `mx.compile(step, inputs=STATE, outputs=STATE)`:
- **Per-step inputs:** `h` (embed of the new token, `[1,1,hidden]`); `offset` as a 0-dim `mx.array`
  (drives rope, slice_update, and total_len=offset+1 for sparse cur_block).
- **STATE (captured, mutated via slice_update — Spike C):** per-layer `kv.keys[i]`, `kv.values[i]`
  (`[1,Kh,cap,D]`), per-layer `index_keys[i]` (`[1,4,cap,index_dim]`), and `offset` (advanced with
  `offset+1` inside, written back). 60 layers × 3 buffers + 1 offset.
- **Constants (closed over):** all packed/quantized weights (q/k/v/o pack from `_omlx_packed_
  projections` 843, gate 1900, gate_up/down experts, `gate_up_ts`/`down_ts`), norm weights, rope
  freqs, scale, block/topk/init/local, alpha/limit/beta, routing config.
- **Output:** `self.norm(h)` final hidden `[1,1,hidden]`. lm_head/logits stay OUTSIDE the compiled
  step (host).
- **Bucket key:** `cap` (rebucket at each 256 growth) × regime (dense/sparse, decode/verify). Rebuild
  the compiled fn on cap change (Spike D: ~11ms, 43µs/token amortized).

**Prerequisite adaptation:** `MiniMaxM3KVCache` + mlx_lm `KVCache` must expose their raw buffers as
compile state and switch their in-place python-slice writes (cache.py:280/352, language.py:361) to
`slice_update` with an `mx.array` offset. This is the single largest mechanical change; it is the
Spike C pattern applied to the real cache classes.

## Fix checklist (ordered by leverage)
1. `mx.array` offset threaded from the cache through rope + all sparse builders (kills per-token retrace).
2. Refactor `fused_index.fused_topk_blocks` / `_multi` (962-978, 902-918) + `_sparse_decode_attention`
   total_len bakes (1562/1566) to take a scalar `mx.array` params input, not python ints.
3. KV/index cache buffers as compile state + `slice_update` writes (Spike C).
4. Extract the steady-decode tensor path out of the `MiniMaxAttention.__call__` dispatcher into the
   compiled step; leave the dispatcher as the eager fallback.
5. Hoist the per-call `OMLX_M3_SPARSE_MIN_K` env read (1745) to load time.
6. Keep ALL `_m3c`/route-trace/warn side-effects on the fallback path (they DCE-out when off; would
   under-count if on). Gate the whole thing behind `OMLX_M3_COMPILE=1` (default OFF).
