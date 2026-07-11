> Verified 2026-07-09 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# KV Cache — hot (RAM) & cold (SSD) prefix caching

**One-paragraph model.** omlx caches KV **by prefix**, in blocks of `block_size` tokens, keyed by a *chain hash* of the tokens (not the weights). A block lives in a RAM block pool (`PagedCacheManager`), and — only if a disk dir is configured — persists to SSD as one `.safetensors` file per block (`PagedSSDCacheManager`). The SSD tier **survives server restarts**. A repeat prompt therefore re-uses cached KV (warm hit) instead of re-prefilling; a novel prefix is a miss (cold prefill). The single most important consequence: **the cache key does not include a weight hash**, so re-quantized/edited weights under the same model name silently serve stale KV (§6).

Decision tree — "why did my prompt re-prefill?":
- Prompt token-for-token different from any prior (even 1 token, or different `extra_keys`/image hash) → **miss by design** → cold prefill. Chain hash breaks at the first differing 64-token block (`paged_cache.py:956` `get_computed_blocks`, breaks on first miss).
- Prompt identical but server just started → SSD scan must finish; first identical prompt is a **cold SSD hit** (disk read), not a re-prefill (§3).
- Prompt identical, prior request's store still in flight → admission may **defer** up to 4–30s (freshness bridge, §4), not re-prefill.
- Prompt identical, but model name/weights changed → **poisoning risk**, or a compat mismatch de-indexes the block (§6).

## 1. Layer map (class → role → file)

| Class / symbol | Role | Location |
|---|---|---|
| `CacheManager` (ABC) | Uniform `fetch/store/evict/clear/get_stats/size/max_size` contract | `omlx/cache/interface.py:15` |
| `CacheConfig`, `CacheFactory` | Build + wire the stack; **caching happens ONLY if `paged_ssd_cache_dir` is set** | `omlx/cache/factory.py:24,51` |
| `PagedCacheManager` | RAM block pool (vLLM BlockPool style): alloc, refcount, COW, O(1) LRU | `omlx/cache/paged_cache.py:484` |
| `CacheBlock` / `FreeKVCacheBlockQueue` / `BlockHashToBlockMap` / `BlockTable` | Block record / free list (doubly-linked LRU) / hash→block map / per-request block list | `paged_cache.py:127,194,378,446` |
| `BlockAwarePrefixCache` | Prefix index (`hash→(len,block_ids)`), `fetch_cache`/`store_cache`/`reconstruct_cache` | `omlx/cache/prefix_cache.py:52` |
| `PagedSSDCacheManager` | Disk tier: `.safetensors` per block, pending-write buffer, startup scan, hot cache | `omlx/cache/paged_ssd_cache.py:902` |
| `SharedHotCacheBudget` | Global LRU of in-RAM decoded blocks **across all loaded models** | `paged_ssd_cache.py:752` |
| `BoundarySnapshotSSDStore` | Ephemeral SSD snapshots of non-sliceable layers (ArraysCache) during prefill | `omlx/cache/boundary_snapshot_store.py:57` |
| `ModelCacheConfig` / `LayerCacheConfig` | Per-layer cache-type map; `is_hybrid`, sliceable-layer count | `omlx/cache/hybrid_cache.py:41,19` |
| `CacheType` / `CacheTypeHandler` / `CacheTypeRegistry` | Type detection + per-type slicing rules | `omlx/cache/type_handlers.py:29,76`; `type_registry.py:1` |
| `PrefillReadyRotatingKVCache` | Clamps `size()` to buffer length for SSD-restored rotating caches | `omlx/cache/_rotating_subclass.py:25` |
| `CacheRecoveryManager` | Detect/recover cache corruption, reschedule affected requests | `omlx/cache/recovery.py:22` |
| `CacheRateTracker` | Windowed hit/miss stats | `omlx/cache/observability.py:13` |

**Composition.** The `Scheduler` owns three handles: `paged_cache_manager` (RAM pool), `block_aware_cache` (prefix index), `paged_ssd_cache` (disk). `paged_cache.set_paged_ssd_cache_manager(...)` wires RAM→disk (`factory.py:233`). Comment of record: *"oMLX only supports paged SSD-based caching. Memory KV cache is managed by mlx-lm's BatchGenerator. When paged SSD cache is disabled, no oMLX caching is performed"* (`factory.py:8-11`). So with no disk dir, the only KV cache is mlx-lm's in-RAM `BatchGenerator` (no cross-request prefix reuse).

## 2. Hot path (RAM) — PagedCacheManager + BlockAwarePrefixCache

**Block pool.** `block_size` default **64 tokens** (`paged_cache.py:505`, `factory.py:42`). `initial_blocks=256`, grown dynamically up to `max_blocks` (1024 factory / 1000 class default) (`paged_cache.py:526-532`). Block 0 is a reserved **null block**, never freed, `ref_count=1` (`paged_cache.py:547-550`). Free list is a doubly-linked `FreeKVCacheBlockQueue` for O(1) LRU (`paged_cache.py:534,194`).

**Refcounts & sharing (NOT copy).** `CacheBlock.is_shared()` = `ref_count>1` (`paged_cache.py:168`). A prefix hit **shares** blocks by `increment_ref` — no data copy (`prefix_cache.py:347-353`). `increment_ref`/`decrement_ref`/`release_for_eviction` at `paged_cache.py:827,842,846`.

**Copy-on-write.** `fork_block_table` bumps refs on every block (`paged_cache.py:1168`); `get_blocks_for_generation` COW-copies a block **only if shared**, before writing (`paged_cache.py:1191-1221`). Key detail: in paged-SSD-only mode `_cow_copy_block` **does not copy KV bytes** — it allocates a fresh block carrying the same `block_hash`/`token_count`; the actual KV is reloaded from SSD when needed (`paged_cache.py:1223-1246`).

**Chain hashing (the cache key).** `compute_block_hash(parent_hash, block_tokens, extra_keys, model_name)` (`paged_cache.py:78`) folds `model_name` into every block hash (`paged_cache.py:102-103`) → per-model isolation. `get_computed_blocks` walks full 64-token blocks, threading `parent_hash`; the **first miss breaks the loop** — only a *contiguous* prefix from token 0 can hit (`paged_cache.py:956-1029`).

**Prefix index (BlockAwarePrefixCache).** `_prefix_index[block_hash] = (prefix_len, block_ids_tuple, num_blocks)`, built by `_update_prefix_index` with chained hashes to avoid O(n²) (`prefix_cache.py:2774-2810`). `_find_best_prefix_match` re-chains and takes the longest exact-length match (`prefix_cache.py:2739`). **De-index on miss/mismatch:** `_forget_incompatible_ssd_block` clears the *local* index only (a block stale here may still be valid for another model sharing the dir) (`prefix_cache.py:204`). **Truncate-to-valid-prefix:** `_find_walk_back_truncation_point` walks back to the latest block where **all non-sliceable layers carry real (non-placeholder) state** (`prefix_cache.py:1410`); `reconstruct_cache` stops at the first invalid block and rewrites `block_table` in place (`prefix_cache.py:1653,1670`).

**Concrete flow — prompt P, 60% of P already cached:**
1. Scheduler admits P → `BlockAwarePrefixCache.fetch_cache(request_id, tokens)` (`prefix_cache.py:311`).
2. → `paged.find_shared_prefix` → `get_computed_blocks`: chain-hash each 64-tok block; matches accumulate until the first uncached block, then break (`paged_cache.py:1156,956`). Matches ≈ 60% of P.
3. **Lazy restore:** for a block whose hash is unknown to RAM but present on SSD, allocate a **metadata-only** block (`ref_count=0`) and insert into the hash map — no bytes loaded yet (`paged_cache.py:1004-1018`).
4. Hits → `create_block_table`, `increment_ref` each shared block; `remaining` = the uncached ~40% of tokens (`prefix_cache.py:343-366`).
5. Prefill computes KV **only for `remaining`**; KV for the shared 60% is materialized by `reconstruct_cache` (`load_block` per block, RAM/SSD) (`prefix_cache.py:1653`).
6. On completion `store_cache` (or SF-1 early publish, §4) re-indexes the newly computed blocks so the next repeat hits 100%.

## 3. Cold path (SSD) — PagedSSDCacheManager

**On-disk layout.** Root `paged_ssd_cache_dir/{model_name}/` (`factory.py:120-122`) → 16 single-hex-char subdirs `0`–`f` (`SUBDIR_CHARS`, `paged_ssd_cache.py:924`) → one file per block `{block_hash_hex}.safetensors` (`paged_ssd_cache.py:1330-1332`). Writes are atomic: write `{stem}_tmp.safetensors`, then rename (`paged_ssd_cache.py:1578`). Default cap `max_paged_ssd_cache_size = 100GB` (`factory.py:46`).

**Save = readable BEFORE it hits disk.** `save_block` inserts the raw tensor bytes into `_pending_write_buffers[hash]` **and** enqueues onto `_write_queue`; a background writer thread flushes to `.safetensors` via `_write_safetensors_no_mx` (`paged_ssd_cache.py:1058-1065,1212-1230`). `load_block` reads in priority order: **hot cache (RAM) → pending-write buffer (RAM, no I/O) → disk index** (`paged_ssd_cache.py:2328-2400`; docstring at 2332 "Checks pending writes first"). So a just-saved block is immediately loadable while the disk write is still in flight (`paged_ssd_cache.py:2371-2392`). Queue depth `_compute_max_pending_writes` scales by block bytes: 10% RAM target, 30% hard cap, floor 32, ceiling 256 (`paged_ssd_cache.py:56-64,79`).

**Survives server restarts.** At construction, if the dir exists, `_scan_existing_files()` globs every subdir's `*.safetensors`, reads each file's metadata, and indexes only blocks **compatible** with the currently loaded model/layout; incompatible files are left on disk (a shared dir can serve multiple models) (`paged_ssd_cache.py:1034-1038,1342-1390`). This rebuilds `_index` from disk — no external DB.

**Flow — server restarted, same 5k prompt arrives:**
1. Startup scan rebuilds `_index` from `.safetensors` on disk (`paged_ssd_cache.py:1342`).
2. `get_computed_blocks` finds each block hash absent from RAM but `paged_ssd_cache.has_block(hash)` true → lazy-registers metadata block (`paged_cache.py:1004`).
3. `reconstruct_cache` → `load_block` reads the safetensors from disk (misses hot cache & pending buffer) (`prefix_cache.py:1653`, `paged_ssd_cache.py:2394-2402`).
4. **Timing (measured):** warm exact-repeat 5k TTFT **~3.0s vs 40s** cold re-prefill (SF-1, `tasks/todo.md:1057-1058,1075`); M3 same-16k-prompt restore TTFT **4.0s** vs ~213s fresh (`tasks/todo.md:503,509`); GLM restored prefixes @14.5k/@57.7k = **2.9s / 5.1s** (`tasks/todo.md:516-517`). CAUTION: the SSD/prefix cache stores **fp16** blocks for turboquant sessions (`tasks/todo.md:175`); int8 MLA-KV sessions persist **native int8 triples** since 2026-07-07 (element-count format dispatch; legacy fp16 blocks still restore via streamed requant — Int8MLAKVCacheHandler).

## 4. SF-1 — early prefix-index publish (this branch's fix)

**The 40s commit-lag bug.** The stock single-worker store published the prefix index only when the *whole* async store completed. On Ultra 550B an exact-repeat prompt therefore **re-prefilled from scratch** (full ~40s); the cache entry became usable only ~40s after the creating request finished — the 2nd submission missed, the 3rd hit (`tasks/todo.md:1057-1058`).

**The fix (inference thread publishes index right after eval).** In `_cleanup_finished` the owner thread does a **FULL `mx.eval(*pre_eval_arrays)`** — not `async_eval` — to fully materialize KV arrays, because MLX streams are thread-local and a lazy array bound to the engine's stream cannot be materialized on the background worker (`scheduler.py:9578-9600`; worker-side rationale `scheduler.py:2459-2491`). It then calls `_publish_cache_index_metadata(...)`, which **immediately** registers block hashes, `_request_tables`, and `_prefix_index`, returning an `_EarlyCacheIndexPublish` (`scheduler.py:2040-2198,9602`). The background worker keeps only the host memcpy + SSD persist, via `_persist_early_published_cache` (`scheduler.py:2200,9638`).

**Retraction — 3 failure paths** call `_retract_early_cache_index_publish` (de-index unpersisted blocks, `forget_block` on SSD, `free_block`, `paged_ssd_cache.py`) (`scheduler.py:2382-2444`):
- Submit `BaseException` before/at executor submit → `scheduler.py:9651-9658`.
- Outer `Exception` in the store block → `scheduler.py:9693-9695`.
- Worker persist returns not-ok, or worker raises → `scheduler.py:2537-2538,2550-2551`.

**Freshness bridge (complementary, for concurrent in-flight stores).** `_should_defer_for_cache_freshness` defers a *new* request's admission **without blocking the step** until a relevant in-flight store is visible (`scheduler.py:6351`). Timeout is EWMA-scaled: `min(30.0, max(4.0, store_ewma*1.5 + 0.5))` — base 4s, **cap 30s** (`scheduler.py:6265-6274`). Minimum common-prefix threshold scales by `prefill_rate*timeout*0.5`, capped at 8192 tokens (`scheduler.py:6276-6297`, const `_CACHE_FRESHNESS_WAIT_MIN_PROMPT_TOKENS=8192` at `6259`).

**Kill switch.** `OMLX_DISABLE_EARLY_INDEX_PUBLISH=1` → `_publish_cache_index_metadata` returns `None` and the code reverts to the stock `store_cache` path (`scheduler.py:2037-2038,2050`, verified by `test_worker_stock_path_unchanged_when_publish_disabled`).

**Tests.** `TestEarlyCacheIndexPublish` (`tests/test_scheduler.py:1929`): `test_publish_registers_hash_visible_before_any_persist` (1968), `test_retract_removes_hash_registration_for_unpersisted_block` (1998), `test_retract_is_selective_leaves_persisted_blocks_intact` (2024), `test_retract_while_reader_holds_ref_survives_and_deindexes` (2176), `test_worker_retracts_on_persist_failure_not_just_clear_entry` (2228), `test_kill_switch_returns_none_without_touching_cache` (1948). `TestCacheFreshnessBridgeScaling` (2307).

**Status.** LIVE and winning (exact-repeat 5k **40s→3.0s, 13×**, `tasks/todo.md:1075`), but **T3–T5 trailer tests remain PENDING** under FOLLOW-UPS before it becomes the permanent default-on (`tasks/todo.md:1089-1090`).

## 5. Hybrid-model caches (mamba/attention & sliding-window mixes)

**Per-layer type map.** `ModelCacheConfig.from_cache_list` detects each layer's `CacheType`; `is_hybrid` = more than one type seen (`hybrid_cache.py:63-128`). Types: `KVCACHE, ROTATING_KVCACHE, BATCH_KVCACHE, BATCH_ROTATING_KVCACHE, ARRAYS_CACHE, QUANTIZED_KVCACHE, CACHE_LIST, POOLING_CACHE, BATCH_POOLING_CACHE` (`type_handlers.py:29-40`). Only the KVCache family is **sliceable**; `ArraysCache`/`RotatingKVCache` are non-sliceable and need whole-state snapshots (`hybrid_cache.py:187-195`).

**Boundary snapshots.** `BoundarySnapshotSSDStore` writes non-sliceable layer state (e.g. GatedDeltaNet/ArraysCache recurrent state) to SSD at block boundaries **during prefill** to free GPU memory, and reloads it per-block at completion for the final store (`boundary_snapshot_store.py:2-12,57-62`). Same async pattern as the SSD cache: serialize on the inference thread (Metal-safe), buffer in `_pending_writes` for instant read-back, background flush (`boundary_snapshot_store.py:9-12`). Files are ephemeral under `base_dir/_boundary_snapshots/<session>/<request>/`, reset each server lifecycle (`boundary_snapshot_store.py:46-54,68`).

**Enlarged blocks (what/why).** For ArraysCache-only hybrids (GatedDeltaNet, **no** RotatingKVCache), `Scheduler._enlarge_block_size_for_arrays_cache` raises `block_size` to `_ARRAYS_CACHE_BLOCK_SIZE = 2048` — fewer boundary-snapshot stops during prefill while still storing valid per-block recurrent state. Skipped if RotatingKVCache was detected (block already aligned to its window) or the user set a larger size (`scheduler.py:1531-1533,2755,2757-2804`).

**Restore-fallback bug class (RotatingKVCache / MiniMax-M3 sliding window).** Stock `RotatingKVCache.size()` returns `min(offset, max_size)`, ignoring the actual buffer length. For SSD-restored caches with `keys.shape[2] < max_size` and `offset >= max_size`, mlx-lm's `merge()` overshoots → shape mismatch, or (if omlx zero-pads) zeros leak into softmax → **infinite loops / empty output** (issues #934/#903/#900). Fix: `PrefillReadyRotatingKVCache.size()` clamps to `keys.shape[2]` (`_rotating_subclass.py:1-46`). Partial/placeholder non-sliceable state on restore is handled by `_find_walk_back_truncation_point` (§2, `prefix_cache.py:1410`).

**Related live bug (ledger).** Compiled decode kept per-bucket KV buffers and reseeded only on bucket **growth**; a shorter follow-up request reused a bucket and decoded on the **prior** request's KV (cross-request contamination). NIAH missed it (unique long prompts always grow fresh buckets); only a temp0 token-identity gate on a *reused* bucket caught it (`tasks/lessons.md:90-96`). The "M3 sparse silently disabled in live serving" scare was a **layer-0 debug artifact** and was refuted — fresh requests do get `MiniMaxM3BatchKVCache` (`tasks/todo.md:350-353,382-384`).

**V2 legacy tuple format (verified in code).** The **hyphenated** `V2-legacy` string is absent, but the concept IS in code: a `(keys, values)` 2-tuple is the **V2 legacy** layout, promoted to V3 on read (`paged_ssd_cache.py:1771,1874,1904` — all three say "V2 legacy"). `load_block` returns per-layer `(keys, values)` tuples, or `List[(keys,values)]` for `CacheList` layers (`paged_ssd_cache.py:2340-2342`); the store side passes `cache_data` as a list of dicts with `state`/`meta_state` keys (`scheduler.py:2221-2253`) — so a raw 2-tuple assumption on the store side would drop the newer dict state.

## 6. Invalidation doctrine — the cache key has no weight hash

The block key is a chain hash of **tokens + parent_hash + extra_keys + model_name** only (`compute_block_hash`, `paged_cache.py:78-103`). There is **no weight-content hash**. Isolation is two-layered: `model_name` folded into every hash, **plus** a per-model on-disk subdir `cache_dir/{model_name}/` (`factory.py:120-122`).

**Consequence (silent poisoning).** Edit or re-quantize the weights but keep the **same** model name → identical prompts produce identical block hashes → the SSD entries (which survive restart, §3) serve KV computed from the **old** weights. Nothing detects this.

**LAW: new weights = NEW MODEL NAME.** Instituted when the DQ8-baked Ultra checkpoint replaced the old one — the serving name was deliberately changed to `Nemotron-3-Ultra-oQNVFP4-dq8`, quote: *"new name deliberate: avoids stale SSD-cache poisoning"* (`tasks/todo.md:1102-1103`). Cross-architecture blocks are inherently incompatible (GLM SSD blocks unusable by Kimi, `tasks/todo.md:261`); `_is_compatible_block` (scan, `paged_ssd_cache.py:1369`) and `_forget_incompatible_ssd_block` (lookup, `prefix_cache.py:204`) enforce layout compatibility but **cannot** catch same-name-different-weights. See `ops-runbook.md` (weight-swap procedure) and `laws.md`.

## 7. Timing implications for benchmarking

- **Fresh-nonce discipline.** A repeat prompt HITS the prefix cache (RAM, or SSD which survives restart). To measure a true cold prefill, prepend a unique nonce so the very first 64-token block hash differs. To measure warm behavior, send the *exact* same prompt on purpose.
- **Continuation stores.** A request's own output extends its cache; a follow-up sharing that prefix reuses blocks via `increment_ref` (no copy, §2). If the prior store is still in flight, a rapid repeat may **defer admission 4–30s** (freshness bridge, §4) rather than re-prefill — do not mistake this pause for slow prefill.
- **Decode speed is restore-independent:** decode after a 4s warm restore == decode after a 61s cold prefill (`tasks/todo.md:513-514`).
- **Restart race:** an old server draining on :8000 can serve stale code while a new server scans the SSD dir — several "no change" benchmark results were served by old code (`tasks/todo.md:380-381`).
- **Cross-request contamination check** (Law 7, runnable): after any capacity/shape-keyed cache change, seed a bucket with request A then send a SHORTER B that must retrieve ITS OWN needles — `scripts/archive/ov_reuse_niah.py` (live, retrieval-based → nondeterminism-immune) + `tasks/compile_spikes/spike_phaseb_reuse.py` (offline reseed-on-vs-off negative control). NIAH alone misses it (unique long prompts always grow fresh buckets).

| Scenario | Cold / uncached | Warm / restored | Cite |
|---|---|---|---|
| Exact-repeat 5k prompt (Ultra 550B, SF-1) | 40s | **3.0s (13×)** | `tasks/todo.md:1057-1058,1075` |
| Same 16k prompt twice (M3) | ~213s prefill | **4.0s** | `tasks/todo.md:503,509` |
| GLM restored prefix @14.5k / @57.7k | — | **2.9s / 5.1s** | `tasks/todo.md:516-517` |
| Ultra 550B first serve (incl 333.55GB load) | 57s | — | `tasks/todo.md:1049` |
| Fresh prefill @103k | 320–370s | — | `tasks/todo.md:656-657` |
| 259-block prefix store @9.5k | — | 0.25s store | `tasks/todo.md:483-484` |
| First request after load (Metal JIT warm) | ~50s penalty | — | `tasks/todo.md:311` |

**Cross-references.** `profiling.md` (how to bench prefill/decode & read `_phase_timer`); `ops-runbook.md` (weight-swap / model-name procedure); `laws.md` (new-weights-new-name law); `omlx.md` (scheduler + store worker internals); `gotchas.md` (restart race, fp16-cache surprise); `models/*.md` (per-model cache types: MiniMax-M3, Nemotron-H).

## 2026-07-08 state — int8-native tier, 512 blocks, restore truth
- **Block size default is now 512 tokens** (`SchedulerConfig.paged_cache_block_size`, scheduler.py:1434;
  was 256 — and the "64-token blocks" claim earlier in this doc was always stale). Changing it orphans
  existing entries (clean miss, no error).
- **int8 MLA sessions persist NATIVE int8 triples** per block (4-element state, class-name+count format
  dispatch in `Int8MLAKVCacheHandler`); legacy fp16 blocks restore via streamed per-layer requant
  (mx.eval + fp16 drop + pool trim every 12 layers — bounds the 256k restore transient to ~1 layer).
  Cross-mode restore works both directions. Grep: `int8 MLA-KV: restore format=`.
- **Restore is FAST and always was:** 130k = 1.3-1.8s total (7ms/block load, 0.04s build) — permanent
  `[restore-profile]` INFO line in `reconstruct_cache` proves it per restore. The "83s restore" was
  model-load conflation (Law 13). Restore-batching campaign CANCELLED.
- **Metal-pool watermark over-count:** after heavy prefill the reusable pool inflates footprint readings
  by ~35GB. Preflight credits `mx.get_cache_memory()` before rejecting (scheduler safety guard); the
  adaptive throttle reclaims-then-remeasures before shrinking. The enforcer's hard abort keeps raw
  footprint (the pool is real wired memory).
- **Predictive preflight OFF by default** (`MemorySettings.preflight_guard`); mid-prefill real-usage
  guard + enforcer remain the protection. MLA/DSA-aware estimator arming exists for when it's enabled
  (58KB/tok int8 GLM vs the 333KB/tok dense fiction that rejected 256k for weeks).
- **Shared-prefix benchmark design** (EXP-085): N questions over one haystack = 1 prefill + N-1 cache
  hits (8-19s each at 103k). The pattern for any multi-question long-ctx eval.
