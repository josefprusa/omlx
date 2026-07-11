# Scheduler and Cache Lifecycle

## Stable Boundaries

- `omlx/scheduler.py` class `Scheduler` admits, steps, aborts, and finishes requests.
- `omlx/cache/interface.py` class `CacheManager` defines the cache-manager contract.
- `omlx/cache/factory.py` class `CacheFactory` selects cache implementations.
- `omlx/cache/prefix_cache.py` class `BlockAwarePrefixCache` owns reusable prefix blocks.
- `omlx/cache/paged_ssd_cache.py` class `PagedSSDCacheManager` owns persistent paged blocks and its index.

## Invariant

Visibility, ownership, persistence, rollback, and freeing are one lifecycle. Publishing reusable state before persistence completes requires a tested retraction path for every failure exit.

Retraction must remove both the chain-hash lookup and prefix-index entry so no
new reader can discover unpersisted data. It releases only the publisher's
unpersisted block references; a reader that already acquired a reference keeps
its ownership and must fail reconstruction by truncating to the last persisted
prefix. Clearing request bookkeeping alone is not rollback.

Transient prefill protection may clamp poisoned tracker-derived estimates
against a static model estimate, but the static term, fixed abort margin, and
final safety multiplier remain independent. A zero clamp setting must reproduce
the unclamped estimator exactly.

## Provenance

Updated from the current scheduler and cache interfaces on 2026-07-11 at upstream `d5fcb22a`, then validated by core reconstruction commit `38093a76` with reader-before-persist, selective rollback, submission-failure, kill-switch, and clamp-zero tests.

Decay condition: recheck when `CacheManager`, `Scheduler.step`, prefix publication, or SSD write completion contracts change.
