# Scheduler and Cache Lifecycle

## Stable Boundaries

- `omlx/scheduler.py` class `Scheduler` admits, steps, aborts, and finishes requests.
- `omlx/cache/interface.py` class `CacheManager` defines the cache-manager contract.
- `omlx/cache/factory.py` class `CacheFactory` selects cache implementations.
- `omlx/cache/prefix_cache.py` class `BlockAwarePrefixCache` owns reusable prefix blocks.
- `omlx/cache/paged_ssd_cache.py` class `PagedSSDCacheManager` owns persistent paged blocks and its index.

## Invariant

Visibility, ownership, persistence, rollback, and freeing are one lifecycle. Publishing reusable state before persistence completes requires a tested retraction path for every failure exit.

## Provenance

Updated from the current scheduler and cache interfaces on 2026-07-11 at upstream `d5fcb22a`.

Decay condition: recheck when `CacheManager`, `Scheduler.step`, prefix publication, or SSD write completion contracts change.
