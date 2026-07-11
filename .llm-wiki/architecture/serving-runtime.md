# Serving Runtime

## Stable Boundaries

- `omlx/server.py` assembles the HTTP application and routes.
- `omlx/engine_pool.py` class `EnginePool` owns loaded engine instances and admission.
- `omlx/engine_core.py` classes `EngineCore` and `AsyncEngineCore` bridge request APIs to generation engines.
- `omlx/scheduler.py` class `Scheduler` owns continuous-batching and cache coordination.
- Engine specializations live under `omlx/engine/`.

## Rule

Serving changes must be traced from the API entry point through engine selection and scheduler behavior. A test at only one layer does not prove the end-to-end path.

## Provenance

Updated from `omlx/server.py`, `omlx/engine_pool.py`, `omlx/engine_core.py`, and `omlx/scheduler.py` on 2026-07-11 at upstream `d5fcb22a`.

Decay condition: recheck when request routing, engine ownership, or scheduler construction moves between modules.
