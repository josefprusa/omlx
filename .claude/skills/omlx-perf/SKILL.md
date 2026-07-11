---
name: omlx-perf
description: Tested performance-engineering knowledge for omlx on the M3 Ultra 512GB box — laws, profiling recipes, MLX/Metal facts, KV-cache mechanics, NVFP4 conversion pipeline, experiment registry with verdicts, gotchas, preflight, per-model dossiers. Use when benching, converting, serving, or diagnosing omlx/MLX performance on this box.
---

> Legacy campaign snapshot from 2026-07-09: Mac Studio M3 Ultra 512GB, MLX 0.31.2, omlx 0.4.5.dev1, archived at `3f2edd60`. Current upstream is MLX 0.32.0 at `d5fcb22a`; reverify every claim before reuse.

# omlx-perf — performance engineering on the 512GB M3 Ultra

**Migration rule.** Read `.llm-wiki/index.md` first, then only the relevant durable pages. This skill preserves legacy operational detail and negative results; current source and tests are authoritative. Port a learning to the wiki only after current-code verification, evidence, human approval, provenance, a decay condition, and wiki lint.

**When to use.** You are benching, converting, serving, or diagnosing decode/prefill speed or quality
on THIS box (omlx + MLX on the Mac Studio M3 Ultra 512GB). Every number here was measured in-session —
a starting point, not gospel; re-verify after any MLX/omlx bump.

**Governance (non-negotiable).** Work on the branch selected by `.llm-wiki/decisions/branch-and-sync-policy.md`; keep custom commits off `main`. Port 8000 and tmux `omlx` are the user's live test bed: read logs, do not drive it without explicit approval. The user runs `sudo`. Raw campaign ledgers live in `tasks/*.md`; this skill is evidence to reverify, not current truth.

## The 19 iron laws (one line each; full proofs + incidents in `laws.md`)
1. A gated fast path needs LIVE engagement evidence before any benchmark number is believed.
2. Isolated microbench wins take the in-stream discount — only ~50-60% survives; only serve legs count.
3. Bandwidth napkin math uses the per-tensor dtype split, NEVER a blended bits-per-weight.
4. Quant-first: serve tensors in source-precision containers; fix kernels, don't requantize vendor weights.
5. Composed-reality testing: test the fully patched/swapped model, not stock classes.
6. Never time an async GPU submit as one wall number (split host-build vs async_eval drain).
7. Single-request PASS ≠ multi-request correct (state reuse + concurrency artifacts).
8. Verify on-metal, not from tests: green units prove neither speed nor engagement.
9. Changed weights = NEW MODEL NAME (the SSD prefix cache survives restarts and poisons on name reuse).
10. Serve-window discipline: baseline first, one lever per restart, engagement grep before tok/s.
11. Installed ≠ engaged: verify the artifact is loaded in the LIVE process, not just present on disk.
12. Per-instance `__class__` swap only — never rebind a method at class scope in a multi-model process.
13. Never time a first request against a fresh server — model load hides inside (`/health` answers before the pool loads; the "83s restore" myth).
14. Nonce every temp-0 probe request — the server replays exact-duplicate requests instantly (0.0s walls).
15. T1/T256 subtraction is invalid across cache restores — continuous single-request GEN timing only (151 and 75 tok/s ghosts).
16. Temp-0 byte-identity gates hold only within ONE forward width on quantized models (qmv vs qmm round differently at ties).
17. HARNESS LAW: name the harness beside every ms/token number; never compare or borrow a number across harnesses.
18. Pipelined stability is its own gate — surviving eager decode proves nothing about async pipelined decode.
19. (extends 2) Eager-measured per-op overhead pools are already partially pipeline-hidden — size fusion wins against the PIPELINED baseline.

## Before ANY serve/bench
Run `scripts/preflight.py` (golden env `MLX_MAX_OPS_PER_BUFFER=4000 MLX_MAX_MB_PER_BUFFER=4000`, wired
sysctl, loaded MLX/nanobind version, `model_settings.json` force_sampling, no-USB, quiet box), THEN the
live engagement greps (`ops-runbook.md §3`: `[ULTRA-DQ8] expected==actual`, `sorted_routes=N/N`,
`[M3CENSUS] fused_hit`) AFTER a warm request. A wrong tok/s from a silently de-engaged fast path is the
single most repeated failure here (Law 1). Use `.venv/bin/python` — never bare `uv run` (Law 11).

## File map (open the one that matches your task)
| File | What's inside | Open when |
|---|---|---|
| `laws.md` | 19 iron laws + the incident that proved each | starting anything; justifying a method |
| `playbooks.md` | 3 decision-trees: speed campaign / conversion / regression diagnosis | running a campaign end-to-end |
| `profiling.md` | `/usr/bin/sample` split, stream-free T1/T256, engagement counters, banked methods | measuring decode/prefill honestly |
| `mlx.md` | MLX 0.31.2 facts: buffer env, MAX_ACTIVE_TASKS, quant modes, gather_qmm, mx.compile | writing/timing MLX ops |
| `metal.md` | writing/judging `mx.fast.metal_kernel`, dispatch floors, the mega-kernel case | deciding on / writing a Metal kernel |
| `kv-cache.md` | RAM+SSD prefix cache, SF-1 early publish, hybrid caches, invalidation doctrine | anything cache / TTFT / long-ctx |
| `omlx.md` | request lifecycle, scheduler, engine pool, patch house-rules, decode-burst, batch toggles | touching the serve path / a patch |
| `conversion.md` | the oQNVFP4 source→serve pipeline + GLM-5.2-NVFP4 deltas | converting a checkpoint |
| `env-setup.md` | zero-to-serving, uv/venv traps, ext rebuild (nanobind ABI), sysctl | first-time setup / rebuild |
| `ops-runbook.md` | restart/verify/bench/swap procedure; memory budget math; crew pipeline | running a serve window |
| `gotchas.md` | the trap museum — grep the SYMPTOM string | you hit a weird error/number |
| `preflight.md` | prose mirror of `scripts/preflight.py` (8 checks) | before serving |
| `dead-levers.md` | verdict graveyard — check before re-testing anything | tempted to try a lever |
| `future-campaigns.md` | 5 mapped-but-unrun workstreams (GLM-NVFP4, batch≥2, mega-kernel, mxfp8-KV, SpecPrefill bug) | "what's next on the box" |
| `experiments/index.md` | 64-row EXP registry (one-liner + verdict); `pre-ultra.md` / `ultra-day.md` = detail | looking up a prior result |
| `models/*.md` | per-model dossiers: glm52, kimi, minimax-m3, nemotron-super, nemotron-ultra, nemotron-puzzle | serving/benching a specific model |
| `scripts/` | supported probes (t1t256, cadence, kernel_parity, acc_bench_serial, preflight, restart) | benching / reproducing |

## Update discipline (you are the next writer)
After any campaign: add `EXP-NNN` rows under `experiments/` (+ its one-liner in `index.md`), update the
model dossier, and record dead/parked levers in `dead-levers.md`. Cite `file:symbol` for code,
`file §heading` for ledgers, and INLINE the load-bearing number (don't just point at an untracked
ledger). Keep every file's staleness header (line 1) current. **The bar: a fresh Sonnet 5 must be able
to act on your addition without asking a question.** Budgets: topics ≤300 lines, `gotchas.md` ≤350,
`experiments/pre-ultra.md` ≤400, dossiers ≤150, this file ≤100.
