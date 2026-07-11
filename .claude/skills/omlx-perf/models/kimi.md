> Verified 2026-07-05 · Mac Studio M3 Ultra 512GB (819GB/s) · MLX 0.31.2 · omlx 0.4.5.dev1 · branch glm5.2-native-kernels-v0.4.5 (uncommitted tree). Measured here, not universal — re-verify after MLX/omlx upgrades.

# Kimi-K2.7-Code (kimi_k25 VLM, 3.6bpw) — THIN dossier

**Thinnest of the five** — downloaded and given the cheap decode-opt playbook, but never fully benched or accuracy-tested. A 1T-param DeepSeek-V3-arch MoE VLM with **MLA but NO DSA and NO MTP** (unlike GLM-5.2), so long-context decode reads the *full* latent cache and sags. Gaps are marked explicitly.

## 1. Identity
- **Serving artifact:** `~/.omlx/models/avlp12/Kimi-K2.7-Code-Alis-MLX-Dynamic-3.6bpw-VLM` (symlink → `$OMLX_COLD_STORAGE/omlx-models/…`). **466 GB** on disk (434 GiB download; `tasks/todo.md` "Kimi-K2.7-Code VLM").
- `model_type=kimi_k25` (VLM wrapper; `text_config.model_type=kimi_k2`, text delegates to mlx-lm **deepseek_v3**). vocab 163840, **61 layers**, native context **262 144 (256k)** (`config.json text_config`).
- **NOT in `~/.omlx/model_settings.json`.** The settings row `Kimi-K2.5-mlx-DQ3_K_M-q8` is a **different, older model** (K2.5, 30k ctx, TurboQuant 4-bit KV) — do not conflate. **GAP: K2.7-Code-VLM has no persisted settings row**; it serves on defaults + the manual wired-limit below.

## 2. Geometry (`tasks/todo.md` "Kimi-K2.7-Code VLM")
- 1T total / **32B active**. DeepSeek-V3 arch: 61 layers, **384 routed experts, top-8**, **MLA 512-latent**, **NO DSA, NO MTP**.
- MoonViT bf16 vision tower (VL). Card claims ~23 tok/s on this hardware.

## 3. Quant layout — per-tensor dtype split
Third-party `avlp12` MLX Dynamic **3.6 bpw** build. **GAP: no per-tensor dtype table captured** in the corpus (unlike GLM's `glm_quant_matrix.md`). Known only that experts dominate the 3.6 bpw weighted average and the container is MLX affine (deepseek_v3 loader). If a byte-ceiling is ever needed, derive it as for GLM (kv-cache.md) — but note the KV geometry below.
- **KV per token = 70 272 B** (61 layers × 576 × 2B, MLA latent; `memory_monitor.estimate_mla_kv_bytes_per_token`, fixed — see §9). 128k ctx ≈ **9.2 GB** KV. **No DSA** → decode attention reads the *full* latent cache (2.3 GB/tok @32k, 9 GB @128k) → long-ctx decode sags (`memory/omlx-glm52-decode-opts.md` 2026-07-03 Kimi entry).

## 4. Serving
- **REQUIRED before load (user must run):** `sudo sysctl iogpu.wired_limit_mb=518144` — peak 467 GB model > the default ~464 GB Metal cap (`tasks/todo.md`). Without it, load fails.
- Served via the omlx engine pool (GLM/others unload on demand). trust_remote_code + card sampling. deps: `tiktoken`+`blobfile`; `mlx_vlm 0.6.3` has `kimi_k25` native.
- **GAP: no confirmed launch line or settings entry** — treat golden env (OPS=4000/MB=4000, env-setup.md) as the fleet default; verify before relying.

## 5. Engagement grep set
- Decode-opt patch (`omlx/patches/dsv3_decode_opts.py`, kill `OMLX_DISABLE_DSV3_DECODE_OPT=1`): a patch-application log line confirms it engaged (wired in `model_loading` for `deepseek_v3`/`kimi_k2`/`kimi_k25`). It does two swaps: (1) `QuantizedMultiLinear` M=1 → `gather_qmm` (≤1 ulp, GLM-proven on embed_q shapes); (2) DeepSeek-V3 MoE weighted-sum as one gemv.
- **GAP: no census/counter line** as rich as M3's `[M3CENSUS]` or Ultra's `[ULTRA-DQ8]` — engagement evidence is the single patch log line only.
- Kimi builds its **own** SSD prefix-cache blocks (GLM's are incompatible — expected).

## 6. Measured speed (batch-1, live server A/B, `tasks/todo.md`)
| context | decode tok/s |
|---|---|
| short | OFF 24.6 → **ON 25.4** (+3.3%; card ref ~23 → we serve ~10% above card) |
| 16k | **21.2** (no-DSA bandwidth curve) |
| 64k / 128k | **CANCELLED** — fresh prefill in the throttled regime crawls (~23 tok/s at 24k depth vs GLM's ~180) |

Output coherent both A/B ways; wording diverges slightly (≤1-ulp embed_q swap flips rare near-ties — same numerics class as the GLM opts). **GAP: long-context decode curve (64k/128k) unmeasured; prefill tok/s unmeasured; VLM image/video path unmeasured.**

## 7. Quality
**GAP — no accuracy benches run** (no gsm8k/mmlu/arc/NIAH for this build). This is the least-validated model in the fleet.

## 8. Levers — LIVE vs DEAD vs PARKED
- **LIVE:** `dsv3_decode_opts` (+3.3% short, lossless-class).
- **PARKED / unmeasured:** fused `gate_up` port (concat gate+up at sanitize → one gather_qmm; the GLM-fork trick, portable, **medium effort, unmeasured win**); the router is already compiled but SwitchGLU gate/up are unfused.
- **The long-ctx unlock (not built):** **int8 MLA-KV** would halve the full-latent reads, but MLX `qsdpa` is unfused/~2× slower (measured on GLM) → the real fix is an **upstream fused int8 `sdpa_vector`**. ThunderMittens has a complete fp8 MLA-KV reference that would de-risk this (`tasks/todo.md` "TM CAPABILITY MAP"). See `future-campaigns.md` #4 and kv-cache.md.

## 9. Known bugs / watch-items
- **FIXED — KV overestimate:** `memory_monitor.estimate_mla_kv_bytes_per_token` was falling back to the expanded-MHA formula for VLM configs → **~23× KV overestimate** → the prefill guard **rejected all Kimi prompts >~4k tokens**. Fixed to descend into `text_config` and count plain per-layer `KVCache` as the MLA cache (now 70 272 B/token). Long context on Kimi was impossible before this (`tasks/todo.md`).
- **OPEN — phantom-transient:** `estimate_chunk_transient_bytes` appears to reserve for **materialized** attention scores (~4.4 MB/chunk-token @14k depth) while MLX SDPA is flash-style and never materializes them → ~30-40 GB phantom reservation on a box with ~50 GB headroom → chunks shrink → **fresh prefill 2-8× slower than physics requires**. This is why 64k/128k benches were cancelled. Repro/plan: measure `mx.get_peak_memory` around a chunk at depth vs the estimate; tighten the estimator for fused-SDPA models (`tasks/todo.md`, highest Kimi priority).

## 10. Conversion provenance
Third-party `avlp12` MLX Dynamic 3.6bpw VLM (not built here). Text path = mlx-lm `deepseek_v3` loader; VLM = mlx_vlm `kimi_k25`. Custom `vision_3d.py` overlay from the model repo **not yet integrated**. No omlx-side conversion; the only omlx code is the `dsv3_decode_opts` patch. Cross-ref conversion.md for the DeepSeek-V3 family container conventions (shared with the deepseek_v4 patches).
