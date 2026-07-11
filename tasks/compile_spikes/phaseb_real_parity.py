# SPDX-License-Identifier: Apache-2.0
"""LEVER #1 Phase B — REAL fs5 parity gate (STANDALONE, server stopped).

Gate (per the lead's box): step-1 max|Δlogit| eager-vs-compiled + >=200-token
temp0 argmax-agreement, on the real MiniMax-M3 oQNVFP4 fs5 model. fs5 IS the
nvfp4+ts model, so this exercises the one op path unexercised offline (nvfp4
gather_qmm + weight_scale_2 / swiglu_oai_ts fused activation).

Method: eager greedy-decodes N steps recording its token stream + per-step
argmax/logits; the compiled path is then fed the SAME tokens from a deep-cloned
warm cache and compared per step. Teacher-forcing isolates the compiled math
from autoregressive drift. Loader/clone scaffolding mirrors the proven
verify_parity.py; warm uses synthetic token blocks (no tokenizer needed).

RUN ONLY with the server stopped (mission ops contract). WARM_LEN > 4096 so the
compiled path engages from decode step 1.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from omlx.utils.model_loading import maybe_apply_pre_load_patches

MODEL_NAME = "MiniMax-M3-oQNVFP4-fs5"
MODEL_PATH = Path(os.environ["OMLX_MINIMAX_FS5_MODEL"])
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 220
WARM_LEN = int(os.environ.get("OMLX_M3_PARITY_WARM_LEN", "5008"))  # > 4096 crossover


def _arrays(tree: Any):
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, (tuple, list)):
        for item in tree:
            yield from _arrays(item)
    elif isinstance(tree, dict):
        for item in tree.values():
            yield from _arrays(item)


def _clone_tree(tree: Any):
    if isinstance(tree, mx.array):
        return mx.array(tree)                    # deep copy — no aliasing
    if isinstance(tree, tuple):
        return tuple(_clone_tree(x) for x in tree)
    if isinstance(tree, list):
        return [_clone_tree(x) for x in tree]
    if isinstance(tree, dict):
        return {k: _clone_tree(v) for k, v in tree.items()}
    return tree


def _clone_cache(cache):
    cloned = []
    for entry in cache:
        ne = entry.__class__()
        ne.state = _clone_tree(entry.state)
        meta = getattr(entry, "meta_state", None)
        if meta is not None:
            ne.meta_state = meta
        cloned.append(ne)
    arrays = list(_arrays([e.state for e in cloned]))
    if arrays:
        mx.eval(*arrays)
    return cloned


def _language_model(model):
    return getattr(model, "language_model", model)


def _vocab_size(lm) -> int:
    args = getattr(lm, "args", None)
    if args is not None and hasattr(args, "vocab_size"):
        return int(args.vocab_size)
    w = getattr(getattr(getattr(lm, "model", None), "embed_tokens", None), "weight", None)
    if w is not None:
        return int(w.shape[0])
    raise RuntimeError("vocab size?")


def _token_block(vocab_size: int, start: int, length: int):
    low = 128 if vocab_size > 512 else 1
    span = max(vocab_size - low, 1)
    toks = (mx.arange(start, start + length, dtype=mx.int32) % span) + low
    return toks.reshape(1, length)


def _step_logits(lm, tokens, cache):
    out = lm(tokens, cache=cache, skip_logits=False)
    lg = out.logits[:, -1, :]
    mx.eval(lg)
    return lg


def main() -> int:
    print("Loading MiniMax-M3 fs5 via VLM pre-load patches (server must be stopped).")
    maybe_apply_pre_load_patches(MODEL_NAME, for_vlm=True)
    if not (Path(MODEL_NAME) / "config.json").exists():
        maybe_apply_pre_load_patches(str(MODEL_PATH), for_vlm=True)
    import mlx_vlm.utils
    model = mlx_vlm.utils.load_model(MODEL_PATH, trust_remote_code=True)
    mx.set_wired_limit(506 * 1024 ** 3)
    lm = _language_model(model)
    mm = lm.model if hasattr(lm, "model") else lm
    vocab = _vocab_size(lm)
    print(f"loaded; lm={type(lm).__name__} vocab={vocab}")

    # warm cache (eager prefill, synthetic tokens)
    os.environ.pop("OMLX_M3_COMPILE", None)
    warm_cache = lm.make_cache()
    lm(_token_block(vocab, 0, WARM_LEN), cache=warm_cache, skip_logits=True)
    mx.eval(*list(_arrays([c.state for c in warm_cache])))
    print(f"warm offset={warm_cache[0].offset} (crossover 4096; compiled engages step 1)")

    first_in = _token_block(vocab, WARM_LEN, 1)

    # --- EAGER greedy: record token stream + per-step argmax/logits ---
    os.environ.pop("OMLX_M3_COMPILE", None)
    if hasattr(mm, "_compiled_decoder"):
        del mm._compiled_decoder
    ce = _clone_cache(warm_cache)
    stream, am_e, lg_e, te = [], [], [], []
    cur = first_in
    for _ in range(N_STEPS):
        t0 = time.perf_counter()
        lg = _step_logits(lm, cur, ce)
        a = int(mx.argmax(lg, axis=-1).item())
        te.append((time.perf_counter() - t0) * 1e3)
        am_e.append(a); lg_e.append(lg); stream.append(a)
        cur = mx.array([[a]])

    # --- COMPILED: teacher-forced with EAGER's exact input stream ---
    os.environ["OMLX_M3_COMPILE"] = "1"
    if hasattr(mm, "_compiled_decoder"):
        del mm._compiled_decoder
    cc = _clone_cache(warm_cache)
    am_c, lg_c, tc = [], [], []
    cur = first_in
    for a_e in stream:
        t0 = time.perf_counter()
        lg = _step_logits(lm, cur, cc)
        _amc = int(mx.argmax(lg, axis=-1).item())
        tc.append((time.perf_counter() - t0) * 1e3)
        am_c.append(_amc)
        lg_c.append(lg)
        cur = mx.array([[a_e]])

    dec = getattr(mm, "_compiled_decoder", None)
    served = dec.compiled_calls if dec else 0
    rebuilds = dec.rebuilds if dec else 0
    step1_err = float(mx.max(mx.abs(lg_e[0].astype(mx.float32) - lg_c[0].astype(mx.float32))))
    maxerr = max(
        float(mx.max(mx.abs(le.astype(mx.float32) - lc.astype(mx.float32))))
        for le, lc in zip(lg_e, lg_c)
    )
    agree = sum(1 for a, b in zip(am_e, am_c) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(am_e, am_c)) if a != b), None)

    def _median(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    # steady-state = skip the first 10 steps (eager lazy warmup / compiled trace+
    # bucket build + one growth land here); median is robust to the rest.
    steady = slice(10, None)
    e_med = _median(te[steady]) if len(te) > 10 else _median(te)
    c_med = _median(tc[steady]) if len(tc) > 10 else _median(tc)
    c_min = min(tc[steady]) if len(tc) > 10 else min(tc)

    print("=" * 64)
    print(f"compiled steps served: {served}/{N_STEPS}  bucket (re)builds: {rebuilds}")
    print(f"STEP-1 max|Δlogit| (identical inputs): {step1_err:.4e}")
    print(f"all-step max|Δlogit|:                  {maxerr:.4e}")
    print(f"argmax agreement: {agree}/{N_STEPS}   first divergence @ step: {first_div}")
    print("-" * 64)
    print(f"STANDALONE step-time (median, steady, single-stream, no MTP):")
    print(f"  eager    {e_med:.2f} ms/tok")
    print(f"  compiled {c_med:.2f} ms/tok   (min {c_min:.2f})")
    print(f"  speedup  {e_med / c_med:.2f}x   (Δ {e_med - c_med:.2f} ms/tok)")
    ok = served >= int(N_STEPS * 0.95) and agree == N_STEPS
    print(f"PHASE-B REAL PARITY GATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
