# SPDX-License-Identifier: Apache-2.0
"""LEVER #1 Phase B — cross-request KV contamination repro (bench-owner's catch).

Two back-to-back ENGAGING requests on the batch cache, sharing ONE persistent
CompiledDecoder: request A (longer) then request B (SHORTER, same cap bucket, so
NO growth → without a per-request reseed the bucket keeps A's KV) with DIFFERENT
content. B's compiled output must be TOKEN-IDENTICAL to B's eager output (i.e. B
decodes on B's KV, not A's). Fixed by reseeding self.state when the continuation
check trips (new request re-prefills → fresh key arrays + reset offset).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMLX_M3_SPARSE_MIN_K", "256")
from omlx.patches.mlx_vlm_minimax_m3_compat import apply_mlx_vlm_minimax_m3_compat_patch  # noqa: E402
apply_mlx_vlm_minimax_m3_compat_patch()
import mlx.core as mx  # noqa: E402
from mlx_vlm.models.minimax_m3_vl.config import TextConfig  # noqa: E402
from mlx_vlm.models.minimax_m3_vl.language import LanguageModel  # noqa: E402

mx.random.seed(0)
cfg = TextConfig(
    hidden_size=1024, intermediate_size=512, dense_intermediate_size=1024,
    shared_intermediate_size=256, num_attention_heads=8, num_key_value_heads=4,
    head_dim=128, num_hidden_layers=5, num_local_experts=8, num_experts_per_tok=4,
    n_shared_experts=1, vocab_size=512, tie_word_embeddings=True, rope_theta=10000.0,
    max_position_embeddings=8192, use_qk_norm=True, scoring_func="sigmoid",
    use_routing_bias=True,
)
m = LanguageModel(cfg)
mx.eval(m.parameters())


def warm_batch(prompt_len, seed):
    mx.random.seed(seed)
    c = m.make_cache()
    m(mx.random.randint(0, cfg.vocab_size, (1, prompt_len)), cache=c, skip_logits=True)
    mx.eval([x.state for x in c if x.state is not None])
    return [type(x).merge([x]) for x in c]   # scheduler-built batch cache


def decode(cache, toks, first):
    am, lgs = [], []
    cur = mx.array([[first]])
    for t in toks:
        lg = m(cur, cache=cache).logits[:, -1, :]
        mx.eval(lg)
        am.append(int(mx.argmax(lg, -1).item()))
        lgs.append(lg)
        cur = mx.array([[t]])
    return am, lgs


def _dlogit(a, b):
    return max(float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))))
               for x, y in zip(a, b))


def run_leg(reseed_on, toks_b, a_toks):
    """A (2100) then B (2050, shorter, reuses bucket) on a persistent decoder.
    reseed_on=False monkeypatches _continuation->True to DISABLE reseed-per-
    request (reproduces a1a037ce: mask fix present, no reseed) -> must leak."""
    os.environ["OMLX_M3_COMPILE"] = "1"
    if hasattr(m.model, "_compiled_decoder"):
        del m.model._compiled_decoder
    A = warm_batch(2100, seed=111)                 # cap bucket 2304
    decode(A, a_toks, first=99)                     # builds+seeds A into the bucket
    dec = m.model._compiled_decoder
    if not reseed_on:
        dec._continuation = lambda cache: True      # force "same sequence" -> no reseed
    r_a = dec.rebuilds
    B = warm_batch(2050, seed=222)                  # SHORTER -> reuses cap 2304 (no growth)
    am, lg = decode(B, toks_b, first=7)
    return am, lg, (dec.rebuilds > r_a)


if __name__ == "__main__":
    STEPS = 24
    toks_b = [int(x) for x in mx.random.randint(0, cfg.vocab_size, (STEPS,)).tolist()]
    a_toks = [int(x) for x in mx.random.randint(0, cfg.vocab_size, (STEPS,)).tolist()]

    # --- EAGER reference for request B (fresh warm, clean) ---
    os.environ.pop("OMLX_M3_COMPILE", None)
    if hasattr(m.model, "_compiled_decoder"):
        del m.model._compiled_decoder
    am_b_eager, lg_b_eager = decode(warm_batch(2050, seed=222), toks_b, first=7)

    # --- LEG 1: reseed ON (9959bdc0) -> must be bit-identical to eager-B ---
    am_on, lg_on, reseeded_on = run_leg(True, toks_b, a_toks)
    agree_on = sum(1 for a, b in zip(am_b_eager, am_on) if a == b)
    d_on = _dlogit(lg_b_eager, lg_on)

    # --- LEG 2: reseed OFF (== a1a037ce: mask fix present, NO reseed) -> must LEAK ---
    am_off, lg_off, reseeded_off = run_leg(False, toks_b, a_toks)
    agree_off = sum(1 for a, b in zip(am_b_eager, am_off) if a == b)
    d_off = _dlogit(lg_b_eager, lg_off)

    print(f"LEG reseed-ON  (9959bdc0): B reseeded={reseeded_on}  argmax {agree_on}/{STEPS}  max|Δlogit|={d_on:.3e}")
    print(f"LEG reseed-OFF (a1a037ce): B reseeded={reseeded_off}  argmax {agree_off}/{STEPS}  max|Δlogit|={d_off:.3e}")
    clean = reseeded_on and agree_on == STEPS and d_on == 0.0
    leaks = (not reseeded_off) and (agree_off < STEPS or d_off > 0.0)
    print("=" * 62)
    print(f"reseed-ON bit-identical to eager-B: {clean}")
    print(f"reseed-OFF diverges (test is not a false-negative): {leaks}")
    print(f"CROSS-REQUEST CONTAMINATION TEST: {'PASS' if (clean and leaks) else 'FAIL'}")
