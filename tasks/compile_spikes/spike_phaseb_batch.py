# SPDX-License-Identifier: Apache-2.0
"""LEVER #1 Phase B — OFFLINE full-parity test against the LIVE batch cache.

Kills the batch-cache bug CLASS offline (lead directive): build the exact live
cache the way the SCHEDULER does — `type(c).merge([c])` per layer → sparse
becomes MiniMaxM3BatchKVCache (array offset, READ-ONLY .offset property), dense
becomes its batch form — then run the FULL parity smoke against it:
engagement + argmax + max|Δlogit| + writeback-survives (no corruption) + a
256-token cache-growth boundary.
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


def cl(t):
    if isinstance(t, mx.array):
        return mx.array(t)
    if isinstance(t, (list, tuple)):
        return type(t)(cl(x) for x in t)
    return t


def clone_state(cache):
    out = m.make_cache()
    for d, s in zip(out, cache):
        d.state = cl(s.state)
        if getattr(s, "meta_state", None) is not None:
            d.meta_state = s.meta_state
    mx.eval([c.state for c in out if c.state is not None])
    return out


def to_batch_via_merge(cache):
    """Batch each cache the way the scheduler does: type(c).merge([c])."""
    return [type(c).merge([c]) for c in cache]


if __name__ == "__main__":
    PROMPT, STEPS = 2100, 260   # 260 steps crosses one 256 growth boundary
    warm = m.make_cache()
    m(mx.random.randint(0, cfg.vocab_size, (1, PROMPT)), cache=warm, skip_logits=True)
    mx.eval([c.state for c in warm if c.state is not None])

    ce = to_batch_via_merge(clone_state(warm))
    cc = to_batch_via_merge(clone_state(warm))

    # show we built the real live condition
    print("cache classes:", [type(c).__name__ for c in cc])
    off_types = {type(c.offset).__name__ for c in cc}
    print(f"offset types across layers: {off_types}")
    ro = []
    for c in cc:
        try:
            c.offset = c.offset
            ro.append(False)
        except AttributeError:
            ro.append(True)
    print(f"read-only .offset per layer: {ro} (batch wrappers = True)")

    toks = [int(x) for x in mx.random.randint(0, cfg.vocab_size, (STEPS,)).tolist()]

    def decode(cache, compiled):
        if compiled:
            os.environ["OMLX_M3_COMPILE"] = "1"
        else:
            os.environ.pop("OMLX_M3_COMPILE", None)
        if hasattr(m.model, "_compiled_decoder"):
            del m.model._compiled_decoder
        am, lg = [], []
        cur = mx.array([[7]])
        for t in toks:
            out = m(cur, cache=cache).logits[:, -1, :]
            mx.eval(out)
            am.append(int(mx.argmax(out, -1).item()))
            lg.append(out)
            cur = mx.array([[t]])
        return am, lg

    err = None
    try:
        am_e, lg_e = decode(ce, False)
        am_c, lg_c = decode(cc, True)
    except Exception as e:
        err = e

    if err is not None:
        print(f"EXCEPTION during decode (writeback/corruption?): {type(err).__name__}: {err}")
        print("BATCH-CACHE FULL PARITY: FAIL")
    else:
        dec = getattr(m.model, "_compiled_decoder", None)
        served = dec.compiled_calls if dec else 0
        rebuilds = dec.rebuilds if dec else 0
        agree = sum(1 for a, b in zip(am_e, am_c) if a == b)
        maxerr = max(float(mx.max(mx.abs(le.astype(mx.float32) - lc.astype(mx.float32))))
                     for le, lc in zip(lg_e, lg_c))
        off_final = {type(c.offset).__name__ for c in cc}
        print(f"compiled steps served: {served}/{STEPS}  bucket (re)builds: {rebuilds} (want >=2: init+growth)")
        print(f"argmax agreement: {agree}/{STEPS}")
        print(f"max |Δlogit| eager-vs-compiled: {maxerr:.4e}")
        print(f"offset types still valid post-run: {off_final}")
        ok = served == STEPS and rebuilds >= 2 and agree == STEPS and maxerr < 1e-2
        print("=" * 60)
        print(f"BATCH-CACHE FULL PARITY (merge-built, +growth, +Δlogit): {'PASS' if ok else 'FAIL'}")
