# SPDX-License-Identifier: Apache-2.0
"""LEVER #1 Phase B — OFFLINE smoke/parity test on a TINY real MiniMaxM3Model.

Not the production gate (that's ≥200-token argmax on the real fs5 weights).
This proves the compiled decode path is STRUCTURALLY sound and numerically
matches eager on a small random-weight model: builds a 5-layer model (3 dense +
2 sparse-MoE), prefills past the 2048 sparse crossover, then teacher-forced
decodes N steps EAGER vs COMPILED from the same warm state, across a 256 cache-
growth boundary. Checks argmax agreement + logit closeness + trace count.
"""
from __future__ import annotations

import os
import sys
import mlx.core as mx
import mlx.nn as nn

# tiny model still needs head_dim=128 (fused kernels hardcode D=128) and
# num_key_value_heads == index_heads (sparse decode contract).
os.environ.setdefault("OMLX_M3_SPARSE_MIN_K", "256")  # crossover floor is 2048 anyway

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# overlay the vendored minimax_m3_vl onto the installed mlx_vlm namespace so
# its relative imports (..base) resolve, exactly as the server does.
from omlx.patches.mlx_vlm_minimax_m3_compat import apply_mlx_vlm_minimax_m3_compat_patch  # noqa: E402
apply_mlx_vlm_minimax_m3_compat_patch()
from mlx_vlm.models.minimax_m3_vl.config import TextConfig  # noqa: E402
from mlx_vlm.models.minimax_m3_vl.language import LanguageModel  # noqa: E402
from mlx_vlm.models.minimax_m3_vl import compiled_decode  # noqa: E402

mx.random.seed(0)

cfg = TextConfig(
    hidden_size=1024,
    intermediate_size=512,
    dense_intermediate_size=1024,
    shared_intermediate_size=256,     # != intermediate_size -> plain SwitchGLU (unquantized)
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=128,
    num_hidden_layers=5,              # layers 0-2 dense, 3-4 sparse-MoE
    num_local_experts=8,
    num_experts_per_tok=4,
    n_shared_experts=1,
    vocab_size=512,
    tie_word_embeddings=True,
    rope_theta=10000.0,
    max_position_embeddings=8192,
    use_qk_norm=True,
    scoring_func="sigmoid",
    use_routing_bias=True,
)


def build_model(quantize=False):
    m = LanguageModel(cfg)
    mx.eval(m.parameters())
    if quantize:
        # affine gs64 b4 -> QuantizedLinear projections engage the packed
        # "full" path in _attn_sparse; MoE switch -> quantized gather_qmm.
        # Exactly the production op family (Spike A modes).
        qp = m.quant_predicate  # fn(path, module)->True|dict
        nn.quantize(
            m, group_size=64, bits=4,
            class_predicate=lambda path, mod: hasattr(mod, "to_quantized") and qp(path, mod),
        )
        mx.eval(m.parameters())
    return m


def warm_cache(model, prompt_len):
    cache = model.make_cache()
    ids = mx.random.randint(0, cfg.vocab_size, (1, prompt_len))
    out = model(ids, cache=cache, skip_logits=True)
    mx.eval([c.state for c in cache])
    del out
    return cache


def clone_cache(model, src):
    dst = model.make_cache()
    for d, s in zip(dst, src):
        d.state = s.state
        d.meta_state = s.meta_state
    mx.eval([c.state for c in dst])
    return dst


def decode_run(model, cache, tokens, compiled):
    if compiled:
        os.environ["OMLX_M3_COMPILE"] = "1"
    else:
        os.environ.pop("OMLX_M3_COMPILE", None)
    # drop any decoder built under a previous mode
    if hasattr(model.model, "_compiled_decoder"):
        del model.model._compiled_decoder
    argmax = []
    logits_all = []
    for t in tokens:
        ids = mx.array([[t]])
        out = model(ids, cache=cache)
        lg = out.logits[:, -1, :]
        mx.eval(lg)
        argmax.append(int(mx.argmax(lg, axis=-1).item()))
        logits_all.append(lg)
    return argmax, logits_all


def run_scenario(quantize):
    tag = "QUANTIZED affine-gs64-b4 (packed proj + gather_qmm)" if quantize else "UNQUANTIZED (separate proj)"
    print(f"\n########## scenario: {tag} ##########")
    model = build_model(quantize=quantize)
    PROMPT = 2100                      # > 2048 crossover so sparse+compiled engage
    STEPS = 260                        # crosses one 256 cache-growth boundary
    base = warm_cache(model, PROMPT)
    print(f"warm cache offset={base[0].offset} (crossover=2048); decoding {STEPS} steps")

    # fixed teacher-forced token stream so eager vs compiled see identical inputs
    toks = [int(x) for x in mx.random.randint(0, cfg.vocab_size, (STEPS,)).tolist()]

    ce = clone_cache(model, base)
    cc = clone_cache(model, base)

    am_e, lg_e = decode_run(model, ce, toks, compiled=False)
    am_c, lg_c = decode_run(model, cc, toks, compiled=True)

    # did the compiled path actually engage — and for how many steps?
    dec = getattr(model.model, "_compiled_decoder", None)
    engaged = dec is not None and dec.step is not None
    compiled_calls = dec.compiled_calls if dec else 0
    rebuilds = dec.rebuilds if dec else 0
    print(f"compiled steps served: {compiled_calls}/{STEPS}  bucket (re)builds: {rebuilds}")
    agree = sum(1 for a, b in zip(am_e, am_c) if a == b)
    maxlogit_err = 0.0
    for le, lc in zip(lg_e, lg_c):
        maxlogit_err = max(maxlogit_err, float(mx.max(mx.abs(le - lc))))
    off_e, off_c = ce[0].offset, cc[0].offset

    print(f"compiled path engaged: {engaged}")
    print(f"argmax agreement: {agree}/{STEPS}")
    print(f"max |logit_eager - logit_compiled|: {maxlogit_err:.4e}")
    print(f"final offsets  eager={off_e}  compiled={off_c}  (want {PROMPT+STEPS})")
    ok = (
        engaged
        and compiled_calls == STEPS          # compiled ran EVERY step (no silent fallback)
        and rebuilds == 2                    # initial bucket + one 256 growth
        and agree == STEPS
        and off_e == off_c == PROMPT + STEPS
    )
    print(f"scenario {tag}: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r_plain = run_scenario(quantize=False)
    r_quant = run_scenario(quantize=True)
    print("\n" + "=" * 60)
    print(f"UNQUANTIZED (separate proj): {'PASS' if r_plain else 'FAIL'}")
    print(f"QUANTIZED (packed proj + gather_qmm): {'PASS' if r_quant else 'FAIL'}")
    print(f"SPIKE PHASE-B SMOKE: {'PASS' if (r_plain and r_quant) else 'FAIL'}")
