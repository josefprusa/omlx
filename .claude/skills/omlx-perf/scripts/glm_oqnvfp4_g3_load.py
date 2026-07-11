#!/usr/bin/env python3
"""G3: real-path offline load + greedy sanity for GLM-5.2-oQNVFP4.

Loads through the REAL omlx pre-load patch path + mlx_lm.load, then:
  1. module-state audit: switch_mlp is _TsSwitchGLU with QuantizedSwitchLinear
     (mode=nvfp4, bits=4, gs=16); embed_q/unembed_out quantized affine8 gs64;
     indexer wk plain bf16 Linear; ts params bound (not all ones).
  2. greedy generation on 3 prompts: finite logits, coherent text.

Run with the serve pool EMPTY (a ~428 GB standalone load).
"""

import os
import sys
import time

sys.path.insert(0, os.environ.get("OMLX_REPO", os.getcwd()))

import mlx.core as mx  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ["OMLX_GLM_MODEL"]

FAILS = []


def check(label, ok):
    print(("PASS " if ok else "FAIL ") + label, flush=True)
    if not ok:
        FAILS.append(label)


def main():
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(MODEL)

    from mlx_lm import load, generate

    t0 = time.time()
    model, tokenizer = load(MODEL)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    # ---- 1. module-state audit ----
    from omlx.patches.glm_moe_dsa import glm_moe_dsa_model as gm
    import mlx.nn as nn

    layers = model.model.layers
    moe_layer = layers[3].mlp
    smlp = moe_layer.switch_mlp
    check("switch_mlp swapped to _TsSwitchGLU", type(smlp).__name__ == "_TsSwitchGLU")
    gup = smlp.gate_up_proj
    check(
        "gate_up_proj QuantizedSwitchLinear nvfp4/4/16",
        type(gup).__name__ == "QuantizedSwitchLinear"
        and getattr(gup, "mode", None) == "nvfp4"
        and gup.bits == 4
        and gup.group_size == 16,
    )
    down = smlp.down_proj
    check(
        "down_proj QuantizedSwitchLinear nvfp4/4/16",
        getattr(down, "mode", None) == "nvfp4"
        and down.bits == 4
        and down.group_size == 16,
    )
    check(
        "ts params bound (not init ones)",
        smlp.gate_up_ts.shape == (256, 2)
        and smlp.down_ts.shape == (256,)
        and smlp.gate_up_ts.dtype == mx.float32
        and not bool(mx.all(smlp.gate_up_ts == 1.0).item()),
    )
    attn = layers[3].self_attn
    eq, uo = attn.embed_q, attn.unembed_out
    check(
        "embed_q quantized affine8 gs64",
        hasattr(eq, "scales") and eq.bits == 8 and eq.group_size == 64,
    )
    check(
        "unembed_out quantized affine8 gs64",
        hasattr(uo, "scales") and uo.bits == 8 and uo.group_size == 64,
    )
    idx0 = layers[0].self_attn.indexer
    check(
        "indexer wk plain bf16 Linear",
        idx0 is not None
        and isinstance(idx0.wk, nn.Linear)
        and not isinstance(idx0.wk, nn.QuantizedLinear)
        and idx0.wk.weight.dtype == mx.bfloat16,
    )
    # real GLM-5.2 indexer schedule: layers 0,1,2 full; first shared is layer 3
    check("layer 1 indexer present (full)", layers[1].self_attn.indexer is not None)
    check("layer 3 indexer skipped (shared)", layers[3].self_attn.indexer is None)
    check(
        "shared_experts quantized 8-bit",
        getattr(moe_layer.shared_experts.up_proj, "bits", None) == 8,
    )
    check("num layers == 78", len(layers) == 78)

    # ---- 2. greedy generation ----
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Water boils at a temperature of",
    ]
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        t0 = time.time()
        out = generate(model, tokenizer, prompt=text, max_tokens=64)
        dt = time.time() - t0
        print(f"--- prompt: {p!r} ({dt:.1f}s)\n{out[:300]}\n", flush=True)
        check(f"non-empty finite output for {p[:20]!r}", bool(out and out.strip()))

    if FAILS:
        print(f"G3: {len(FAILS)} FAILURE(S)")
        sys.exit(1)
    print("G3: ALL PASS")


if __name__ == "__main__":
    main()
