# SPDX-License-Identifier: Apache-2.0
"""Parity tests for GLM NVFP4 global tensor scale sidecars."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from omlx.patches.glm_moe_dsa import (
    apply_glm_moe_dsa_patch,
    nvfp4_ts,
    switch_layers,
)
from omlx.patches.glm_moe_dsa import (
    glm_moe_dsa_model as glm,
)
from omlx.patches.glm_moe_dsa.switch_layers import SwitchGLU

apply_glm_moe_dsa_patch()

INPUT_DIM = 32
HIDDEN_DIM = 16
NUM_EXPERTS = 4
TOP_K = 2


def _switch_pair(seed=0):
    mx.random.seed(seed)
    stock = SwitchGLU(
        INPUT_DIM,
        HIDDEN_DIM,
        NUM_EXPERTS,
        fused_gate_up=True,
        inverse_scatter=True,
    )
    scaled = SwitchGLU(
        INPUT_DIM,
        HIDDEN_DIM,
        NUM_EXPERTS,
        fused_gate_up=True,
        inverse_scatter=True,
    )
    scaled.gate_up_proj.weight = stock.gate_up_proj.weight
    scaled.down_proj.weight = stock.down_proj.weight
    scaled.gate_up_ts = mx.ones((NUM_EXPERTS, 2), dtype=mx.float32)
    scaled.down_ts = mx.ones((NUM_EXPERTS,), dtype=mx.float32)
    mx.eval(stock.parameters(), scaled.parameters())
    return stock, scaled


def _inputs(tokens, seed=1):
    mx.random.seed(seed)
    x = mx.random.normal((1, tokens, INPUT_DIM))
    indices = mx.random.randint(
        0, NUM_EXPERTS, (1, tokens, TOP_K)
    ).astype(mx.uint32)
    mx.eval(x, indices)
    return x, indices


def _prescaled_reference(stock, gate_up_ts, down_ts):
    reference = SwitchGLU(
        INPUT_DIM,
        HIDDEN_DIM,
        NUM_EXPERTS,
        fused_gate_up=True,
        inverse_scatter=True,
    )
    weight = stock.gate_up_proj.weight
    reference.gate_up_proj.weight = mx.concatenate(
        [
            weight[:, :HIDDEN_DIM] * gate_up_ts[:, 0, None, None],
            weight[:, HIDDEN_DIM:] * gate_up_ts[:, 1, None, None],
        ],
        axis=1,
    )
    reference.down_proj.weight = stock.down_proj.weight * down_ts[:, None, None]
    mx.eval(reference.parameters())
    return reference


@pytest.mark.parametrize("tokens", [8, 32])
def test_unit_scales_are_exact_identity(tokens):
    stock, scaled = _switch_pair()
    x, indices = _inputs(tokens)

    assert mx.array_equal(stock(x, indices), scaled(x, indices))


@pytest.mark.parametrize("tokens", [8, 32])
def test_runtime_scales_match_prescaled_weights(tokens):
    stock, scaled = _switch_pair()
    gate_up_ts = mx.array(
        [[0.5, 1.5], [0.75, 1.25], [1.5, 0.5], [1.25, 0.75]],
        dtype=mx.float32,
    )
    down_ts = mx.array([0.5, 0.75, 1.25, 1.5], dtype=mx.float32)
    scaled.gate_up_ts = gate_up_ts
    scaled.down_ts = down_ts
    reference = _prescaled_reference(stock, gate_up_ts, down_ts)
    x, indices = _inputs(tokens)

    assert mx.allclose(
        scaled(x, indices), reference(x, indices), rtol=1e-5, atol=1e-6
    )


def test_kill_switch_reproduces_unscaled_output(monkeypatch):
    monkeypatch.setenv("OMLX_GLM_DISABLE_NVFP4_TS", "1")
    stock, scaled = _switch_pair()
    scaled.gate_up_ts = mx.full((NUM_EXPERTS, 2), 1.5)
    scaled.down_ts = mx.full((NUM_EXPERTS,), 0.5)
    x, indices = _inputs(8)

    assert mx.array_equal(stock(x, indices), scaled(x, indices))


def test_runtime_logs_enabled_and_disabled_engagement(monkeypatch, caplog):
    switch_layers._NVFP4_TS_LOGGED_MODES.clear()
    _, scaled = _switch_pair()
    x, indices = _inputs(8)

    with caplog.at_level("WARNING", logger=switch_layers.__name__):
        scaled(x, indices)
        monkeypatch.setenv("OMLX_GLM_DISABLE_NVFP4_TS", "1")
        scaled(x, indices)

    assert "tensor-scale fold enabled" in caplog.text
    assert "tensor-scale fold disabled" in caplog.text


def _tiny_model():
    args = glm.ModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=32,
        hidden_size=32,
        index_head_dim=16,
        index_n_heads=2,
        index_topk=4,
        intermediate_size=32,
        moe_intermediate_size=HIDDEN_DIM,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=NUM_EXPERTS,
        routed_scaling_factor=1.0,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=16,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=TOP_K,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        attention_bias=False,
    )
    return glm.Model(args)


def test_sanitize_is_noop_without_sidecars():
    model = _tiny_model()

    model.sanitize({"model.layers.1.mlp.gate.weight": mx.zeros((4, 32))})

    switch = model.model.layers[1].mlp.switch_mlp
    assert not hasattr(switch, "gate_up_ts")
    assert not hasattr(switch, "down_ts")


def test_sanitize_registers_sidecars_for_strict_loading():
    model = _tiny_model()
    key = "model.layers.1.mlp.switch_mlp.gate_up_ts"
    weights = {key: mx.ones((NUM_EXPERTS, 2), dtype=mx.float32)}

    assert key in model.sanitize(weights)

    parameters = dict(tree_flatten(model.parameters()))
    assert parameters[key].shape == (NUM_EXPERTS, 2)
    assert parameters[key].dtype == mx.float32
    assert parameters[
        "model.layers.1.mlp.switch_mlp.down_ts"
    ].shape == (NUM_EXPERTS,)


def test_cast_predicate_keeps_sidecars_fp32():
    predicate = _tiny_model().cast_predicate

    assert not predicate("model.layers.1.mlp.switch_mlp.gate_up_ts")
    assert not predicate("model.layers.1.mlp.switch_mlp.down_ts")
    assert predicate("model.layers.1.self_attn.q_a_proj.weight")


def test_patch_is_idempotent():
    assert nvfp4_ts.apply_glm_nvfp4_ts_patch() is False
