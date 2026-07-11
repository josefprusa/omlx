# SPDX-License-Identifier: Apache-2.0
"""Tests for heterogeneous NemotronH Puzzle model registration."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from omlx.utils import model_loading

TINY_CONFIG = {
    "model_type": "nemotron_h_puzzle",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "max_position_embeddings": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "attention_bias": False,
    "mamba_num_heads": 4,
    "mamba_head_dim": 16,
    "mamba_proj_bias": False,
    "ssm_state_size": 8,
    "conv_kernel": 4,
    "n_groups": 2,
    "mlp_bias": False,
    "layer_norm_epsilon": 1e-5,
    "use_bias": False,
    "use_conv_bias": True,
    "moe_latent_size": 32,
    "moe_shared_expert_intermediate_size": 48,
    "n_group": 1,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "layers_block_type": ["mamba", "moe", "attention", "moe"],
    "block_configs": [
        {"block_type": "mamba"},
        {
            "block_type": "moe",
            "moe_intermediate_size": 128,
            "num_experts_per_tok": 2,
        },
        {"block_type": "attention"},
        {
            "block_type": "moe",
            "moe_intermediate_size": 256,
            "num_experts_per_tok": 3,
        },
    ],
}


@pytest.fixture(scope="module")
def puzzle_module():
    from omlx.patches.nemotron_h_puzzle import apply_nemotron_h_puzzle_patch

    apply_nemotron_h_puzzle_patch()
    import mlx_lm.models.nemotron_h_puzzle as puzzle

    return puzzle


def build_model(puzzle_module):
    args = puzzle_module.ModelArgs.from_dict(TINY_CONFIG)
    model = puzzle_module.Model(args)
    mx.eval(model.parameters())
    return model, args


def test_patch_registers_module_and_is_idempotent(puzzle_module):
    from omlx.patches.nemotron_h_puzzle import (
        apply_nemotron_h_puzzle_patch,
        is_applied,
    )

    assert is_applied()
    assert apply_nemotron_h_puzzle_patch() is False
    assert sys.modules["mlx_lm.models.nemotron_h_puzzle"] is puzzle_module


def test_preload_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
    apply = MagicMock(return_value=True)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.nemotron_h_puzzle",
        MagicMock(apply_nemotron_h_puzzle_patch=apply),
    )
    (tmp_path / "config.json").write_text('{"model_type":"nemotron_h_puzzle"}')

    model_loading.maybe_apply_pre_load_patches(str(tmp_path))

    apply.assert_called_once_with()


def test_config_derives_layer_count(puzzle_module):
    config = dict(TINY_CONFIG)
    config.pop("num_hidden_layers")

    args = puzzle_module.ModelArgs.from_dict(config)

    assert args.num_hidden_layers == 4
    assert args.moe_intermediate_size is None
    assert args.num_experts_per_tok is None


def test_rejects_incomplete_moe_block(puzzle_module):
    config = dict(TINY_CONFIG)
    config["block_configs"] = list(TINY_CONFIG["block_configs"])
    config["block_configs"][1] = {"block_type": "moe"}
    args = puzzle_module.ModelArgs.from_dict(config)

    with pytest.raises(ValueError, match="requires moe_intermediate_size"):
        puzzle_module.Model(args)


def test_per_layer_shapes_and_top_k(puzzle_module):
    model, _ = build_model(puzzle_module)
    first = model.backbone.layers[1].mixer
    second = model.backbone.layers[3].mixer

    assert first.switch_mlp.fc1.weight.shape == (8, 128, 32)
    assert second.switch_mlp.fc1.weight.shape == (8, 256, 32)
    assert first.switch_mlp.fc2.weight.shape == (8, 32, 128)
    assert second.switch_mlp.fc2.weight.shape == (8, 32, 256)
    assert first.gate.top_k == 2
    assert second.gate.top_k == 3


def test_forward_produces_finite_logits(puzzle_module):
    model, args = build_model(puzzle_module)

    logits = model(mx.array([[1, 2, 3, 4]]))
    mx.eval(logits)

    assert logits.shape == (1, 4, args.vocab_size)
    assert not bool(mx.any(mx.isnan(logits)))


def test_sanitize_remaps_and_stacks_raw_weights(puzzle_module):
    model, _ = build_model(puzzle_module)
    weights = {}
    for expert in range(8):
        prefix = f"model.layers.1.mixer.experts.{expert}"
        weights[f"{prefix}.up_proj.weight"] = mx.zeros((128, 32))
        weights[f"{prefix}.down_proj.weight"] = mx.zeros((32, 128))
    weights["model.layers.0.mixer.conv1d.weight"] = mx.zeros((72, 1, 4))
    weights["mtp.layers.0.norm.weight"] = mx.zeros((64,))

    result = model.sanitize(weights)

    assert result[
        "backbone.layers.1.mixer.switch_mlp.fc1.weight"
    ].shape == (8, 128, 32)
    assert result[
        "backbone.layers.1.mixer.switch_mlp.fc2.weight"
    ].shape == (8, 32, 128)
    assert result["backbone.layers.0.mixer.conv1d.weight"].shape == (72, 4, 1)
    assert not any(".experts." in key or key.startswith("mtp.") for key in result)
