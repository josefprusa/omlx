# SPDX-License-Identifier: Apache-2.0
"""Tests for the deterministic NemotronH Puzzle oQ48 converter."""

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.tools import puzzle_convert


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize(
    "quantize,spec",
    [
        (puzzle_convert.quantize_shell, puzzle_convert.SHELL_SPEC),
        (puzzle_convert.quantize_expert, puzzle_convert.EXPERT_SPEC),
    ],
)
def test_quantization_matches_quantized_linear(dtype, quantize, spec):
    mx.random.seed(0)
    linear = nn.Linear(128, 256, bias=False)
    linear.weight = linear.weight.astype(dtype)

    actual = quantize(linear.weight)
    expected = nn.QuantizedLinear.from_linear(linear, **spec)
    mx.eval(*actual, expected.parameters())

    assert mx.array_equal(actual[0], expected.weight)
    assert mx.array_equal(actual[1], expected.scales)
    assert mx.array_equal(actual[2], expected.biases)


def test_recipe_counts_match_real_layout():
    blocks = (
        [{"block_type": "mamba"}] * 40
        + [{"block_type": "moe"}] * 40
        + [{"block_type": "attention"}] * 8
    )

    shell_count = sum(
        len(puzzle_convert.SHELL_STAGES[puzzle_convert.BLOCK_CHAR[b["block_type"]]])
        for b in blocks
    )

    assert shell_count == 256
    assert shell_count + 1 == 257
    assert 2 * sum(b["block_type"] == "moe" for b in blocks) == 80


def test_output_config_encodes_recipe_and_drops_mtp():
    blocks = [
        {"block_type": "mamba"},
        {
            "block_type": "moe",
            "moe_intermediate_size": 1280,
            "num_experts_per_tok": 4,
        },
        {"block_type": "attention"},
    ]
    source = {
        "model_type": "nemotron_h_puzzle",
        "block_configs": blocks,
        "layers_block_type": [b["block_type"] for b in blocks],
        "num_nextn_predict_layers": 1,
        "mtp_block_configs": [{}],
    }

    output = puzzle_convert.build_output_config(source, blocks)

    assert output["num_hidden_layers"] == 3
    assert output["quantization"] == output["quantization_config"]
    assert all(key not in output for key in puzzle_convert.MTP_CONFIG_KEYS)
    assert output["quantization"]["group_size"] == 64
    assert output["quantization"]["bits"] == 4
    assert output["quantization"]["backbone.layers.0.mixer.in_proj"] == (
        puzzle_convert.SHELL_SPEC
    )
    assert output["quantization"]["backbone.layers.1.mixer.fc1_latent_proj"] == (
        puzzle_convert.SHELL_SPEC
    )
    assert output["quantization"]["backbone.layers.2.mixer.o_proj"] == (
        puzzle_convert.SHELL_SPEC
    )


def test_resolve_blocks_rejects_missing_layout():
    with pytest.raises(ValueError, match="block_configs"):
        puzzle_convert.resolve_blocks({})


def test_shard_writer_round_trip(tmp_path):
    writer = puzzle_convert.ShardWriter(tmp_path, limit_bytes=20)
    writer.add({"first": mx.arange(4, dtype=mx.float32)})
    writer.add({"second": mx.arange(4, dtype=mx.float32)})

    total = writer.finalize()

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert total > 0
    assert len(set(index["weight_map"].values())) == 2
    assert mx.array_equal(
        mx.load(str(tmp_path / index["weight_map"]["first"]))["first"],
        mx.arange(4, dtype=mx.float32),
    )
