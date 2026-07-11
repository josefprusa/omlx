# SPDX-License-Identifier: Apache-2.0
"""Heterogeneous per-layer configuration for NemotronH Puzzle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import mlx.nn as nn
from mlx_lm.models.nemotron_h import (
    Model as NemotronHForCausalLM,
)
from mlx_lm.models.nemotron_h import (
    ModelArgs as NemotronHArgs,
)
from mlx_lm.models.nemotron_h import (
    NemotronHBlock,
    NemotronHModel,
)


@dataclass
class ModelArgs(NemotronHArgs):
    block_configs: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, params):
        values = dict(params)
        if "num_hidden_layers" not in values:
            blocks = values.get("layers_block_type") or values.get("block_configs")
            if blocks is not None:
                values["num_hidden_layers"] = len(blocks)
        return super().from_dict(values)


def _args_for_layer(args: ModelArgs, index: int) -> ModelArgs:
    if not args.block_configs or index >= len(args.block_configs):
        raise ValueError(f"Puzzle config is missing block_configs[{index}]")
    block = args.block_configs[index]
    width = block.get("moe_intermediate_size")
    top_k = block.get("num_experts_per_tok")
    if width is None or top_k is None:
        raise ValueError(
            f"Puzzle MoE block {index} requires moe_intermediate_size and "
            "num_experts_per_tok"
        )
    return replace(
        args,
        moe_intermediate_size=int(width),
        num_experts_per_tok=int(top_k),
    )


class PuzzleBackbone(NemotronHModel):
    def __init__(self, args: ModelArgs):
        nn.Module.__init__(self)
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            NemotronHBlock(
                _args_for_layer(args, index) if block_type == "E" else args,
                block_type,
            )
            for index, block_type in enumerate(args.hybrid_override_pattern)
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.fa_idx = 0
        for block in args.hybrid_override_pattern:
            if block == "*":
                break
            if block == "M":
                self.fa_idx += 1
        self.ssm_idx = 0
        for block in args.hybrid_override_pattern:
            if block == "M":
                break
            if block == "*":
                self.ssm_idx += 1


class Model(NemotronHForCausalLM):
    def __init__(self, args: ModelArgs):
        nn.Module.__init__(self)
        self.args = args
        self.backbone = PuzzleBackbone(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.model_type = args.model_type

    def sanitize(self, weights):
        if any(key.startswith("model.") for key in weights):
            weights = {
                (
                    f"backbone.{key.removeprefix('model.')}"
                    if key.startswith("model.")
                    else key
                ): value
                for key, value in weights.items()
            }
        return super().sanitize(weights)


__all__ = ["Model", "ModelArgs", "PuzzleBackbone"]
