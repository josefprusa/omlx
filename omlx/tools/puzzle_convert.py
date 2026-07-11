# SPDX-License-Identifier: Apache-2.0
"""Convert a BF16 NemotronH Puzzle checkpoint to the oQ48 serving recipe."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import mlx.core as mx

from omlx.oq import _LazyTensorIndex, _quantize_chunked

SHELL_SPEC = {"group_size": 64, "bits": 8, "mode": "affine"}
EXPERT_SPEC = {"group_size": 64, "bits": 4, "mode": "affine"}
SHELL_STAGES = {
    "M": ("in_proj", "out_proj"),
    "E": (
        "shared_experts.up_proj",
        "shared_experts.down_proj",
        "fc1_latent_proj",
        "fc2_latent_proj",
    ),
    "*": ("q_proj", "o_proj"),
}
BLOCK_CHAR = {"mamba": "M", "moe": "E", "attention": "*"}
MTP_CONFIG_KEYS = (
    "num_nextn_predict_layers",
    "mtp_block_configs",
    "mtp_layers_block_type",
)


def quantize_shell(weight):
    return _quantize_chunked(weight, **SHELL_SPEC)


def quantize_expert(weight):
    return _quantize_chunked(weight, **EXPERT_SPEC)


def resolve_blocks(config: dict) -> list[dict]:
    blocks = config.get("block_configs")
    if isinstance(blocks, list) and blocks:
        return blocks
    types = config.get("layers_block_type")
    if isinstance(types, list) and types:
        return [{"block_type": block_type} for block_type in types]
    raise ValueError("Puzzle config needs block_configs or layers_block_type")


def build_output_config(source: dict, blocks: list[dict]) -> dict:
    config = copy.deepcopy(source)
    for key in MTP_CONFIG_KEYS:
        config.pop(key, None)
    quantization = dict(EXPERT_SPEC)
    for index, block in enumerate(blocks):
        char = BLOCK_CHAR[block["block_type"]]
        prefix = f"backbone.layers.{index}.mixer"
        for target in SHELL_STAGES[char]:
            quantization[f"{prefix}.{target}"] = dict(SHELL_SPEC)
    quantization["lm_head"] = dict(SHELL_SPEC)
    config["model_type"] = "nemotron_h_puzzle"
    config["num_hidden_layers"] = len(blocks)
    config["quantization"] = quantization
    config["quantization_config"] = copy.deepcopy(quantization)
    return config


def _quantized(prefix: str, weight, quantize) -> dict[str, mx.array]:
    packed, scales, biases = quantize(weight.astype(mx.bfloat16))
    return {
        f"{prefix}.weight": packed,
        f"{prefix}.scales": scales,
        f"{prefix}.biases": biases,
    }


def _load(index, name: str):
    if name not in index:
        raise KeyError(f"Missing source tensor {name}")
    return index[name]


def _shell(index, source: str, output: str) -> dict[str, mx.array]:
    return _quantized(output, _load(index, f"{source}.weight"), quantize_shell)


def convert_mamba(index, source: str, output: str) -> dict[str, mx.array]:
    tensors = {
        f"{output}.{name}": _load(index, f"{source}.{name}")
        for name in (
            "A_log",
            "D",
            "dt_bias",
            "conv1d.weight",
            "conv1d.bias",
            "norm.weight",
        )
    }
    for target in SHELL_STAGES["M"]:
        tensors.update(_shell(index, f"{source}.{target}", f"{output}.{target}"))
    return tensors


def convert_attention(index, source: str, output: str) -> dict[str, mx.array]:
    tensors = {}
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        source_name = f"{source}.{projection}.weight"
        output_name = f"{output}.{projection}"
        if projection in SHELL_STAGES["*"]:
            tensors.update(_quantized(output_name, _load(index, source_name), quantize_shell))
        else:
            tensors[f"{output_name}.weight"] = _load(index, source_name).astype(
                mx.bfloat16
            )
    return tensors


def convert_experts(index, source: str, output: str, count: int) -> dict[str, mx.array]:
    results = {name: [[], [], []] for name in ("fc1", "fc2")}
    for expert in range(count):
        base = f"{source}.experts.{expert}"
        for source_projection, output_projection in (
            ("up_proj", "fc1"),
            ("down_proj", "fc2"),
        ):
            values = quantize_expert(
                _load(index, f"{base}.{source_projection}.weight").astype(mx.bfloat16)
            )
            for bucket, value in zip(results[output_projection], values):
                bucket.append(value)

    tensors = {}
    for projection, buckets in results.items():
        for suffix, values in zip(("weight", "scales", "biases"), buckets):
            tensors[f"{output}.switch_mlp.{projection}.{suffix}"] = mx.stack(values)
    return tensors


def convert_moe(index, source: str, output: str, experts: int) -> dict[str, mx.array]:
    tensors = {
        f"{output}.gate.weight": _load(index, f"{source}.gate.weight").astype(
            mx.bfloat16
        ),
        f"{output}.gate.e_score_correction_bias": _load(
            index, f"{source}.gate.e_score_correction_bias"
        ),
    }
    for target in SHELL_STAGES["E"]:
        tensors.update(_shell(index, f"{source}.{target}", f"{output}.{target}"))
    tensors.update(convert_experts(index, source, output, experts))
    return tensors


class ShardWriter:
    def __init__(self, output: Path, limit_bytes: int):
        self.output = output
        self.limit_bytes = limit_bytes
        self.pending: dict[str, mx.array] = {}
        self.pending_bytes = 0
        self.shards: list[tuple[Path, list[str]]] = []
        self.total_size = 0

    def add(self, tensors: dict[str, mx.array]) -> None:
        size = sum(tensor.nbytes for tensor in tensors.values())
        if self.pending and self.pending_bytes + size > self.limit_bytes:
            self._flush()
        self.pending.update(tensors)
        self.pending_bytes += size

    def _flush(self) -> None:
        path = self.output / f"model-{len(self.shards) + 1:05d}-of-PENDING.safetensors"
        mx.save_safetensors(str(path), self.pending, metadata={"format": "mlx"})
        self.shards.append((path, list(self.pending)))
        self.total_size += path.stat().st_size
        self.pending = {}
        self.pending_bytes = 0
        mx.clear_cache()

    def finalize(self) -> int:
        if self.pending:
            self._flush()
        count = len(self.shards)
        weight_map = {}
        for index, (old_path, names) in enumerate(self.shards, 1):
            new_name = f"model-{index:05d}-of-{count:05d}.safetensors"
            old_path.rename(self.output / new_name)
            weight_map.update(dict.fromkeys(names, new_name))
        payload = {"metadata": {"total_size": self.total_size}, "weight_map": weight_map}
        (self.output / "model.safetensors.index.json").write_text(
            json.dumps(payload, indent=2)
        )
        return self.total_size


def _copy_sidecars(source: Path, output: Path) -> None:
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        "hf_quant_config.json",
    }
    for path in source.iterdir():
        if (
            path.is_file()
            and path.name not in excluded
            and not path.name.endswith(".safetensors")
            and not path.name.endswith(".safetensors.index.json")
        ):
            shutil.copy2(path, output / path.name)


def convert(source: Path, output: Path, shard_size_gb: float = 5.0) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("Source and output must differ")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.safetensors")):
        raise ValueError("Output already contains safetensors files")

    config = json.loads((source / "config.json").read_text())
    blocks = resolve_blocks(config)
    index = _LazyTensorIndex(sorted(source.glob("*.safetensors")))
    writer = ShardWriter(output, int(shard_size_gb * 1_000_000_000))
    writer.add(
        {
            "backbone.embeddings.weight": _load(index, "model.embeddings.weight").astype(
                mx.bfloat16
            ),
            "backbone.norm_f.weight": _load(index, "model.norm_f.weight").astype(
                mx.bfloat16
            ),
        }
    )
    writer.add(_quantized("lm_head", _load(index, "lm_head.weight"), quantize_shell))

    experts = int(config["n_routed_experts"])
    for layer, block in enumerate(blocks):
        char = BLOCK_CHAR[block["block_type"]]
        source_prefix = f"model.layers.{layer}.mixer"
        output_prefix = f"backbone.layers.{layer}.mixer"
        tensors = {
            f"backbone.layers.{layer}.norm.weight": _load(
                index, f"model.layers.{layer}.norm.weight"
            ).astype(mx.bfloat16)
        }
        if char == "M":
            tensors.update(convert_mamba(index, source_prefix, output_prefix))
        elif char == "*":
            tensors.update(convert_attention(index, source_prefix, output_prefix))
        else:
            tensors.update(convert_moe(index, source_prefix, output_prefix, experts))
        mx.eval(*tensors.values())
        writer.add(tensors)
        mx.synchronize()
        mx.clear_cache()

    (output / "config.json").write_text(
        json.dumps(build_output_config(config, blocks), indent=2)
    )
    _copy_sidecars(source, output)
    writer.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-size-gb", type=float, default=5.0)
    args = parser.parse_args(argv)
    convert(args.src, args.out, args.shard_size_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
