# SPDX-License-Identifier: Apache-2.0
"""Validate a converted Puzzle oQ48 artifact against its BF16 source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from omlx.oq import _LazyTensorIndex
from omlx.tools.puzzle_convert import (
    BLOCK_CHAR,
    EXPERT_SPEC,
    MTP_CONFIG_KEYS,
    SHELL_SPEC,
    SHELL_STAGES,
    quantize_expert,
    quantize_shell,
    resolve_blocks,
)


def _index(path: Path):
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files in {path}")
    return _LazyTensorIndex(files)


def expected_modules(blocks: list[dict]) -> tuple[list[tuple[str, str]], list[int]]:
    shell = [("lm_head", "lm_head")]
    moe_layers = []
    for layer, block in enumerate(blocks):
        char = BLOCK_CHAR[block["block_type"]]
        source = f"model.layers.{layer}.mixer"
        output = f"backbone.layers.{layer}.mixer"
        shell.extend(
            (f"{output}.{target}", f"{source}.{target}")
            for target in SHELL_STAGES[char]
        )
        if char == "E":
            moe_layers.append(layer)
    return shell, moe_layers


def _spread(items, count):
    if len(items) <= count:
        return list(items)
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def check_census(output, blocks: list[dict]) -> None:
    shell, moe_layers = expected_modules(blocks)
    expected = {path for path, _ in shell}
    expected.update(
        f"backbone.layers.{layer}.mixer.switch_mlp.{projection}"
        for layer in moe_layers
        for projection in ("fc1", "fc2")
    )
    actual = {
        name.removesuffix(".scales")
        for name in output
        if name.endswith(".scales")
    }
    if actual != expected:
        raise AssertionError(
            f"Quantized module census mismatch: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )
    for module in expected:
        for suffix in ("weight", "scales", "biases"):
            if f"{module}.{suffix}" not in output:
                raise AssertionError(f"Missing {module}.{suffix}")
    if any(name.startswith("mtp.") for name in output):
        raise AssertionError("Converted artifact still contains MTP tensors")


def check_config(config: dict, blocks: list[dict]) -> None:
    if any(key in config for key in MTP_CONFIG_KEYS):
        raise AssertionError("Converted config still contains MTP settings")
    quantization = config.get("quantization")
    if quantization != config.get("quantization_config"):
        raise AssertionError("quantization and quantization_config differ")
    if {key: quantization[key] for key in EXPERT_SPEC} != EXPERT_SPEC:
        raise AssertionError("Global expert quantization recipe differs")
    shell, _ = expected_modules(blocks)
    for module, _ in shell:
        if quantization.get(module) != SHELL_SPEC:
            raise AssertionError(f"Wrong shell recipe for {module}")


def _equal(actual, expected) -> bool:
    mx.eval(actual, expected)
    return bool(mx.array_equal(actual, expected))


def check_parity(source, output, blocks: list[dict], experts: int) -> int:
    shell, moe_layers = expected_modules(blocks)
    shell_by_type = {"M": [], "E": [], "*": [], "global": []}
    for output_name, source_name in shell:
        if output_name == "lm_head":
            shell_by_type["global"].append((output_name, source_name))
            continue
        layer = int(output_name.split(".")[2])
        shell_by_type[BLOCK_CHAR[blocks[layer]["block_type"]]].append(
            (output_name, source_name)
        )

    checked = 0
    samples = [values[len(values) // 2] for values in shell_by_type.values() if values]
    for output_name, source_name in samples:
        expected = quantize_shell(source[f"{source_name}.weight"].astype(mx.bfloat16))
        for suffix, value in zip(("weight", "scales", "biases"), expected):
            if not _equal(output[f"{output_name}.{suffix}"], value):
                raise AssertionError(f"Shell parity failed for {output_name}.{suffix}")
        checked += 1
        mx.clear_cache()

    expert_ids = [0, experts - 1]
    for layer in _spread(moe_layers, 2):
        source_prefix = f"model.layers.{layer}.mixer.experts"
        output_prefix = f"backbone.layers.{layer}.mixer.switch_mlp"
        for source_projection, output_projection in (("up_proj", "fc1"), ("down_proj", "fc2")):
            stored = tuple(
                output[f"{output_prefix}.{output_projection}.{suffix}"]
                for suffix in ("weight", "scales", "biases")
            )
            for expert in expert_ids:
                expected = quantize_expert(
                    source[
                        f"{source_prefix}.{expert}.{source_projection}.weight"
                    ].astype(mx.bfloat16)
                )
                if not all(_equal(value[expert], reference) for value, reference in zip(stored, expected)):
                    raise AssertionError(
                        f"Expert parity failed for layer={layer} expert={expert} "
                        f"projection={output_projection}"
                    )
                checked += 1
            del stored
            mx.clear_cache()
    return checked


def run(source_path: Path, output_path: Path) -> dict[str, int]:
    source_config = json.loads((source_path / "config.json").read_text())
    output_config = json.loads((output_path / "config.json").read_text())
    blocks = resolve_blocks(source_config)
    source = _index(source_path)
    output = _index(output_path)
    check_census(output, blocks)
    check_config(output_config, blocks)
    checked = check_parity(source, output, blocks, int(source_config["n_routed_experts"]))
    return {"layers": len(blocks), "parity_samples": checked}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.src, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
