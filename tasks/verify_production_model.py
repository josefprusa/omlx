#!/usr/bin/env python3
"""Run a direct serving-engine smoke test against one local production model."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import mlx.core as mx

from omlx.engine.batched import BatchedEngine
from omlx.engine.vlm import VLMBatchedEngine
from omlx.model_settings import ModelSettingsManager


def _gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


async def run(args: argparse.Namespace) -> dict:
    model_path = Path(args.model).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"missing model config: {model_path / 'config.json'}")

    settings = ModelSettingsManager(Path(args.settings_dir).expanduser()).get_settings(
        args.settings_id or model_path.name
    )
    if args.int8_start is not None:
        settings.int8_mla_kv_enabled = True
        settings.int8_mla_kv_bits = 8
        settings.int8_mla_kv_start = args.int8_start
    if args.disable_mtp:
        settings.mtp_enabled = False

    engine_cls = VLMBatchedEngine if args.vlm else BatchedEngine
    engine = engine_cls(
        str(model_path),
        trust_remote_code=settings.trust_remote_code,
        enable_thinking=settings.enable_thinking,
        model_settings=settings,
    )

    mx.reset_peak_memory()
    load_started = time.perf_counter()
    await engine.start()
    load_seconds = time.perf_counter() - load_started

    started = time.perf_counter()
    first_token_seconds = None
    final = None
    chunks: list[str] = []
    prompt = ("context " * args.padding_words) + args.prompt
    try:
        async for output in engine.stream_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        ):
            final = output
            if output.new_text:
                chunks.append(output.new_text)
                if first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - started
    finally:
        total_seconds = time.perf_counter() - started
        model_type = engine.model_type
        active_memory = mx.get_active_memory()
        peak_memory = mx.get_peak_memory()
        await engine.stop()

    if final is None or final.completion_tokens < 1:
        raise RuntimeError("generation produced no tokens")

    decode_seconds = max(total_seconds - (first_token_seconds or total_seconds), 0.0)
    decode_tokens = max(final.completion_tokens - 1, 0)
    result = {
        "model": model_path.name,
        "model_type": model_type,
        "engine": engine_cls.__name__,
        "load_seconds": round(load_seconds, 3),
        "prompt_tokens": final.prompt_tokens,
        "completion_tokens": final.completion_tokens,
        "first_token_seconds": round(first_token_seconds or total_seconds, 3),
        "total_generation_seconds": round(total_seconds, 3),
        "decode_tokens_per_second": round(decode_tokens / decode_seconds, 3)
        if decode_seconds
        else None,
        "active_memory_gib": _gib(active_memory),
        "peak_memory_gib": _gib(peak_memory),
        "finish_reason": final.finish_reason,
        "output": "".join(chunks).strip(),
        "int8_mla_kv_enabled": bool(
            getattr(settings, "int8_mla_kv_enabled", False)
        ),
        "int8_mla_kv_start": getattr(settings, "int8_mla_kv_start", None),
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--settings-dir", default="~/.omlx")
    parser.add_argument("--settings-id")
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--int8-start", type=int)
    parser.add_argument("--disable-mtp", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--padding-words", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default="What is 2 + 2? Reply with the final number clearly.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
