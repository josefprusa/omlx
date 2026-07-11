# SPDX-License-Identifier: Apache-2.0
"""Register heterogeneous NemotronH Puzzle support with mlx-lm."""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)
_APPLIED = False


def apply_nemotron_h_puzzle_patch() -> bool:
    """Register the local model only when mlx-lm does not provide one."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        module = importlib.import_module("mlx_lm.models.nemotron_h_puzzle")
        source = "mlx-lm"
    except ImportError:
        module = importlib.import_module(f"{__name__}.model")
        sys.modules["mlx_lm.models.nemotron_h_puzzle"] = module
        models = importlib.import_module("mlx_lm.models")
        models.nemotron_h_puzzle = module
        source = "omlx"

    _APPLIED = True
    logger.info("NemotronH Puzzle model registered from %s", source)
    return source == "omlx"


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_nemotron_h_puzzle_patch", "is_applied"]
