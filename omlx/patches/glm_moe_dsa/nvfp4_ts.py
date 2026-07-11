# SPDX-License-Identifier: Apache-2.0
"""Restore NVFP4 global tensor scales for GLM routed experts."""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_glm_nvfp4_ts_patch() -> bool:
    """Teach the vendored GLM model to load optional NVFP4 scale sidecars."""
    global _APPLIED
    if _APPLIED:
        return False

    from . import glm_moe_dsa_model as glm

    original_sanitize = glm.Model.sanitize
    original_cast_predicate = glm.Model.cast_predicate.fget

    def sanitize(self, weights):
        weights = original_sanitize(self, weights)
        if not any(key.endswith(".gate_up_ts") for key in weights):
            return weights

        num_experts = int(self.args.n_routed_experts)
        registered = 0
        for layer in self.model.layers:
            switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
            if switch is None:
                continue
            switch.gate_up_ts = mx.ones((num_experts, 2), dtype=mx.float32)
            switch.down_ts = mx.ones((num_experts,), dtype=mx.float32)
            registered += 1
        logger.info("Registered GLM NVFP4 scale sidecars on %d MoE layers", registered)
        return weights

    def cast_predicate(self):
        base = original_cast_predicate(self)
        return lambda key: base(key) and not key.endswith(("gate_up_ts", "down_ts"))

    glm.Model.sanitize = sanitize
    glm.Model.cast_predicate = property(cast_predicate)
    _APPLIED = True
    return True


__all__ = ["apply_glm_nvfp4_ts_patch"]
