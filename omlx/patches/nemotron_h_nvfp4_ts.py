# SPDX-License-Identifier: Apache-2.0
"""Restore NVFP4 global expert scales for NemotronH checkpoints."""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)
_APPLIED = False
_LOGGED: set[str] = set()


def _disabled() -> bool:
    return os.environ.get("OMLX_NEMO_DISABLE_NVFP4_TS") == "1"


def _moe_call(self, x):
    residual = x
    indices, scores = self.gate(x)
    scales = None
    if hasattr(self.switch_mlp, "fc1_ts"):
        if _disabled():
            mode = "disabled"
        else:
            mode = "engaged"
            scales = (
                self.switch_mlp.fc1_ts[indices] ** 2
            ) * self.switch_mlp.fc2_ts[indices]
        if mode not in _LOGGED:
            _LOGGED.add(mode)
            logger.warning("NemotronH NVFP4 tensor-scale fold %s", mode)
    if scales is not None:
        scores = scores.astype(mx.float32) * scales
    if self.moe_latent_size is not None:
        x = self.fc1_latent_proj(x)
    y = self.switch_mlp(x, indices)
    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if self.moe_latent_size is not None:
        y = self.fc2_latent_proj(y)
    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(residual)
    return y


def apply_nemotron_h_nvfp4_ts_patch() -> bool:
    global _APPLIED
    if _APPLIED:
        return False

    from mlx_lm.models import nemotron_h

    scaled_moe = type(
        "_NemotronHNVFP4ScaledMoE",
        (nemotron_h.NemotronHMoE,),
        {"__call__": _moe_call},
    )
    original_sanitize = nemotron_h.Model.sanitize
    original_cast_predicate = nemotron_h.Model.cast_predicate.fget

    def sanitize(self, weights):
        weights = original_sanitize(self, weights)
        if not any(key.endswith(".fc1_ts") for key in weights):
            return weights
        count = 0
        for layer in self.layers:
            mixer = getattr(layer, "mixer", None)
            switch = getattr(mixer, "switch_mlp", None)
            if switch is None:
                continue
            mixer.__class__ = scaled_moe
            switch.fc1_ts = mx.ones((self.args.n_routed_experts,), dtype=mx.float32)
            switch.fc2_ts = mx.ones((self.args.n_routed_experts,), dtype=mx.float32)
            count += 1
        logger.info("Registered NemotronH NVFP4 scale sidecars on %d MoE layers", count)
        return weights

    def cast_predicate(self):
        base = original_cast_predicate(self)
        return lambda key: base(key) and not key.endswith(("fc1_ts", "fc2_ts"))

    nemotron_h.Model.sanitize = sanitize
    nemotron_h.Model.cast_predicate = property(cast_predicate)
    _APPLIED = True
    return True


__all__ = ["apply_nemotron_h_nvfp4_ts_patch"]
