# SPDX-License-Identifier: Apache-2.0
"""Tests for NemotronH's exact NVFP4 expert tensor-scale fold."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx

from omlx.patches import nemotron_h_nvfp4_ts
from omlx.utils import model_loading


def test_score_fold_uses_relu2_homogeneity(monkeypatch):
    indices = mx.array([[[0, 1]]], dtype=mx.uint32)
    scores = mx.array([[[0.25, 0.75]]], dtype=mx.float32)
    switch_output = mx.array([[[[2.0], [3.0]]]], dtype=mx.float32)
    switch = MagicMock(return_value=switch_output)
    switch.fc1_ts = mx.array([2.0, 3.0])
    switch.fc2_ts = mx.array([5.0, 7.0])
    module = SimpleNamespace(
        gate=MagicMock(return_value=(indices, scores)),
        switch_mlp=switch,
        moe_latent_size=None,
        config=SimpleNamespace(n_shared_experts=None),
    )

    output = nemotron_h_nvfp4_ts._moe_call(module, mx.ones((1, 1, 1)))

    expected_scores = scores * mx.array([[[2.0**2 * 5.0, 3.0**2 * 7.0]]])
    expected = (switch_output * expected_scores[..., None]).sum(axis=-2)
    assert mx.array_equal(output, expected)


def test_kill_switch_skips_scale_fold(monkeypatch):
    monkeypatch.setenv("OMLX_NEMO_DISABLE_NVFP4_TS", "1")
    indices = mx.array([[[0]]], dtype=mx.uint32)
    scores = mx.array([[[0.5]]])
    switch = MagicMock(return_value=mx.array([[[[4.0]]]]))
    switch.fc1_ts = mx.array([10.0])
    switch.fc2_ts = mx.array([10.0])
    module = SimpleNamespace(
        gate=MagicMock(return_value=(indices, scores)),
        switch_mlp=switch,
        moe_latent_size=None,
        config=SimpleNamespace(n_shared_experts=None),
    )

    output = nemotron_h_nvfp4_ts._moe_call(module, mx.ones((1, 1, 1)))

    assert mx.array_equal(output, mx.array([[[2.0]]]))


def test_preload_dispatch_uses_config_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
    apply = MagicMock(return_value=True)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.nemotron_h_nvfp4_ts",
        MagicMock(apply_nemotron_h_nvfp4_ts_patch=apply),
    )
    (tmp_path / "config.json").write_text(
        '{"model_type":"nemotron_h","omlx_moe_nvfp4_ts":true}'
    )

    model_loading.maybe_apply_pre_load_patches(str(tmp_path))

    apply.assert_called_once_with()
