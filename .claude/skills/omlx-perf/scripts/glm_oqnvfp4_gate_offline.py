#!/usr/bin/env python3
"""Offline G1/G2 gates for a GLM-5.2-oQNVFP4 conversion (real bytes).

G1 census: the output tensor-name set must EXACTLY equal the expected set
derived from the output config (layers emitted, indexer_types, quant modes).
G2 bit-parity (sampled, real bytes):
  - expert byte-identity: fused gate_up/down slices == source code/scale bytes
  - expert dequant crosscheck vs an independent numpy E2M1*E4M3 reference
  - DQ8 triples == mx.quantize(source bf16) (array_equal)
  - absorption: embed_q/unembed_out == quantize(fold(source kv_b))
  - ts sidecars == source weight_scale_2 scalars (f32)

Usage: gate_offline.py --src <hf_src> --out <converted_dir>
Exits non-zero on any failure; last line 'OFFLINE GATES: ALL PASS' on success.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.environ.get("OMLX_REPO", os.getcwd()))
from omlx.tools import oqnvfp4_glm_convert as conv  # noqa: E402

conv._ensure_mlx()

_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def fp4_vals(codes_u8):
    lo, hi = codes_u8 & 0x0F, codes_u8 >> 4
    nib = np.stack([lo, hi], axis=-1).reshape(codes_u8.shape[0], -1)
    return np.where(nib & 0x8, -1.0, 1.0).astype(np.float32) * _E2M1[nib & 0x7]


def e4m3_vals(b_u8):
    b = b_u8.astype(np.uint32)
    sign = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
    exp, mant = (b >> 3) & 0xF, (b & 0x7).astype(np.float32)
    val = np.where(
        exp == 0,
        (2.0**-6) * (mant / 8.0),
        (2.0 ** (exp.astype(np.float32) - 7.0)) * (1.0 + mant / 8.0),
    )
    return sign * val


def shard_names(dir_path: Path) -> dict:
    """name -> (shard, dtype, shape) from headers only (no tensor loads)."""
    names = {}
    for f in sorted(dir_path.glob("*.safetensors")):
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, meta in hdr.items():
            if k != "__metadata__":
                names[k] = (f, meta["dtype"], tuple(meta["shape"]))
    return names


FAILS = []


def check(label, ok):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        FAILS.append(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with (args.out / "config.json").open() as f:
        cfg = json.load(f)
    with (args.src / "config.json").open() as f:
        src_cfg = json.load(f)
    dims = conv.read_dims(src_cfg)

    out_names = shard_names(args.out)
    src_index = conv.TensorIndex(args.src)

    emitted = sorted(
        {
            int(k.split(".")[2])
            for k in out_names
            if k.startswith("model.layers.")
        }
    )
    n_layers = len(emitted)
    print(f"emitted layers: {n_layers} ({emitted[0]}..{emitted[-1]})")

    # ---------------- G1: exact census ----------------
    expected = set()

    def q8(base):
        expected.update(f"{base}.{s}" for s in ("weight", "scales", "biases"))

    def nv4(base):
        expected.update(f"{base}.{s}" for s in ("weight", "scales"))

    q8("model.embed_tokens")
    q8("lm_head")
    expected.add("model.norm.weight")
    shared_mode = cfg["quantization"].get(
        f"model.layers.{max(emitted)}.mlp.shared_experts.up_proj", {}
    ).get("mode", "affine")
    for l in emitted:
        p, attn = f"model.layers.{l}", f"model.layers.{l}.self_attn"
        expected.update(
            (
                f"{p}.input_layernorm.weight",
                f"{p}.post_attention_layernorm.weight",
                f"{attn}.q_a_layernorm.weight",
                f"{attn}.kv_a_layernorm.weight",
            )
        )
        for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            q8(f"{attn}.{proj}")
        q8(f"{attn}.embed_q")
        q8(f"{attn}.unembed_out")
        if dims.indexer_types[l] == "full":
            expected.update(
                f"{attn}.indexer.{s}"
                for s in (
                    "wq_b.weight",
                    "wk.weight",
                    "weights_proj.weight",
                    "k_norm.weight",
                    "k_norm.bias",
                )
            )
        if l < dims.first_k_dense_replace:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                q8(f"{p}.mlp.{proj}")
        else:
            expected.update(
                (
                    f"{p}.mlp.gate.weight",
                    f"{p}.mlp.gate.e_score_correction_bias",
                    f"{p}.mlp.switch_mlp.gate_up_ts",
                    f"{p}.mlp.switch_mlp.down_ts",
                )
            )
            nv4(f"{p}.mlp.switch_mlp.gate_up_proj")
            nv4(f"{p}.mlp.switch_mlp.down_proj")
            for proj in ("gate_proj", "up_proj", "down_proj"):
                (nv4 if shared_mode == "nvfp4" else q8)(
                    f"{p}.mlp.shared_experts.{proj}"
                )

    extra = set(out_names) - expected
    missing = expected - set(out_names)
    check(f"G1 census exact (n={len(expected)})", not extra and not missing)
    for s in sorted(extra)[:8]:
        print(f"   stray: {s}")
    for s in sorted(missing)[:8]:
        print(f"   missing: {s}")
    check(
        "G1 no kv_b/input_scale/experts.N/MTP",
        not any(
            ("kv_b_proj" in k)
            or ("input_scale" in k)
            or (".mlp.experts." in k)
            or k.startswith(f"model.layers.{dims.num_hidden_layers}")
            for k in out_names
        ),
    )
    check(
        "G1 config flags",
        cfg.get("omlx_moe_nvfp4_ts") is True
        and "num_nextn_predict_layers" not in cfg
        and cfg["quantization"] == cfg["quantization_config"],
    )
    # every quantized module on disk has a config entry and vice versa
    disk_q_modules = {
        k[: -len(".scales")] for k in out_names if k.endswith(".scales")
    }
    cfg_modules = {
        k for k, v in cfg["quantization"].items() if isinstance(v, dict)
    }
    check(
        "G1 quant-dict 1:1 with disk",
        disk_q_modules == cfg_modules,
    )
    for s in sorted(disk_q_modules ^ cfg_modules)[:8]:
        print(f"   quant-dict mismatch: {s}")

    # ---------------- G2: sampled bit-parity ----------------
    def load_out(name):
        f, _, _ = out_names[name]
        return mx.load(str(f))[name]

    moe_layers = [l for l in emitted if l >= dims.first_k_dense_replace]
    sample_layer = moe_layers[len(moe_layers) // 2]
    moe = f"model.layers.{sample_layer}.mlp"
    h = dims.moe_intermediate_size
    gate_up_w = np.array(load_out(f"{moe}.switch_mlp.gate_up_proj.weight"))
    gate_up_s = np.array(load_out(f"{moe}.switch_mlp.gate_up_proj.scales"))
    down_w = np.array(load_out(f"{moe}.switch_mlp.down_proj.weight"))
    down_s = np.array(load_out(f"{moe}.switch_mlp.down_proj.scales"))
    gate_up_ts = load_out(f"{moe}.switch_mlp.gate_up_ts")
    down_ts = load_out(f"{moe}.switch_mlp.down_ts")

    for e in (0, dims.n_routed_experts // 2, dims.n_routed_experts - 1):
        base = f"{moe}.experts.{e}"
        g_codes = np.array(src_index.load(f"{base}.gate_proj.weight"))
        u_codes = np.array(src_index.load(f"{base}.up_proj.weight"))
        d_codes = np.array(src_index.load(f"{base}.down_proj.weight"))
        g_sc = np.array(src_index.load(f"{base}.gate_proj.weight_scale"))
        u_sc = np.array(src_index.load(f"{base}.up_proj.weight_scale"))
        ok = (
            (gate_up_w[e, :h].view(np.uint8).reshape(g_codes.shape) == g_codes).all()
            and (gate_up_w[e, h:].view(np.uint8).reshape(u_codes.shape) == u_codes).all()
            and (down_w[e].view(np.uint8).reshape(d_codes.shape) == d_codes).all()
            and (gate_up_s[e, :h] == g_sc).all()
            and (gate_up_s[e, h:] == u_sc).all()
        )
        check(f"G2 expert byte-identity L{sample_layer} e{e}", ok)
        for proj, col_or_down in (("gate_proj", 0), ("up_proj", 1), ("down_proj", -1)):
            src_ts = float(src_index.load(f"{base}.{proj}.weight_scale_2").item())
            got = float(
                down_ts[e].item() if col_or_down < 0 else gate_up_ts[e, col_or_down].item()
            )
            check(f"G2 ts exact L{sample_layer} e{e} {proj}", got == src_ts)

    # dequant crosscheck vs independent numpy reference (expert 0 gate rows)
    base = f"{moe}.experts.0.gate_proj"
    codes = np.array(src_index.load(f"{base}.weight"))[:64]
    scales = np.array(src_index.load(f"{base}.weight_scale"))[:64]
    w_mx, s_mx = conv.repack_nvfp4(mx.array(codes), mx.array(scales))
    deq = np.array(
        mx.dequantize(w_mx, s_mx, None, group_size=16, bits=4, mode="nvfp4").astype(
            mx.float32
        )
    )
    ref = fp4_vals(codes) * np.repeat(e4m3_vals(scales), 16, axis=1)
    finite = np.isfinite(ref)
    check(
        "G2 dequant == numpy ModelOpt reference (max|diff|=0)",
        bool(finite.all()) and float(np.abs(deq - ref).max()) == 0.0,
    )

    # DQ8 parity samples
    dq8_samples = [
        f"model.layers.{emitted[0]}.self_attn.q_a_proj",
        f"model.layers.{emitted[-1]}.self_attn.o_proj",
        "model.embed_tokens",
    ]
    if moe_layers:
        dq8_samples.append(f"model.layers.{moe_layers[0]}.mlp.shared_experts.down_proj")
    for name in dq8_samples:
        spec = cfg["quantization"].get(name, {})
        if spec.get("mode") != "affine":
            continue
        src_w = src_index.load(f"{name}.weight")
        qw, s, b = mx.quantize(src_w, group_size=64, bits=8, mode="affine")
        ok = (
            mx.array_equal(load_out(f"{name}.weight"), qw)
            and mx.array_equal(load_out(f"{name}.scales"), s)
            and mx.array_equal(load_out(f"{name}.biases"), b)
        )
        check(f"G2 DQ8 parity {name}", bool(ok))

    # absorption parity
    for l in (emitted[0], emitted[-1]):
        attn = f"model.layers.{l}.self_attn"
        wk, wv = conv.fold_kv_b(src_index.load(f"{attn}.kv_b_proj.weight"), dims)
        for base, w in ((f"{attn}.embed_q", wk), (f"{attn}.unembed_out", wv)):
            qw, s, b = mx.quantize(w, group_size=64, bits=8, mode="affine")
            ok = (
                mx.array_equal(load_out(f"{base}.weight"), qw)
                and mx.array_equal(load_out(f"{base}.scales"), s)
                and mx.array_equal(load_out(f"{base}.biases"), b)
            )
            check(f"G2 absorption parity {base}", bool(ok))

    if FAILS:
        print(f"OFFLINE GATES: {len(FAILS)} FAILURE(S)")
        sys.exit(1)
    print("OFFLINE GATES: ALL PASS")


if __name__ == "__main__":
    main()
