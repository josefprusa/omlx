#!/usr/bin/env python3
# omlx-perf skill · MiniMax-M3 "DQ8 bake" offline gate · added 2026-07-06 (EXP-065 live-leg prep)
#
# FINDING this gate proves: the oQNVFP4 (and fs5) source ALREADY ships an affine8-gs64 text
# shell, so a "DQ8 bake" of the dense shell is a NO-OP — the M3 converter (oqnvfp4_convert.py)
# quantizes q/k/v/o + index + dense-mlp + shared + lm_head + embed to affine8-gs64 BY
# CONSTRUCTION (there is no --dq8 flag and no separable bf16-shell build for M3, unlike Ultra).
#
# G1 (inward census): every intended DQ8 target is ALREADY an affine8-gs64 triple
#   (U32 .weight + bf16 .scales + bf16 .biases), matching config quantization default; and the
#   dq8-eligible BF16 *weight* left in the TEXT path == 0. Experts stay nvfp4 (U32/U8), ts sidecars present.
# G2 (idempotency parity): for sampled targets, dequantize the stored affine8 triple -> bf16 and
#   re-quantize with mx.quantize(gs64,8,affine); report how many packed words differ. Zero diff ==
#   the bake is bit-idempotent (a true no-op); nonzero == a re-bake would only re-tighten scales
#   (lossy-on-lossy, still pointless).
#
# Usage: .venv/bin/python scripts/m3_dq8_offline_gate.py [<model_dir>]
#   default model_dir = the plain oQNVFP4 source.
import json, glob, struct, os, sys, collections
import mlx.core as mx

SRC = sys.argv[1] if len(sys.argv) > 1 else os.environ["OMLX_MINIMAX_MODEL"]
GS, BITS, MODE = 64, 8, "affine"

# ---- header-only dtype/shape map + weight_map(name->shard) ----
def headers(md):
    h = {}
    for f in sorted(glob.glob(os.path.join(md, "*.safetensors"))):
        with open(f, "rb") as fh:
            (hl,) = struct.unpack("<Q", fh.read(8)); H = json.loads(fh.read(hl))
        for n, m in H.items():
            if n == "__metadata__": continue
            h[n] = (m["dtype"], tuple(m["shape"]))
    return h

H = headers(SRC)
wmap = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
cfg = json.load(open(os.path.join(SRC, "config.json")))
q = cfg.get("quantization", {}) or cfg.get("quantization_config", {})
default = {k: v for k, v in q.items() if not isinstance(v, dict)}
permod = {k: v for k, v in q.items() if isinstance(v, dict)}

# ---- intended DQ8 dense-shell targets in the TEXT path (per task scope) ----
targets = []
for L in range(60):
    p = f"language_model.model.layers.{L}.self_attn"
    targets += [f"{p}.q_proj", f"{p}.o_proj"]          # attention q/o (task stage C analog)
for L in range(3):                                     # first 3 dense layers' MLP
    p = f"language_model.model.layers.{L}.mlp"
    targets += [f"{p}.gate_proj", f"{p}.up_proj", f"{p}.down_proj"]
targets += ["language_model.lm_head"]                  # lm_head (task: only if bf16)

def is_affine8_gs64(base):
    w, s, b = H.get(base + ".weight"), H.get(base + ".scales"), H.get(base + ".biases")
    if not w: return None, "no .weight"
    if w[0] == "BF16": return False, "BF16 weight (dq8-ELIGIBLE, not yet baked)"
    if not (s and b): return False, f"weight={w[0]} but missing scales/biases"
    if w[0] != "U32": return False, f"weight dtype {w[0]}"
    if s[0] not in ("BF16", "F16") or b[0] not in ("BF16", "F16"):
        return False, f"scales/biases dtype {s[0]}/{b[0]}"
    out, win = w[1]; so, sin = s[1]
    if so != out or win * (32 // BITS) != sin * GS:
        return False, f"layout out={out},win={win},sin={sin} not affine8-gs64"
    return True, "affine8-gs64"

# ================= G1: inward census =================
already, eligible, problems = [], [], []
for t in targets:
    ok, why = is_affine8_gs64(t)
    if ok is True: already.append(t)
    elif why.startswith("BF16 weight"): eligible.append(t)
    else: problems.append((t, why))

# any remaining dq8-eligible BF16 *weight* in the TEXT path at all (broad sweep, excludes norms/router/experts)?
def cls(n):
    s = n.lower()
    if "vision_tower" in s or "multi_modal" in s or "patch_merge" in s: return "vision"
    if "norm" in s: return "norm"
    if "gate.weight" in s: return "router"
    if "switch_mlp" in s or ".experts." in s: return "expert"
    if any(x in s for x in ("q_proj", "k_proj", "v_proj", "o_proj", "index_q", "index_k",
                            "gate_proj", "up_proj", "down_proj", "lm_head", "embed")): return "text_shell"
    return "other"
bf16_text_weight = [n for n, (dt, sh) in H.items()
                    if dt in ("BF16", "F16") and n.endswith(".weight") and cls(n) == "text_shell"]
bf16_vision_weight = sum(1 for n, (dt, sh) in H.items()
                         if dt in ("BF16", "F16") and n.endswith(".weight") and cls(n) == "vision")
n_expert_w = sum(1 for n in H if n.endswith(".weight") and "switch_mlp" in n)   # nvfp4 stacks
n_ts = sum(1 for n in H if n.endswith("gate_up_ts") or n.endswith("down_ts"))

print("=" * 72)
print(f"G1 INWARD CENSUS · {os.path.basename(SRC)}")
print(f"  intended text-shell DQ8 targets ...... {len(targets)}")
print(f"  ALREADY affine8-gs64 (baked) ......... {len(already)}")
print(f"  still BF16 weight (dq8-eligible) ..... {len(eligible)}")
print(f"  layout/other problems ................ {len(problems)}")
print(f"  config default quant ................. {default}")
print(f"  per-module quant entries ............. {len(permod)} (expect nvfp4 experts)")
print(f"  BF16 .weight left in TEXT shell ...... {len(bf16_text_weight)}   <-- dq8-eligible text weight")
print(f"  BF16 .weight in VISION tower ......... {bf16_vision_weight}   (out of text decode hot path)")
print(f"  nvfp4 expert weight stacks ........... {n_expert_w}   ts sidecars: {n_ts}")
for t, why in problems[:8]: print("     PROBLEM:", t, "->", why)
g1 = (len(eligible) == 0 and len(problems) == 0 and len(bf16_text_weight) == 0
      and default.get("bits") == 8 and default.get("group_size") == 64 and default.get("mode") == "affine")
print(f"  G1 verdict: {'PASS (shell already fully affine8-gs64; bake is a no-op)' if g1 else 'REVIEW'}")

# ================= G2: idempotency parity =================
def load(name):
    return mx.load(os.path.join(SRC, wmap[name]))[name]

samples = [
    "language_model.model.layers.0.self_attn.q_proj",
    "language_model.model.layers.0.mlp.down_proj",
    "language_model.model.layers.29.self_attn.o_proj",
    "language_model.model.layers.59.self_attn.q_proj",
    "language_model.lm_head",
]
print("=" * 72)
print("G2 IDEMPOTENCY PARITY (dequant stored affine8 -> re-quantize affine8-gs64 -> compare)")
g2 = True
for base in samples:
    if base not in [t for t in targets] and base != "language_model.lm_head":
        continue
    w = load(base + ".weight"); s = load(base + ".scales"); b = load(base + ".biases")
    deq = mx.dequantize(w, s, b, group_size=GS, bits=BITS, mode=MODE)          # bf16 effective weight
    rq, rs, rb = mx.quantize(deq.astype(mx.bfloat16), group_size=GS, bits=BITS, mode=MODE)
    mx.eval(deq, rq, rs, rb)
    finite = bool(mx.all(mx.isfinite(deq)).item())
    weq = bool(mx.array_equal(w, rq).item())
    seq = bool(mx.array_equal(s, rs).item())
    beq = bool(mx.array_equal(b, rb).item())
    diff_words = int(mx.sum(w != rq).item())
    total_words = int(w.size)
    print(f"  {base.split('.',2)[-1]:42s} finite={finite} weight_equal={weq} "
          f"scales_eq={seq} biases_eq={beq}  diff_words={diff_words}/{total_words}")
    g2 = g2 and finite  # finiteness is the hard gate; idempotency is reported, not required
print(f"  G2 verdict: {'PASS (stored triples are valid affine8-gs64)' if g2 else 'FAIL (non-finite dequant)'}")
print("=" * 72)
print("OFFLINE GATES:", "ALL PASS" if (g1 and g2) else "REVIEW")
sys.exit(0 if (g1 and g2) else 1)
