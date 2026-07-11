#!/usr/bin/env python3
# omlx-perf skill · dq8 cost model · added 2026-07-06 (EXP-065) · FIXED 2026-07-06 · header-only
# Is a DQ8 dense-shell bake worth it for a given model? Header-only safetensors census —
# no weight load, no GPU, no server contact.
#
# TWO TRAPS this script exists to avoid (both bit EXP-065's first pass — see dead-levers.md):
#   1. `.scales` / `.biases` sidecars are BF16 by nature. They are NOT bakeable weight — they are
#      the OUTPUT of an existing affine/nvfp4 quant. Counting them as "bf16 weight" fakes a shell.
#   2. A VL model's vision tower (CLIP) + multimodal projector are BF16 but OFF the text decode
#      hot path. dq8-ing them does nothing for text tok/s. Split them out.
# So: dq8-eligible = BF16 *.weight on the TEXT path, class in {attn_qo, dense_mlp, lmhead},
#     excluding sidecars, vision/mm, routers, norms, embeddings, k/v (M0 LOSS).
#
# Decision rule: dq8 pays only if that eligible bf16 is a LARGE fraction of per-token reads AND
# decode is bandwidth-bound (profiling.md). A converter that already emits an affine8 shell BY
# CONSTRUCTION (like omlx's M3 oqnvfp4_convert.py) leaves ~0 eligible -> the bake is a no-op.
# Usage: .venv/bin/python scripts/dq8_costmodel.py <model_dir> [<model_dir> ...]
import json, os, glob, struct, sys, collections, re

DT = {"BF16":2,"F16":2,"F32":4,"F64":8,"F8_E4M3":1,"F8_E5M2":1,
      "U8":1,"I8":1,"U16":2,"I16":2,"U32":4,"I32":4,"U64":8,"I64":8,"BOOL":1}

def is_sidecar(n):
    return (n.endswith(".scales") or n.endswith(".biases") or "weight_scale" in n
            or n.endswith("_ts") or ".scales" in n or ".biases" in n)

def is_vision(n):
    s=n.lower()
    return any(k in s for k in ("vision","visual","vit","clip","image","pixel",
                                "patch_embed","patch_merge","multi_modal","mm_projector","img_","vis_"))

def classify(n):
    s=n.lower()
    if is_sidecar(n): return "sidecar"
    if is_vision(n): return "vision/mm"
    if "embed" in s or "tok_embeddings" in s: return "embed"
    if "lm_head" in s or s=="output.weight": return "lmhead"
    if "switch" in s or "experts" in s: return "expert"
    if "shared" in s: return "shared_expert"
    if "gate.weight" in s or "router" in s or "e_score" in s: return "router"
    if "norm" in s: return "norm"
    if "k_proj" in s or "v_proj" in s or "index_k" in s or "index_v" in s: return "attn_kv"
    if "q_proj" in s or "o_proj" in s or "out_proj" in s or "index_q" in s or "qkv" in s: return "attn_qo"
    if "up_proj" in s or "down_proj" in s or "gate_proj" in s or "gate_up" in s or "mlp" in s: return "dense_mlp"
    return "other"

def scan(model_dir):
    files=sorted(glob.glob(os.path.join(model_dir,"*.safetensors")))
    if not files:
        print("NO SAFETENSORS in", model_dir); return
    by_dtype=collections.Counter(); cls_tot=collections.Counter()
    elig_bf16=collections.Counter()   # bf16 weight in dq8-target classes, text only
    ntensors=0
    for f in files:
        with open(f,"rb") as fh:
            (hlen,)=struct.unpack("<Q", fh.read(8))
            hdr=json.loads(fh.read(hlen))
        for name,meta in hdr.items():
            if name=="__metadata__": continue
            dt=meta["dtype"]; shape=meta.get("shape",[])
            numel=1
            for x in shape: numel*=x
            b=numel*DT.get(dt,0)
            by_dtype[dt]+=b
            c=classify(name); cls_tot[c]+=b
            if dt in ("BF16","F16") and name.endswith(".weight") and c in ("attn_qo","dense_mlp","lmhead"):
                elig_bf16[c]+=b
            ntensors+=1
    total=sum(by_dtype.values()); G=1024**3
    print(f"\n=== {os.path.basename(model_dir)}  ({ntensors} tensors, {len(files)} shards, {total/G:.1f} GiB) ===")
    print("by dtype:")
    for dt,b in by_dtype.most_common():
        print(f"  {dt:10s} {b/G:8.2f} GiB  {100*b/total:5.1f}%")
    print("by class (all dtypes):")
    for c,b in cls_tot.most_common():
        if b/G>0.05: print(f"  {c:14s} {b/G:8.2f} GiB")
    elig=sum(elig_bf16.values())
    print(f"\nDQ8-ELIGIBLE bf16 text weight (attn_qo+dense_mlp+lmhead, sidecars+vision EXCLUDED): {elig/G:.2f} GiB")
    for c,b in elig_bf16.most_common(): print(f"    {c:12s} {b/G:.3f} GiB")
    if elig/G < 0.5:
        print("  => ~0 eligible: the shell is ALREADY quantized (or is a MoE with 4-bit experts).")
        print("     A dq8 bake is a NO-OP here. Do NOT build it (EXP-065).")
    else:
        print(f"  => affine8 would cut ~{elig/2/G:.2f} GiB whole-model. Only worth it if decode is")
        print(f"     bandwidth-bound (cross-check a measured ceiling, EXP-053 method) — else dispatch dominates.")

if len(sys.argv)<2:
    print("usage: .venv/bin/python scripts/dq8_costmodel.py <model_dir> [<model_dir> ...]"); sys.exit(1)
for md in sys.argv[1:]:
    scan(md)
