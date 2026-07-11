"""First-principles decode decomposition for Puzzle oQ48 (offline, box must be idle).

Per-layer-type in-stream cost (chained calls, one eval) vs byte-physics, plus
whole-token eager and pipelined rates. Law 6: never time an async submit as one
wall number -> every timed region is a chained loop drained by ONE mx.eval.
"""
import os, time, sys
import mlx.core as mx
sys.path.insert(0, os.environ.get("OMLX_REPO", os.getcwd()))

mx.set_wired_limit(506 * 1024**3)  # profiling.md law: 65x page-fault illusion otherwise

from omlx.utils.model_loading import maybe_apply_pre_load_patches
from pathlib import Path
P = Path(os.environ["OMLX_PUZZLE_MODEL"])
maybe_apply_pre_load_patches(P)
from mlx_lm import load
model, tok = load(str(P))

BW = 819e9  # bytes/s

def qbytes(mod):
    n = 0
    for _, v in mod.parameters().items():
        pass
    # walk leaf arrays
    def walk(m):
        nonlocal n
        for k, v in m.items():
            if isinstance(v, mx.array):
                n += v.nbytes
            elif isinstance(v, dict):
                walk(v)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        walk(it)
    walk(mod.parameters())
    return n

layers = model.backbone.layers
pattern = [l.block_type for l in layers]
print("pattern counts:", {c: pattern.count(c) for c in set(pattern)})

cache = model.make_cache()
# prefill a few tokens so states/KV exist
ids = mx.array([tok.encode("The observatory logged another calibration pass at dawn.")])
logits = model(ids, cache=cache)
mx.eval(logits)

h = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
mx.eval(h)

def bench_block(fn, iters=60, warmup=8):
    x = h
    for _ in range(warmup):
        x = fn(x)
    mx.eval(x)
    t0 = time.perf_counter()
    x = h
    for _ in range(iters):
        x = fn(x)
    mx.eval(x)
    return (time.perf_counter() - t0) / iters * 1e6  # us

# map cache indices (only M and * consume cache slots)
cache_idx, ci = {}, 0
for i, l in enumerate(layers):
    if l.block_type in ("M", "*"):
        cache_idx[i] = ci
        ci += 1

results = {}
# one mamba layer
i_m = pattern.index("M")
c = cache[cache_idx[i_m]]
results["mamba(x1)"] = bench_block(lambda x: layers[i_m](x, mask=None, cache=c))
# small moe (inter 1280) and big moe (inter 2688)
import json
cfg = json.load(open(P / "config.json"))
bcs = cfg["block_configs"]
i_small = next(i for i, b in enumerate(bcs) if b["block_type"] == "moe" and b["moe_intermediate_size"] == 1280)
i_big   = next(i for i, b in enumerate(bcs) if b["block_type"] == "moe" and b["moe_intermediate_size"] == 2688)
results[f"moe_small(1280,k{bcs[i_small]['num_experts_per_tok']})"] = bench_block(lambda x: layers[i_small](x, mask=None, cache=None))
results[f"moe_big(2688,k{bcs[i_big]['num_experts_per_tok']})"]   = bench_block(lambda x: layers[i_big](x, mask=None, cache=None))
# attention
i_a = pattern.index("*")
ca = cache[cache_idx[i_a]]
results["attention(x1)"] = bench_block(lambda x: layers[i_a](x, mask=None, cache=ca))
# lm_head (dependency trick keeps the chain honest)
def lmh(x):
    y = model.lm_head(model.backbone.norm_f(x))
    return x + (y[..., :4096] * mx.array(0.0, dtype=x.dtype))
results["norm_f+lm_head"] = bench_block(lmh, iters=40)

# sub-parts of one MoE layer to localize
moe = layers[i_big].mixer
def moe_gate_only(x):
    inds, scores = moe.gate(x)
    return x + (scores.astype(x.dtype).sum() * 0)
results["moe.gate_only"] = bench_block(moe_gate_only)
def moe_shared_only(x):
    return x + moe.shared_experts(x)
results["moe.shared_only"] = bench_block(moe_shared_only)
def moe_latent_round(x):
    return x + moe.fc2_latent_proj(moe.fc1_latent_proj(x))
results["moe.latent_pair_only"] = bench_block(moe_latent_round)

# sub-parts of one mamba layer
mm = layers[i_m].mixer
def mamba_projs_only(x):
    p = mm.in_proj(x)
    return x + mm.out_proj(p[..., :mm.intermediate_size])
results["mamba.in+out_proj_only"] = bench_block(mamba_projs_only)

for k, v in results.items():
    print(f"{k:32s} {v:9.1f} us")

n_m, n_e, n_a = pattern.count("M"), pattern.count("E"), pattern.count("*")
# scale moe cost by width mix: use small/big as anchors, weight by config census
from collections import Counter
mix = Counter((b["moe_intermediate_size"]) for b in bcs if b["block_type"] == "moe")
small_like = sum(v for kk, v in mix.items() if kk <= 1792)
big_like = sum(v for kk, v in mix.items() if kk > 1792)
moe_avg = (results[[k for k in results if k.startswith("moe_small")][0]] * small_like
           + results[[k for k in results if k.startswith("moe_big")][0]] * big_like) / (small_like + big_like)
total = (n_m * results["mamba(x1)"] + n_e * moe_avg + n_a * results["attention(x1)"]
         + results["norm_f+lm_head"])
print(f"\nassembled token estimate: {total/1000:.2f} ms  ({1e6/total*1:.1f} tok/s)")

# whole-token measured two ways
y = mx.array([[tok.encode('a')[0]]])
def token_step(y):
    lg = model(y, cache=cache)
    return mx.argmax(lg[:, -1:], axis=-1)
# eager (host-serialized upper bound)
for _ in range(5):
    y = token_step(y); mx.eval(y)
t0 = time.perf_counter()
for _ in range(60):
    y = token_step(y); mx.eval(y)
eager = (time.perf_counter() - t0) / 60 * 1000
print(f"whole-token eager:      {eager:.2f} ms ({1000/eager:.1f} tok/s)")
# one-ahead pipeline (mlx_lm-style async)
ys = [y]
for _ in range(3):
    ys.append(token_step(ys[-1])); mx.async_eval(ys[-1])
mx.eval(ys[-1])
t0 = time.perf_counter()
prev = ys[-1]
for _ in range(60):
    nxt = token_step(prev)
    mx.async_eval(nxt)
    mx.eval(prev)
    prev = nxt
mx.eval(prev)
pipe = (time.perf_counter() - t0) / 60 * 1000
print(f"whole-token one-ahead:  {pipe:.2f} ms ({1000/pipe:.1f} tok/s)")
print("PID for sampling:", __import__('os').getpid())
# long tail decode so /usr/bin/sample can catch us
t_end = time.time() + 25
n = 0
while time.time() < t_end:
    prev = token_step(prev); mx.eval(prev); n += 1
print(f"tail decoded {n} tokens (eager)")
