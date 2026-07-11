"""Chained second-draft acceptance: head drafts t+2 from its OWN hidden (GLM-style chain)."""
import os, sys, importlib.util
import mlx.core as mx
import mlx.nn as nn
mx.set_wired_limit(506 * 1024**3)
spec = importlib.util.spec_from_file_location(
    "mlx_lm.models.nemotron_h_puzzle",
    os.path.join(os.environ.get("OMLX_REPO", os.getcwd()), "omlx/patches/nemotron_h_puzzle/nemotron_h_puzzle_model.py"))
mod = importlib.util.module_from_spec(spec); mod.__package__ = "mlx_lm.models"
sys.modules["mlx_lm.models.nemotron_h_puzzle"] = mod; spec.loader.exec_module(mod)
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.nemotron_h import NemotronHBlock
from mlx_lm.models.switch_layers import SwitchLinear
from dataclasses import replace

model, tok = load(os.environ["OMLX_PUZZLE_MODEL"])
args = model.args
class MtpHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        b0 = NemotronHBlock(args, "*")
        b0.eh_proj = nn.Linear(2*args.hidden_size, args.hidden_size, bias=False)
        b0.enorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        b0.hnorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        b1 = NemotronHBlock(replace(args, moe_intermediate_size=2688, num_experts_per_tok=22), "E")
        b1.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.layers = [b0, b1]
    def run(self, hidden, tok_ids, embed):
        b0, b1 = self.layers
        e = b0.enorm(embed(tok_ids)); hh = b0.hnorm(hidden)
        x = b0.eh_proj(mx.concatenate([e, hh], axis=-1))
        x = b0(x, mask=create_attention_mask(x, None), cache=None)
        x = b1(x)
        return x, b1.final_layernorm(x)   # (pre-final chainable, normed for lm_head)
head = MtpHead(args)
def qpred(path, m):
    if isinstance(m, SwitchLinear): return {"group_size": 64, "bits": 4}
    if not isinstance(m, nn.Linear): return False
    if any(t in path for t in ("eh_proj","q_proj","o_proj","shared_experts","latent_proj")):
        return {"group_size": 64, "bits": 8}
    return False
nn.quantize(head, group_size=64, bits=4, class_predicate=qpred)
sc = mx.load(str(__import__('pathlib').Path.home() / ".omlx/mtp_sidecars/puzzle75_mtp_oq48.safetensors"))
head.load_weights([(k[len("mtp."):], v) for k, v in sc.items()], strict=True)
head.eval(); mx.eval(head.parameters())
bb = model.backbone
def trunk_forward(inputs, cache):
    h = bb.embeddings(inputs)
    am = create_attention_mask(h, cache[bb.fa_idx]); sm = create_ssm_mask(h, cache[bb.ssm_idx])
    ci = 0
    for layer in bb.layers:
        c = None
        if layer.block_type in ("M","*"): c = cache[ci]; ci += 1
        h = layer(h, mask=(am if layer.block_type=="*" else sm), cache=c)
    return h, model.lm_head(bb.norm_f(h))
PROMPTS = {
 "code": "Write a complete Python implementation of Dijkstra's shortest path algorithm with a binary heap, including docstrings.",
 "math": "A warehouse has 1240 boxes. Each truck carries 85 boxes per trip and makes 3 trips per day. How many full days until fewer than 100 boxes remain? Work through this step by step.",
}
N = 480
for name, prompt in PROMPTS.items():
    ids = tok.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)
    cache = model.make_cache()
    h, lg = trunk_forward(mx.array([ids]), cache)
    hiddens=[h[:,-1:]]; toks=[mx.argmax(lg[:,-1:],axis=-1).item()]
    for _ in range(N-1):
        h, lg = trunk_forward(mx.array([[toks[-1]]]), cache)
        hiddens.append(h[:,-1:]); toks.append(mx.argmax(lg[:,-1:],axis=-1).item())
    H = bb.norm_f(mx.concatenate(hiddens, axis=1)); T = mx.array([toks])
    # step 1: draft t+2 from (h_i, t_{i+1})
    x_pre, x_norm = head.run(H[:, :-3], T[:, 1:-2], bb.embeddings)
    d1 = mx.argmax(model.lm_head(x_norm), axis=-1)
    hit1 = (d1 == T[:, 2:-1])
    # step 2 chained: h' = head's own hidden; two variants: pre-final vs normed; token = TRUE t_{i+2}
    for hname, Hc in [("chain-prenorm", x_pre), ("chain-postnorm", x_norm)]:
        _, y_norm = head.run(Hc, T[:, 2:-1], bb.embeddings)
        d2 = mx.argmax(model.lm_head(y_norm), axis=-1)
        hit2 = (d2 == T[:, 3:])
        a1 = mx.mean(hit1).item()
        a2_cond = (mx.sum(hit1 & hit2) / mx.maximum(mx.sum(hit1), 1)).item()
        print(f"[{name}][{hname}] a1={a1:.3f}  a2|1={a2_cond:.3f}  a1*a2={a1*a2_cond:.3f}")
