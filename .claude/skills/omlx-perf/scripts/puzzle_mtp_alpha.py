"""Teacher-forced MTP acceptance (alpha1) for Puzzle-75B oQ48.
Doubles as the wiring gate: wrong dataflow => alpha ~ 1/vocab."""
import os, sys, importlib.util, time
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
from dataclasses import replace

model, tok = load(os.environ["OMLX_PUZZLE_MODEL"])
args = model.args

# ---- build MTP head ----
class MtpHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        b0 = NemotronHBlock(args, "*")
        b0.eh_proj = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        b0.enorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        b0.hnorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        b1 = NemotronHBlock(replace(args, moe_intermediate_size=2688, num_experts_per_tok=22), "E")
        b1.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.layers = [b0, b1]
    def __call__(self, hidden, tok_ids, embed, kv_cache=None):
        b0, b1 = self.layers
        e = b0.enorm(embed(tok_ids))
        hh = b0.hnorm(hidden)
        x = b0.eh_proj(mx.concatenate([e, hh], axis=-1))
        mask = create_attention_mask(x, kv_cache)
        x = b0(x, mask=mask, cache=kv_cache)
        x = b1(x)
        return b1.final_layernorm(x)

head = MtpHead(args)
from mlx_lm.models.switch_layers import SwitchLinear
def qpred(path, m):
    if isinstance(m, SwitchLinear):
        return {"group_size": 64, "bits": 4}
    if not isinstance(m, nn.Linear):
        return False
    if any(t in path for t in ("eh_proj","q_proj","o_proj","shared_experts","latent_proj")):
        return {"group_size": 64, "bits": 8}
    return False
nn.quantize(head, group_size=64, bits=4, class_predicate=qpred)
sc = mx.load(str(__import__('pathlib').Path.home() / ".omlx/mtp_sidecars/puzzle75_mtp_oq48.safetensors"))
head.load_weights([(k[len("mtp."):], v) for k, v in sc.items()], strict=True)
head.eval(); mx.eval(head.parameters())
print("MTP head loaded strict OK")

# ---- trunk step that also returns pre-norm hidden ----
bb = model.backbone
def trunk_forward(inputs, cache):
    h = bb.embeddings(inputs)
    attn_mask = create_attention_mask(h, cache[bb.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[bb.ssm_idx])
    ci = 0
    for layer in bb.layers:
        c = None
        if layer.block_type in ("M", "*"):
            c = cache[ci]; ci += 1
        m = attn_mask if layer.block_type == "*" else ssm_mask
        h = layer(h, mask=m, cache=c)
    return h, model.lm_head(bb.norm_f(h))

PROMPTS = {
 "code":  "Write a complete Python implementation of Dijkstra's shortest path algorithm with a binary heap, including docstrings.",
 "math":  "A warehouse has 1240 boxes. Each truck carries 85 boxes per trip and makes 3 trips per day. How many full days until fewer than 100 boxes remain? Work through this step by step.",
 "prose": "Write a reflective essay about how mountain villages change when a new railway line arrives.",
}
N = 640
for name, prompt in PROMPTS.items():
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    cache = model.make_cache()
    h, lg = trunk_forward(mx.array([ids]), cache)
    hiddens = [h[:, -1:]]
    toks = [mx.argmax(lg[:, -1:], axis=-1).item()]
    for _ in range(N - 1):
        h, lg = trunk_forward(mx.array([[toks[-1]]]), cache)
        hiddens.append(h[:, -1:])
        toks.append(mx.argmax(lg[:, -1:], axis=-1).item())
    H = mx.concatenate(hiddens, axis=1)          # (1, N, 4096) pre-norm hidden
    T = mx.array([toks])                          # (1, N)
    # teacher-forced: h_i + emb(t_{i+1}) -> predict t_{i+2}
    H = bb.norm_f(H)  # variant sweep: NVIDIA head expects POST-norm hidden (+7.5pp vs pre)
    x = head(H[:, :-2], T[:, 1:-1], bb.embeddings, kv_cache=None)
    draft = mx.argmax(model.lm_head(x), axis=-1)  # (1, N-2)
    target = T[:, 2:]
    hits = (draft == target)
    a1 = mx.mean(hits).item()
    # split at </think> if present
    text_toks = toks
    think_id_pos = None
    dec = tok.decode(text_toks)
    # find token index where </think> ends by cumulative decode
    import bisect
    if "</think>" in dec:
        # crude: walk tokens until decoded prefix contains </think>
        for i in range(len(text_toks)):
            if "</think>" in tok.decode(text_toks[:i+1]):
                think_id_pos = i
                break
    if think_id_pos and think_id_pos < N - 40:
        a_think = mx.mean(hits[:, :max(1,think_id_pos-2)]).item()
        a_answer = mx.mean(hits[:, think_id_pos:]).item()
        print(f"alpha1[{name}] = {a1:.3f} overall | thinking {a_think:.3f} (n={think_id_pos}) | ANSWER {a_answer:.3f} (n={N-2-think_id_pos})")
    else:
        print(f"alpha1[{name}] = {a1:.3f} (no </think> within {N} toks)")
    print(f"   tail sample: {tok.decode(toks[-40:])!r}")
