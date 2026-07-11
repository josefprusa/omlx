#!/usr/bin/env python3
"""Stream-free decode probe: warmed T1/T256 pairs so all fixed overheads cancel.

WHY: streaming client timings lie on this stack (thinking tokens land in
delta.reasoning_content, and one model shipped a one-SSE-chunk serving bug that
made a 128-token answer arrive as a single chunk -> bogus "48 tok/s"). A
non-streaming request returns usage.completion_tokens authoritatively; two
requests that differ ONLY in max_tokens cancel prefill + connection + sampler
warmup, so decode = (completion_256 - completion_1) / (wall_256 - wall_1).

METHOD (per prompt): send max_tokens=1 THREE times.
  1) prime  -> populates the prefix cache (RAM+SSD) for this exact prompt
  2) T1     -> cached-prefill + exactly 1 decoded token (measures fixed cost)
  3) T256   -> cached-prefill + N decoded tokens
Decode tok/s = (c256 - c1) / (t256 - t1). Overheads present in BOTH T1 and T256
subtract out, leaving pure steady-state decode.

USAGE (never point --base-url at a server someone else is live-testing):
  .venv/bin/python scripts/t1t256_probe.py \
      --base-url http://127.0.0.1:8000 \
      --model Nemotron-3-Ultra-oQNVFP4-dq8 \
      --prompt-tokens 5000            # 0 = short code prompt; else fresh-nonce filler

EXPECTED OUTPUT (shape; numbers are the production Ultra ledger, todo.md:1101):
  [short] prompt=41 prime=68.0s T1c=0.31s T256c=19.8s -> decode=13.08 tok/s
     OUT: def merge_sort(arr): ...
  [5k]    prompt=5007 prime=42.0s T1c=3.10s T256c=22.6s -> decode=13.07 tok/s
     OUT: The secondary mirror actuators show drift; ...

Every timing prompt carries a fresh nonce so the prefix cache cannot serve a
stale identical prompt (RAM+SSD prefix cache survives restarts). To measure the
cache-HIT path deliberately, pass a FIXED --nonce and run twice.

# > Verified 2026-07-05 . Mac Studio M3 Ultra 512GB (819GB/s) . MLX 0.31.2 .
# omlx 0.4.5.dev1 . branch glm5.2-native-kernels-v0.4.5 (uncommitted tree).
# Measured here, not universal - re-verify after MLX/omlx upgrades.
"""
import argparse
import json
import os
import random
import time
import urllib.request

# Fixed sanitized default; override for other job sandboxes. Key is READ at
# runtime from this path, never embedded in the script.
DEFAULT_KEY_FILE = os.environ.get("OMLX_API_KEY_FILE", "")

_PARA = ("The observatory logged another calibration pass at dawn, noting drift in the "
         "secondary mirror actuators and a slow thermal gradient across the truss. "
         "Engineers recorded seeing conditions, guider residuals, and dome shutter "
         "timings before archiving the run for the survey pipeline. ")  # ~50 tokens


def build_prompt(prompt_tokens, nonce):
    if prompt_tokens <= 0:
        return ("Write a Python function `merge_sort(arr)` that sorts a list using the "
                "merge sort algorithm, then explain how it works step by step.")
    reps = max(1, prompt_tokens // 50)
    return (f"[{nonce}] Read the following operations log excerpts.\n\n" + _PARA * reps +
            "\n\nIn two sentences, what physical subsystem shows drift and what data "
            "was archived?")


def ask(base_url, model, key, prompt, maxtok):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxtok, "temperature": 0.0}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        obj = json.load(r)
    dt = time.time() - t0
    u = obj["usage"]
    msg = obj["choices"][0]["message"]
    # count-both-channels: thinking models return content in reasoning_content
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return dt, u["prompt_tokens"], u["completion_tokens"], text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True,
                    help="e.g. http://127.0.0.1:8000 (REQUIRED; no default so you "
                         "cannot accidentally hit a live server)")
    ap.add_argument("--model", default="Nemotron-3-Ultra-oQNVFP4-dq8")
    ap.add_argument("--api-key-file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="0 = short code prompt; N = fresh-nonce filler of ~N tokens")
    ap.add_argument("--nonce", default=None,
                    help="fixed nonce reuses the prefix cache (cache-HIT test); "
                         "default is random per run (fresh-prefill test)")
    ap.add_argument("--decode-tokens", type=int, default=256)
    args = ap.parse_args()
    if not args.api_key_file:
        ap.error("set OMLX_API_KEY_FILE or pass --api-key-file")

    key = open(args.api_key_file).read().strip()
    nonce = args.nonce or f"probe-{time.time_ns()}-{random.randint(0, 1 << 30)}"
    label = "short" if args.prompt_tokens <= 0 else f"{args.prompt_tokens // 1000}k"
    prompt = build_prompt(args.prompt_tokens, nonce)

    t_prime, pt, _, _ = ask(args.base_url, args.model, key, prompt, 1)
    t1, _, c1, _ = ask(args.base_url, args.model, key, prompt, 1)
    t256, _, c256, out = ask(args.base_url, args.model, key, prompt, args.decode_tokens)
    dec = (c256 - c1) / (t256 - t1) if t256 > t1 and c256 > c1 else float("nan")
    print(f"[{label}] prompt={pt} prime={t_prime:.2f}s T1c={t1:.2f}s "
          f"T{c256}c={t256:.2f}s -> decode={dec:.2f} tok/s")
    print("   OUT:", out[:120].replace("\n", " "))


if __name__ == "__main__":
    main()
