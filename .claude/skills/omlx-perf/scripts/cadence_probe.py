#!/usr/bin/env python3
"""SSE streaming cadence probe that counts BOTH delta channels.

WHY THIS EXISTS: a client that measures TTFT/tok-s from delta.content ALONE lies
on this stack. Reasoning models emit their thinking into delta.reasoning_content
first and only later switch to delta.content -> a content-only counter sees
seconds of phantom "dead air" before its first token, and mis-times decode. This
was chased as an omlx serving bug and refuted: the tokens were streaming the
whole time, just in the reasoning channel (channel/server details: see
../omlx.md). This probe reports each channel separately so you can SEE the split.

It also exposes the genuine one-SSE-chunk serving artifact (todo.md:1055): if a
128-token answer arrives as a single content chunk at completion, you will see
content chunks=1 with a late first@ timestamp -> distrust any streamed tok/s and
switch to scripts/t1t256_probe.py (stream-free) for the real decode number.

USAGE (do NOT point --base-url at a server under live test):
  .venv/bin/python scripts/cadence_probe.py \
      --base-url http://127.0.0.1:8000 \
      --model Nemotron-3-Ultra-oQNVFP4-dq8 \
      --max-tokens 128

EXPECTED OUTPUT (shape; illustrates a thinking model + the dead-air trap):
  [cadence] reasoning: chunks=63 first@0.48s last@8.90s chars=511 median-gap=0.132s
  [cadence] content:   chunks=1  first@8.95s last@8.95s chars=402 median-gap=n/a
  NOTE: content first-token 8.95s >> reasoning first-token 0.48s
        -> a content-only client would report ~8.5s of false dead air.

# > Verified 2026-07-05 . Mac Studio M3 Ultra 512GB (819GB/s) . MLX 0.31.2 .
# omlx 0.4.5.dev1 . branch glm5.2-native-kernels-v0.4.5 (uncommitted tree).
# Measured here, not universal - re-verify after MLX/omlx upgrades.
"""
import argparse
import json
import os
import time
import urllib.request

DEFAULT_KEY_FILE = os.environ.get("OMLX_API_KEY_FILE", "")


def report(name, chunks):
    if not chunks:
        print(f"[cadence] {name:9s} chunks=0 (no tokens on this channel)")
        return None
    first, last = chunks[0][0], chunks[-1][0]
    chars = sum(n for _, n in chunks)
    if len(chunks) > 1:
        gaps = [chunks[i][0] - chunks[i - 1][0] for i in range(1, len(chunks))]
        med = f"{sorted(gaps)[len(gaps) // 2]:.3f}s"
    else:
        med = "n/a"
    print(f"[cadence] {name:9s} chunks={len(chunks)} first@{first:.2f}s "
          f"last@{last:.2f}s chars={chars} median-gap={med}")
    return first


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True,
                    help="e.g. http://127.0.0.1:8000 (REQUIRED; no default)")
    ap.add_argument("--model", default="Nemotron-3-Ultra-oQNVFP4-dq8")
    ap.add_argument("--api-key-file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--prompt",
                    default="What is 23*17? Think carefully, then give the answer.")
    ap.add_argument("--nonce", default=None, help="default random (fresh prefill)")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    if not args.api_key_file:
        ap.error("set OMLX_API_KEY_FILE or pass --api-key-file")

    key = open(args.api_key_file).read().strip()
    nonce = args.nonce or f"cadence-{time.time_ns()}"
    prompt = f"[{nonce}] {args.prompt}"
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens, "temperature": 0.0, "stream": True}
    req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    t0 = time.time()
    content, reasoning = [], []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            now = time.time() - t0
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                c = d.get("content")
                rc = d.get("reasoning_content") or d.get("reasoning")
                if c:
                    content.append((now, len(c)))
                if rc:
                    reasoning.append((now, len(rc)))

    r_first = report("reasoning", reasoning)
    c_first = report("content", content)
    if r_first is not None and c_first is not None and c_first - r_first > 0.5:
        print(f"NOTE: content first-token {c_first:.2f}s >> reasoning first-token "
              f"{r_first:.2f}s -> a content-only client would report "
              f"~{c_first - r_first:.1f}s of false dead air.")
    if len(content) == 1 and args.max_tokens > 8:
        print("NOTE: content arrived as ONE chunk -> streamed tok/s is meaningless "
              "here; use scripts/t1t256_probe.py for the real decode rate.")


if __name__ == "__main__":
    main()
