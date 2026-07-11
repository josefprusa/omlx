#!/usr/bin/env python3
"""Serial graded-accuracy bench (gsm8k / mmlu / arc) via the server's OpenAI API.

Zero-shot, thinking off (enable_thinking:false), temp 0, identical prompts per
model -> the RELATIVE delta between two models/quantizations is the metric, not
the absolute score. SERIAL by design: concurrency corrupts quality on some
serving paths (SpecPrefill under 6-way load dropped gsm8k 92% -> 15% while serial
stayed 92% -- lessons.md 2026-07-05). Always reproduce a surprising quality gap
serially before blaming the model.

FIX vs the campaign original: any suite with n=0 is SKIPPED with a message instead
of crashing on `ok / len(items)` (ZeroDivisionError). Pass --gsm8k-n 0 to run only
mmlu+arc, etc.

USAGE (do NOT point --base-url at a server under live test):
  .venv/bin/python scripts/acc_bench_serial.py \
      --base-url http://127.0.0.1:8000 \
      --model Nemotron-3-Ultra-oQNVFP4-dq8 \
      --gsm8k-n 60 --mmlu-n 150 --arc-n 150
Requires the `datasets` package (HuggingFace) + network on first run.

EXPECTED OUTPUT (shape; numbers are the Ultra production ledger, todo.md:1052):
  RESULT gsm8k Nemotron-3-Ultra-oQNVFP4-dq8 acc=0.9167 n=60
  RESULT mmlu  Nemotron-3-Ultra-oQNVFP4-dq8 acc=0.8530 n=150
  RESULT arc   Nemotron-3-Ultra-oQNVFP4-dq8 acc=0.9670 n=150
  ACC_DONE Nemotron-3-Ultra-oQNVFP4-dq8
  (with --mmlu-n 0:)
  SKIP mmlu (n=0)

# > Verified 2026-07-05 . Mac Studio M3 Ultra 512GB (819GB/s) . MLX 0.31.2 .
# omlx 0.4.5.dev1 . branch glm5.2-native-kernels-v0.4.5 (uncommitted tree).
# Measured here, not universal - re-verify after MLX/omlx upgrades.
"""
import argparse
import json
import os
import re
import urllib.request

DEFAULT_KEY_FILE = os.environ.get("OMLX_API_KEY_FILE", "")


def ask(base_url, model, key, prompt, maxtok):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxtok, "temperature": 0.0, "enable_thinking": False}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.load(r)
    msg = obj["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def last_number(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return m[-1].rstrip(".") if m else None


def letter(s, valid="ABCD"):
    for pat in (rf"answer is \(?([{valid}])\)?", rf"^\(?([{valid}])\)?[.\s)]",
                rf"\b([{valid}])\b"):
        m = re.search(pat, s.strip(), re.M | re.I)
        if m:
            return m.group(1).upper()
    return None


def run(name, model, items, fn):
    """Serial evaluation. Returns acc, or None if the suite is empty (n=0)."""
    n = len(items)
    if n < 1:                              # <-- the div-zero guard (was a crash)
        print(f"SKIP {name} (n=0)", flush=True)
        return None
    ok = 0
    for done, item in enumerate(items, 1):
        ok += fn(item)
        if done % 50 == 0:
            print(f"  {name} {done}/{n} acc={ok / done:.3f}", flush=True)
    acc = ok / n
    print(f"RESULT {name} {model} acc={acc:.4f} n={n}", flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True,
                    help="e.g. http://127.0.0.1:8000 (REQUIRED; no default)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--gsm8k-n", type=int, default=150)
    ap.add_argument("--mmlu-n", type=int, default=300)
    ap.add_argument("--arc-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not args.api_key_file:
        ap.error("set OMLX_API_KEY_FILE or pass --api-key-file")

    key = open(args.api_key_file).read().strip()
    import datasets  # deferred: only needed when actually running

    def sel(ds, n):
        return list(ds.shuffle(seed=args.seed).select(range(n))) if n > 0 else []

    # gsm8k -----------------------------------------------------------------
    g = sel(datasets.load_dataset("openai/gsm8k", "main", split="test"), args.gsm8k_n)

    def gsm_item(row):
        gold = last_number(row["answer"].split("####")[-1])
        out = ask(args.base_url, args.model, key, row["question"] +
                  "\n\nSolve step by step briefly, then give the final numeric "
                  "answer on the last line as: ANSWER: <number>", 512)
        return last_number(out) == gold
    run("gsm8k", args.model, g, gsm_item)

    # mmlu ------------------------------------------------------------------
    m = sel(datasets.load_dataset("cais/mmlu", "all", split="test"), args.mmlu_n)

    def mmlu_item(row):
        p = (row["question"] + "\n" +
             "\n".join(f"{l}. {c}" for l, c in zip("ABCD", row["choices"])) +
             "\n\nAnswer with just the letter.")
        return letter(ask(args.base_url, args.model, key, p, 8)) == "ABCD"[row["answer"]]
    run("mmlu", args.model, m, mmlu_item)

    # arc_challenge ---------------------------------------------------------
    a = sel(datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"),
            args.arc_n)

    def arc_item(row):
        labs = row["choices"]["label"]
        p = (row["question"] + "\n" +
             "\n".join(f"{l}. {t}" for l, t in zip(labs, row["choices"]["text"])) +
             "\n\nAnswer with just the letter.")
        return letter(ask(args.base_url, args.model, key, p, 8),
                      valid="".join(labs)) == row["answerKey"]
    run("arc", args.model, a, arc_item)

    print("ACC_DONE", args.model, flush=True)


if __name__ == "__main__":
    main()
