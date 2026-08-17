"""End-to-end reproduction: learn a refusal direction, sweep alpha, measure the tax.

    python src/run_sweep.py --alphas 0 0.5 1.0 1.5 2.0

Writes results/sweep.csv, results/generations.jsonl, results/direction.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from metrics import classify, perplexity, time_per_token  # noqa: E402
from steering import (  # noqa: E402
    DirectionalAblation,
    generate,
    last_token_states,
    learn_direction,
    load_model,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--limit", type=int, default=0, help="eval on first N prompts (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--layer", type=int, default=-1, help="-1 = pick by separation score")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    torch.manual_seed(0)

    print(f"loading {args.model} on {args.device} ...", flush=True)
    model, tok = load_model(args.model, device=args.device)

    pairs = read_jsonl(DATA / "contrastive_pairs.jsonl")
    eval_prompts = read_jsonl(DATA / "over_refusal_100.jsonl")
    holdout = [p["refuse_side"] for p in pairs[-8:]]   # illicit-framed control set
    train_pairs = pairs[:-8]                           # direction is learned only from these
    if args.limit:
        eval_prompts = eval_prompts[: args.limit]
    ppl_text = (DATA / "ppl_corpus.txt").read_text(encoding="utf-8")

    print(f"learning direction from {len(train_pairs)} contrastive pairs ...", flush=True)
    refuse_states = last_token_states(model, tok, [p["refuse_side"] for p in train_pairs], args.device)
    comply_states = last_token_states(model, tok, [p["comply_side"] for p in train_pairs], args.device)
    directions, scores = learn_direction(refuse_states, comply_states)

    layer = int(scores.argmax().item()) if args.layer < 0 else args.layer
    v = directions[layer]
    print(f"layer {layer} of {directions.shape[0] - 1} selected, Cohen's d = {scores[layer]:.2f}")
    (RESULTS / "direction.json").write_text(json.dumps({
        "model": args.model,
        "selected_layer": layer,
        "n_layers_plus_embed": int(directions.shape[0]),
        "cohens_d_per_layer": [round(float(s), 4) for s in scores],
        "selected_cohens_d": round(float(scores[layer]), 4),
        "n_train_pairs": len(train_pairs),
        "n_holdout_illicit": len(holdout),
        "n_eval_prompts": len(eval_prompts),
    }, indent=2))

    rows, gen_log = [], []
    for alpha in args.alphas:
        t_start = time.time()
        print(f"\n=== alpha = {alpha} ===", flush=True)
        with DirectionalAblation(model, v, alpha):
            counts = {"ANSWERED": 0, "REFUSAL": 0, "DEGENERATE": 0}
            for i, item in enumerate(eval_prompts):
                text = generate(model, tok, item["prompt"], args.device, args.max_new_tokens)
                label = classify(text)
                counts[label] += 1
                gen_log.append({"alpha": alpha, "set": "benign_over_refusal",
                                "id": item["id"], "category": item["category"],
                                "label": label, "response": text})
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{len(eval_prompts)} {counts}", flush=True)

            hcounts = {"ANSWERED": 0, "REFUSAL": 0, "DEGENERATE": 0}
            for j, prompt in enumerate(holdout):
                text = generate(model, tok, prompt, args.device, args.max_new_tokens)
                label = classify(text)
                hcounts[label] += 1
                gen_log.append({"alpha": alpha, "set": "holdout_illicit",
                                "id": f"HOLD-{j:02d}", "category": "illicit_framed",
                                "label": label, "response": text})

            ppl = perplexity(model, tok, ppl_text, args.device)
            ms_tok = time_per_token(model, tok, eval_prompts[0]["prompt"], args.device)

        n = len(eval_prompts)
        h = len(holdout)
        row = {
            "alpha": alpha,
            "response_rate": round(100.0 * (n - counts["REFUSAL"]) / n, 1),
            "coherent_response_rate": round(100.0 * counts["ANSWERED"] / n, 1),
            "refusal_rate": round(100.0 * counts["REFUSAL"] / n, 1),
            "degenerate_rate": round(100.0 * counts["DEGENERATE"] / n, 1),
            "holdout_illicit_response_rate": round(100.0 * (h - hcounts["REFUSAL"]) / h, 1),
            "holdout_illicit_coherent_rate": round(100.0 * hcounts["ANSWERED"] / h, 1),
            "perplexity": round(ppl, 3),
            "ms_per_token": round(ms_tok, 2),
            "wall_clock_s": round(time.time() - t_start, 1),
        }
        rows.append(row)
        print(f"  -> {row}", flush=True)

    base = rows[0]
    for row in rows:
        row["ppl_delta_pct"] = round(100.0 * (row["perplexity"] - base["perplexity"]) / base["perplexity"], 2)
        row["latency_overhead_pct"] = round(
            100.0 * (row["ms_per_token"] - base["ms_per_token"]) / base["ms_per_token"], 2)

    with open(RESULTS / "sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(RESULTS / "generations.jsonl", "w", encoding="utf-8") as fh:
        for entry in gen_log:
            fh.write(json.dumps(entry) + "\n")

    print(f"\nwrote {RESULTS / 'sweep.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
