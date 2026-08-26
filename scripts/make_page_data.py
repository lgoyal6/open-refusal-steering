"""Build the JSON the results page reads.

The sweep table and a sample of the generations behind it, so a reader can move
alpha and watch every behavioural metric read perfect while perplexity says the
model has stopped working.

Only the benign over-refusal set is carried into the page. The held-out illicit
control appears as a rate and nothing else: its response rate is the number that
matters, and its prompts and completions are not what a results page is for.

    python3 scripts/make_page_data.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data"

NUMERIC = (
    "alpha", "response_rate", "coherent_response_rate", "refusal_rate",
    "degenerate_rate", "holdout_illicit_response_rate", "perplexity",
    "ppl_x_baseline", "ms_per_token",
)


def sweep() -> list[dict]:
    with (ROOT / "results/sweep_partial_n100.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    return [{k: float(r[k]) for k in NUMERIC if k in r} for r in rows]


def samples() -> list[dict]:
    """One prompt followed across the two collapse points, benign set only."""
    path = ROOT / "results/generations_alpha_2.5_3.0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # One prompt from each of three categories, whichever id comes first, so
    # the sample is not hand-picked for the most damning output.
    benign = [r for r in rows if r["set"] == "benign_over_refusal"]
    wanted, seen = [], set()
    for r in sorted(benign, key=lambda r: (r["category"], r["id"])):
        if r["category"] in seen:
            continue
        if len(seen) >= 3:
            break
        seen.add(r["category"])
        wanted.append(r["id"])
    out = []
    for r in benign:
        if r["id"] not in wanted:
            continue
        out.append({
            "alpha": r["alpha"],
            "id": r["id"],
            "category": r["category"],
            # What the repository's own classifier called it. At alpha 3.0 this
            # says ANSWERED for output that is not language.
            "label": r["label"],
            "response": r["response"],
        })
    return sorted(out, key=lambda r: (r["id"], r["alpha"]))


def main() -> None:
    payload = {
        "sweep": sweep(),
        "samples": samples(),
        "note": {
            "model": "Qwen2.5-0.5B-Instruct",
            "n_benign": 100,
            "n_holdout_illicit": 8,
            "decoding": "CPU, greedy, seed 0",
            # Reported rather than quietly dropped: two of the nine points
            # launched never finished, and nothing here extrapolates past 3.0.
            "incomplete": [3.5, 4.0],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sweep.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} kB")
    print("alphas:", [r["alpha"] for r in payload["sweep"]])
    print("samples:", len(payload["samples"]), "(benign only)")


if __name__ == "__main__":
    main()
