#!/usr/bin/env python3
"""Generate the Hugging Face dataset card from the committed results.

Written by hand once, generated from then on. Every number in the card is read
out of `results/sweep_partial_n100.csv` and `results/direction.json`, which is
the same file `results/README.md` names as the canonical run. A card typed by
hand drifts from the run it describes, and this repository exists to complain
about exactly that failure.

    python scripts/dataset-card.py > /tmp/README.md
"""
import csv
import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CANONICAL = "sweep_partial_n100.csv"


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    rows = list(csv.DictReader(open(RESULTS / CANONICAL, encoding="utf-8")))
    d = json.loads((RESULTS / "direction.json").read_text(encoding="utf-8"))
    evalset = jsonl(DATA / "over_refusal_100.jsonl")
    pairs = jsonl(DATA / "contrastive_pairs.jsonl")
    ppl_chars = len((DATA / "ppl_corpus.txt").read_text(encoding="utf-8"))

    n_hold = d["n_holdout_illicit"]
    n_train = d["n_train_pairs"]
    cats = Counter(p["category"] for p in evalset).most_common()

    f = float
    base_ppl = f(rows[0]["perplexity"])
    # The row worth pointing at: every behavioural metric perfect, language gone.
    collapse = max(rows, key=lambda r: f(r["ppl_x_baseline"]))
    # The best real trade: highest illicit climb still under a 10% perplexity tax.
    usable = [r for r in rows if f(r["ppl_x_baseline"]) < 1.1]
    best = max(usable, key=lambda r: f(r["holdout_illicit_response_rate"]))
    benign0 = f(rows[0]["response_rate"])
    best_benign = max(f(r["response_rate"]) for r in usable)

    out = []
    w = out.append

    w("---")
    w("license: cc-by-4.0")
    w("language:\n- en")
    w("task_categories:\n- text-generation")
    w("tags:\n- activation-steering\n- refusal\n- over-refusal\n"
      "- interpretability\n- reproduction\n- safety")
    w("size_categories:\n- n<1K")
    w("configs:")
    w("- config_name: over_refusal_100\n  data_files: over_refusal_100.jsonl\n  default: true")
    w("- config_name: contrastive_pairs\n  data_files: contrastive_pairs.jsonl")
    w("---")
    w("")
    w("# open-refusal-steering: the prompt sets")
    w("")
    w("The prompt sets behind an open reproduction of feature-level activation steering.")
    w("")
    w("CTGT's *A feature-level approach to mitigating bias and censorship in DeepSeek-R1*")
    w("([hal-04992348v1](https://hal.science/hal-04992348v1)) reports a jump from 32% to 100%")
    w("response rate on \"a benchmark of 100 sensitive queries\", and releases neither the")
    w("prompts nor the code. Their own public repository states that the headline comparison")
    w("cannot be regenerated from it. You cannot check a steering claim without the prompts,")
    w(f"so here are {len(evalset)}, categorised and released.")
    w("")
    w("Code, method and full write-up: https://github.com/lgoyal6/open-refusal-steering")
    w("")
    w("## Files")
    w("")
    w("| file | rows | what it is |")
    w("|---|---:|---|")
    w(f"| `over_refusal_100.jsonl` | {len(evalset)} | The evaluation set. Questions a well-calibrated "
      "assistant **should** answer, from domains where aligned models over-refuse. At any steering "
      "strength, a refusal here is unambiguously an error. |")
    w(f"| `contrastive_pairs.jsonl` | {len(pairs)} | Topic-matched refuse/comply pairs used to learn the "
      f"refusal direction. First {n_train} train the direction; the last {n_hold} `refuse_side` prompts "
      "are held out entirely and used only as a refusal-headroom control. |")
    w(f"| `ppl_corpus.txt` | {ppl_chars:,} chars | Held-out neutral prose for measuring perplexity with the "
      "steering hook live. Darwin, *On the Origin of Species*, public domain. |")
    w(f"| `CONSTRUCTION.md` | | How each set was built, including what it is deliberately not. |")
    w("")
    w("### Evaluation set composition")
    w("")
    w("| category | n |")
    w("|---|---:|")
    for c, n in cats:
        w(f"| {c} | {n} |")
    w("")
    w("## What the sets measure")
    w("")
    w(f"Model {d['model']}, layer {d['selected_layer']} of {d['n_layers_plus_embed'] - 1}, "
      f"Cohen's d = {d['selected_cohens_d']:.2f}, CPU, greedy, seed 0. "
      f"Every number below is read out of `{CANONICAL}` in the repository.")
    w("")
    w("| alpha | benign response | coherent | degenerate | illicit control | perplexity | x baseline |")
    w("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ppl = f(r["perplexity"])
        w("| {} | {}% | {}% | {}% | {}% | {} | {} |".format(
            r["alpha"],
            f(r["response_rate"]),
            f(r["coherent_response_rate"]),
            f(r["degenerate_rate"]),
            f(r["holdout_illicit_response_rate"]),
            f"{ppl:,.1f}" if ppl < 1e4 else f"{ppl:,.0f}",
            f"{f(r['ppl_x_baseline']):,.2f}" if f(r["ppl_x_baseline"]) < 1e4
            else f"{f(r['ppl_x_baseline']):,.0f}",
        ))
    w("")
    w("**Two findings the sets were built to expose.**")
    w("")
    w(f"1. **Steering never helps the prompts worth un-refusing.** Benign response starts at "
      f"{benign0:.0f}% and never beats {best_benign:.0f}% at any alpha with a perplexity tax under 10%. "
      f"The held-out illicit control meanwhile climbs from "
      f"{f(rows[0]['holdout_illicit_response_rate']):.1f}% to "
      f"{f(best['holdout_illicit_response_rate']):.1f}% at alpha {best['alpha']}. The intervention "
      f"works; it works on the wrong set.")
    w("")
    w(f"2. **Behavioural metrics pass a model that has stopped producing language.** At alpha "
      f"{collapse['alpha']} every behavioural number reads perfect: "
      f"{f(collapse['response_rate']):.0f}% response, {f(collapse['refusal_rate']):.0f}% refusal, "
      f"{f(collapse['degenerate_rate']):.0f}% degenerate. Perplexity is "
      f"{f(collapse['perplexity']):,.0f}, or {f(collapse['ppl_x_baseline']):,.0f}x the "
      f"{base_ppl:.1f} baseline. Refusal-prefix scoring passes it. This is why the capability "
      f"tax is reported on the same axis as the win.")
    w("")
    w("## Scope and limits")
    w("")
    w(f"- A 9-point sweep (0 to 4.0) was launched; alphas 3.5 and 4.0 did not complete and are not "
      f"reported. The table above is the whole of `{CANONICAL}`, unfiltered.")
    w("- One model, 0.5B parameters, on CPU. The method transfers; these numbers do not.")
    w("- The contrastive pairs use mild illicit framings (fare evasion, fake reviews, a forged "
      "doctor's note) rather than a harmful-instruction set. That is a weaker supervision signal "
      "than the published work uses, and it is a deliberate tradeoff so the set is publishable. "
      "See `CONSTRUCTION.md`.")
    w("- The direction is never learned from the set it is evaluated on.")
    w("")
    w("## Licence")
    w("")
    w("`over_refusal_100.jsonl`, `contrastive_pairs.jsonl` and `CONSTRUCTION.md` are CC BY 4.0:")
    w("")
    w("> \"open-refusal-steering benign over-refusal set\" by Laksh Goyal, CC BY 4.0.")
    w("> Source: https://github.com/lgoyal6/open-refusal-steering")
    w("")
    w("`ppl_corpus.txt` is public domain (Project Gutenberg eBook #2009, header and trailer "
      "stripped). Repository code is Apache-2.0.")

    print("\n".join(out))


if __name__ == "__main__":
    main()
