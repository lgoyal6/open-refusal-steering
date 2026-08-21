"""Plot the steering tradeoff curve from a sweep CSV in results/.

    python src/plot.py                              # plots results/sweep.csv, i.e. your run
    python src/plot.py --csv sweep_partial_n100.csv  # reproduces the committed figure
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

INK = "#1b3a5c"
TAX = "#a03323"
GREY = "#8a8a8a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sweep.csv",
                    help="filename inside results/ to plot (default: your own sweep run)")
    args = ap.parse_args()

    src = RESULTS / args.csv
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Run src/run_sweep.py first, or pass "
            f"--csv sweep_partial_n100.csv to plot the committed data."
        )
    with open(src, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    alpha = [float(r["alpha"]) for r in rows]
    resp = [float(r["response_rate"]) for r in rows]
    coherent = [float(r["coherent_response_rate"]) for r in rows]
    ppl = [float(r["perplexity"]) for r in rows]
    ppl_x = [float(r["ppl_x_baseline"]) for r in rows]
    hold = [float(r["holdout_illicit_response_rate"]) for r in rows]
    ms = [float(r["ms_per_token"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- panel 1: the headline axis and the tax, same x ---
    ax = axes[0]
    ax.plot(alpha, resp, "o-", color=INK, lw=2, label="response rate (100 benign)")
    ax.plot(alpha, hold, "s--", color=GREY, lw=1.5, label="response rate (8 held-out illicit)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("response rate (%)")
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(alpha, ppl, "^-", color=TAX, lw=2, label="perplexity (log scale)")
    ax2.set_yscale("log")
    ax2.set_ylabel("perplexity, held-out corpus (log)", color=TAX)
    ax2.tick_params(axis="y", colors=TAX)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="center right")
    ax.set_title("Response rate rises. So does perplexity.", fontsize=11)

    # --- panel 2: the tradeoff frontier ---
    ax = axes[1]
    ax.plot(ppl_x, coherent, "o-", color=INK, lw=2)
    ax.set_xscale("log")
    # label only the points that carry the argument; the low-alpha cluster overlaps
    for a, x, y in zip(alpha, ppl_x, coherent):
        if a in (0.0, 2.5, 3.0):
            ax.annotate(f"a={a:g}", (x, y), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color="#444")
    # the rightmost point scores best on every behavioural metric and is a broken model
    ax.annotate("a=3.0 scores 100% on every\nbehavioural metric.\nThe model emits token soup.",
                xy=(ppl_x[-1], coherent[-1]), xytext=(-30, -78),
                textcoords="offset points", fontsize=8, color=TAX, ha="right",
                arrowprops=dict(arrowstyle="->", color=TAX, lw=1.2))
    ax.axvline(1.005, color=TAX, ls=":", lw=1.6)
    ax.text(1.005, 55, "  CTGT bound: '< 0.5% variation'",
            color=TAX, fontsize=8, va="bottom", rotation=90)
    ax.set_xlabel("perplexity, multiple of baseline (log)  ->  capability tax")
    ax.set_ylabel("coherent response rate (%)")
    ax.grid(alpha=0.25)
    ax.set_title("Higher is not better: the best-scoring point is the broken one", fontsize=11)

    # --- panel 3: latency ---
    ax = axes[2]
    ax.plot(alpha, ms, "o-", color=INK, lw=2)
    ax.set_xlabel("alpha")
    ax.set_ylabel("ms per generated token")
    ax.set_ylim(0, max(ms) * 1.35)
    ax.grid(alpha=0.25)
    ax.set_title("Latency: below the noise floor", fontsize=11)

    fig.suptitle(
        "Directional ablation  h' = h - alpha (h.v) v   |   Qwen2.5-0.5B-Instruct, CPU, greedy",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RESULTS / "tradeoff.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
