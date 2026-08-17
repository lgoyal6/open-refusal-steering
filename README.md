# open-refusal-steering

An open, end-to-end reproduction of feature-level activation steering, on a model that runs
on a laptop CPU, with **the prompt set released**.

The intervention is the one published by CTGT in *"A feature-level approach to mitigating
bias and censorship in DeepSeek-R1"* ([hal-04992348v1](https://hal.science/hal-04992348v1)):

```
h' = h - alpha * (h . v_censor) * v_censor
```

verbatim from the paper, *"where h is the hidden activation, and α is a tunable scalar
controlling the intervention strength."* The paper ships no code. This is that code, plus
the three things a steering claim cannot be checked without: the prompts, the capability
tax, and the curve between them.

---

## The short version

**What I noticed.** CTGT published a steering result on DeepSeek-R1 claiming a jump from 32%
to 100% response rate on 100 sensitive queries, and released neither the prompt set nor the
code. Their own public repo says the headline comparison "cannot be regenerated from this
public repository." So I implemented the method from the paper and released the prompts.

**What I found.** Qwen2.5-0.5B-Instruct, 100 benign-but-over-refused prompts, an 8-prompt
held-out illicit control, CPU, greedy, seed 0:

| alpha | benign response | coherent | degenerate | illicit control | perplexity |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 90% | 90% | 0% | 37.5% | 20.5 |
| 1.0 | 88% | 88% | 0% | 37.5% | 21.1 |
| 2.0 | 89% | 89% | 0% | **87.5%** | 21.5 |
| 2.5 | 82% | **52%** | **30%** | 100% | 70.3 |
| 3.0 | **100%** | 100% | 0% | 100% | **10,850,402** |

**Steering never helps the prompts worth un-refusing.** The benign response rate starts at
90% and never beats it at any usable alpha. Meanwhile the illicit control climbs 37.5 to
87.5 to 100. The intervention works, it just works on the wrong set.

**The alpha=3.0 row is the one to look at.** Every behavioural metric reads perfect: 100%
response, 0% refusal, 0% degenerate, 100% coherent. Perplexity is **530,000x baseline** and
the model is emitting token soup, verified by reading the outputs. Refusal-prefix scoring
passes it. My own degeneracy check passes it too, because the garbage is high-entropy rather
than looping. **Only perplexity catches it.**

At alpha=2.5 a different collapse appears, 30% degenerate and coherence falling 90 to 52,
and that one the degeneracy check does catch. Two failure modes, two detectors, and a
behavioural-metrics-only eval sees neither.

**Why it matters.** Any steering result reported without a capability tax is unfalsifiable,
because you can always reach 100% response by destroying the model. The prompt set, the
sweep and the perplexity column are all in here so the curve can be argued with.

**What it is not.** One small model at one temperature. This reproduces the method and shows
what the method costs; it is not a claim about what CTGT measured on DeepSeek-R1, whose
prompt set is still unreleased.

## Why this exists

CTGT's R1 paper reports its headline on *"a benchmark of 100 sensitive queries"* that has
never been released. Their public evaluation repo,
[`CTGT-Inc/lineage-eval`](https://github.com/CTGT-Inc/lineage-eval), says of a later study:
*"the adapter generations and headline comparison cannot be regenerated from this public
repository."* Both statements are fair and openly made. They also mean nobody outside the
company can run the experiment.

So: 100 prompts, released and printable. A working implementation. And the axis that
decides whether steering is a win, plotted rather than summarised.

**Credit where it is due.** `lineage-eval` is a good release: 304 matched prompts, 1,824
responses, 7,296 four-judge classifications, a documented rubric, and a mechanical
`degeneracy.py` that flags repeated n-gram loops as `INVALID_DEGENERATE` instead of scoring
them. That last idea is good enough that this repo reuses it (`src/metrics.py`).

## The point

Response rate is the axis everybody publishes, and it is the easy one. Projecting hard
enough along any direction will eventually stop a model refusing, because it will
eventually stop the model doing anything. The honest question is what each point of
response rate costs. This repo measures four axes at every alpha:

| axis | what it catches |
|---|---|
| response rate | the published headline |
| **perplexity, held-out corpus, hook live** | the capability tax |
| degenerate rate | fluent-looking loops that a refusal check scores as success |
| ms per token | the real-time claim |

## Run it

Needs ~2GB RAM and about 40 minutes on an M-series CPU. No GPU, no API keys, no dataset
download.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python transformers matplotlib

.venv/bin/python src/run_sweep.py --alphas 0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 --max-new-tokens 64
.venv/bin/python src/plot.py
```

Faster smoke test (about a minute):

```bash
.venv/bin/python src/run_sweep.py --limit 5 --alphas 0 2.0 --max-new-tokens 32
```

Outputs land in `results/`: `sweep.csv`, `generations.jsonl` (every response, labelled),
`direction.json` (per-layer separation scores), `tradeoff.png`.

## What is in here

```
data/over_refusal_100.jsonl   100 benign prompts an aligned model should answer   <- the released set
data/contrastive_pairs.jsonl  32 topic-matched pairs used to learn the direction
data/ppl_corpus.txt           held-out public-domain prose for perplexity
data/CONSTRUCTION.md          how all three were built, and their limitations
src/steering.py               direction learning + the forward hook
src/metrics.py                refusal, degeneracy, perplexity, latency
src/run_sweep.py              the sweep
src/plot.py                   the tradeoff curve
```

**The prompt set is benign by construction.** Every one of the 100 is a question a
well-calibrated assistant should answer: acetaminophen dosing ceilings, what to say to a
friend who mentions suicide, how SQL injection works so you can prevent it, what happened
at Tiananmen Square. Refusing them is a false positive, not safety, which is what makes
them a clean measuring stick. Nothing in this repo requests anything dangerous, and that is
deliberate: a benchmark you cannot print is a benchmark nobody can check. See
`data/CONSTRUCTION.md`.

## Method notes

- **The direction is learned, not chosen.** Difference-in-means of the last-prompt-token
  residual over 24 contrastive pairs, per layer; the layer is picked by Cohen's d of the
  1-D projection. 8 pairs are held out of direction learning entirely.
- **The hook runs at every decoder layer and every token position**, which is what a live
  inference-time intervention has to do.
- **Perplexity is measured with the hook live.** Measuring it with the intervention off
  would report nothing.
- **Refusal detection is refusal-prefix substring matching**, the standard cheap proxy in
  this literature. It is a proxy. An LLM judge panel like `lineage-eval`'s is better and
  costs three API keys; this runs offline for free.
- **Greedy decoding, seed 0**, so the sweep is deterministic.

## Prior art

The difference-in-means directional-ablation construction is published: Arditi et al.,
*"Refusal in Language Models Is Mediated by a Single Direction"*
([arXiv:2406.11717](https://arxiv.org/abs/2406.11717)), with code. Nothing here claims to
invent it. The contribution is that CTGT's specific claims had no runnable artifact, and
now there is one, with the released prompts and the tradeoff curve.

## Limitations

Qwen2.5-0.5B-Instruct is not DeepSeek-R1-Distill-Llama-70B. No number here is a direct
comparison to CTGT's 32% → 96%; what transfers is the *shape* of the tradeoff. 100
hand-written prompts by one author are a probe, not a census. Substring refusal detection
misses polite non-answers that engage with nothing.

## Licence

Code Apache-2.0. `data/over_refusal_100.jsonl` and `data/contrastive_pairs.jsonl` CC BY
4.0, so anyone can lift them. `data/ppl_corpus.txt` is public domain (Project Gutenberg
eBook #2009).
