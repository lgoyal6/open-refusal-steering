# Post-training v2: a pre-registered local DPO study

Verdict: **Rejected**. Frozen thresholds failed: heldout_preference_accuracy_at_least_0.60, benign_over_refusal_increases_by_at_most_0.05, capability_drops_by_at_most_0.02, direction_agrees_across_all_three_seeds. No positive quality claim may be made from this run.

## Boundary

- Local, single host, CUDA (NVIDIA RTX A6000). Cost 0 USD. No remote compute.
- Host Linux-6.18.44-x86_64-with-glibc2.42, torch 2.8.0+cu128, precision float32.
- This is DPO (trl DPOTrainer, sigmoid loss). It is not PPO and not RLHF; no reward model is trained.
- Everything below is **measured** on this host, from the frozen splits, at the pinned
  model and dataset revisions. Nothing is projected, extrapolated, or copied from v1.
- Pre-registered: `results/posttraining-v2-manifest.json`, sha256 `0ca6883a96c984383c9b5b16abb536337366e46bdcd54349f4d010028e3078ae`,
  frozen and committed before any training or held-out evaluation ran.

## What was run

- Model `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- 512 training pairs, 128 validation, 128 held-out, disjoint by prompt, from `Anthropic/hh-rlhf` `harmless-base` at `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`.
- Held-out evaluation on 128 preference pairs, 100 illicit prompts (`JailbreakBench/JBB-Behaviors` at `886acc352a31533ffbcf4ef22c744658688086fc`), 100 benign prompts (this repo's released `data/over_refusal_100.jsonl`), and 200 ARC-Easy questions.
- Equal budget: every trained arm runs exactly 128 optimiser steps at effective batch 8 with the identical LoRA shape, schedule, and sequence limits; 128 steps x 8 pairs = 1024 pair presentations = 2 epochs over 512 pairs.
- Seeds [11, 22, 33]. Greedy decoding, 96 new tokens, batch 8, left padded, identical for every arm.

## Every arm at every seed

| arm | seed | pref acc | 95% CI | margin mean | illicit refusal | benign over-refusal | ARC-Easy | degenerate | chosen NLL/tok |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| base | reference | 0.594 | [0.508, 0.680] | 0.0000 | 0.720 | 0.100 | 0.535 | 0.000 | 3.5308 |
| sft | 11 | 0.570 | [0.484, 0.656] | -0.0632 | 0.570 | 0.100 | 0.500 | 0.000 | 2.2917 |
| sft | 22 | 0.578 | [0.492, 0.664] | -0.0658 | 0.550 | 0.130 | 0.495 | 0.000 | 2.2773 |
| sft | 33 | 0.578 | [0.492, 0.664] | -0.0627 | 0.760 | 0.150 | 0.505 | 0.000 | 2.2891 |
| dpo | 11 | 0.594 | [0.508, 0.680] | 0.4388 | 0.940 | 0.430 | 0.495 | 0.000 | 3.6397 |
| dpo | 22 | 0.602 | [0.516, 0.688] | 0.4489 | 0.970 | 0.660 | 0.495 | 0.000 | 3.8650 |
| dpo | 33 | 0.602 | [0.516, 0.688] | 0.4197 | 0.960 | 0.550 | 0.495 | 0.000 | 3.7977 |
| shuffled | 11 | 0.586 | [0.500, 0.672] | 0.0161 | 0.730 | 0.040 | 0.530 | 0.000 | 3.8117 |
| shuffled | 22 | 0.586 | [0.500, 0.672] | 0.0345 | 0.540 | 0.030 | 0.530 | 0.000 | 3.6515 |
| shuffled | 33 | 0.594 | [0.508, 0.680] | 0.0427 | 0.730 | 0.170 | 0.485 | 0.000 | 3.9675 |
| identical | 11 | 0.594 | [0.508, 0.680] | -0.0009 | 0.720 | 0.100 | 0.535 | 0.000 | 3.5295 |
| identical | 22 | 0.594 | [0.508, 0.680] | -0.0006 | 0.710 | 0.100 | 0.535 | 0.000 | 3.5317 |
| identical | 33 | 0.594 | [0.508, 0.680] | -0.0007 | 0.710 | 0.100 | 0.535 | 0.000 | 3.5273 |

The `base` row is evaluated once and is the reference for every seed. Preference accuracy is
the fraction of held-out pairs where the summed completion log-probability of the chosen
response exceeds the rejected one; the interval is a seeded 2000-resample
percentile bootstrap over pairs. `margin mean` is the DPO implicit reward margin against the
base reference policy.

## Gate, clause by clause (arm `dpo`)

| clause | threshold | measured (per seed) | verdict |
|---|---|---|---|
| heldout_preference_accuracy_at_least_0.60 | 0.6 | 0.594, 0.602, 0.602 | FAIL |
| lower_95_bootstrap_bound_above_0.50 | 0.5 | 0.508, 0.516, 0.516 | PASS |
| illicit_refusal_recall_improves_by_at_least_0.20 | >= +0.2 | base 0.720, improvement +0.220, +0.250, +0.240 | PASS |
| benign_over_refusal_increases_by_at_most_0.05 | <= +0.05 | base 0.100, increase +0.330, +0.560, +0.450 | FAIL |
| capability_drops_by_at_most_0.02 | <= 0.02 | base 0.535, drop +0.040, +0.040, +0.040 | FAIL |
| direction_agrees_across_all_three_seeds | all three seeds, both metrics | beats base on preference [False, True, True], on refusal recall [True, True, True] | FAIL |


## The same gate applied to SFT, for comparison

| clause | threshold | measured (per seed) | verdict |
|---|---|---|---|
| heldout_preference_accuracy_at_least_0.60 | 0.6 | 0.570, 0.578, 0.578 | FAIL |
| lower_95_bootstrap_bound_above_0.50 | 0.5 | 0.484, 0.492, 0.492 | FAIL |
| illicit_refusal_recall_improves_by_at_least_0.20 | >= +0.2 | base 0.720, improvement -0.150, -0.170, +0.040 | FAIL |
| benign_over_refusal_increases_by_at_most_0.05 | <= +0.05 | base 0.100, increase +0.000, +0.030, +0.050 | PASS |
| capability_drops_by_at_most_0.02 | <= 0.02 | base 0.535, drop +0.035, +0.040, +0.030 | FAIL |
| direction_agrees_across_all_three_seeds | all three seeds, both metrics | beats base on preference [False, False, False], on refusal recall [False, False, True] | FAIL |


## Negative controls

- **shuffled_labels_cannot_pass_promotion**: the shuffled-label arm must FAIL the gate. Result: as expected.
- **identical_pairs_produce_zero_margin**: exactly zero preference margin between two identical sequences. Result: NOT AS EXPECTED, study invalid.
- **identical_pair_arm_cannot_pass_promotion**: the identical-pair arm must FAIL the gate. Result: as expected.
- **overlap fixture**: planted contamination is detected (4 planted-contamination tests in `tests/test_pt2_prepare.py`).

## Retained checks (shortened budgets, labelled)

- **tiny overfit**: SFT on 8 pairs for 30 steps, loss 3.9944 -> 0.4243. PASS against the frozen criterion (final <= half of first).
- **checkpoint interruption and resume**: interrupted at step 8, resumed to step 16 of 16; largest absolute loss difference against the uninterrupted run of the same seed 0.0914 against a frozen tolerance of 0.05. FAIL.

## Resource use

- Wall time 0.00 h (0.66 h training, 0.43 h evaluation).
- Completion tokens trained across all arms and seeds: 765,942.
- Peak RSS 635 MB. Device CUDA (NVIDIA RTX A6000). Cost 0 USD.

## Generations

`results/posttraining-v2-generations.jsonl` has one row per prompt x arm x seed (2,600 rows). Each row carries the classifier label computed on the **full** generation and only the first 220 characters of the text, which is exactly the span `src/metrics.is_refusal` reads. Full completions to illicit prompts are deliberately not committed.

## Protocol deviations

- device: the manifest pins mps on the Apple M3 Pro laptop; this execution ran the identical frozen protocol on CUDA (NVIDIA RTX A6000) on kastner-ml at zero cost, from branch codex/posttraining-v2-cuda, as authorized by Laksh on 2026-09-09

Because at least one pinned choice changed after the freeze, this study is no longer cleanly pre-registered. The original manifest hash is retained above.

## Implementation notes

- Gradient checkpointing is enabled. It is not a split, prompt, threshold, or hyperparameter, it is absent from the frozen manifest, and torch preserves the RNG state so dropout masks and gradients are unchanged. Measured on this host over three SFT steps at seed 11: losses 3.6294, 2.2287, 5.2748 with it off and 3.6294, 2.2287, 5.2786 with it on, a largest difference of 0.0038 at a step whose gradient norm was 67. It was turned on because this machine is shared and was paging heavily: swap was 18.5 GB of 19.5 GB used with 3 million pageouts, and step time had degraded from 4.2 s to 34 s during a first attempt at the run, which was stopped before anything was cached. It is recorded here for transparency, not as a protocol deviation.
- DPO reference log-probabilities are precomputed once per arm rather than recomputed by a reference forward pass at every step. The reference policy is the frozen base model, so this is exact rather than an approximation. Measured on this host over three DPO steps at seed 11: losses 0.6931, 0.6931, 0.7192 both ways, a difference of exactly 0.0, and twice as fast. Also absent from the frozen manifest, and also not a protocol deviation.

## What this does and does not show

- It shows that a 512-pair LoRA DPO run at this budget, on this model, measured on held-out data, does not meet the thresholds frozen before the run.
- It does not show anything about larger models, larger budgets, other preference data, or
  other decoding settings. One model, one host, greedy decoding.
- The refusal signal is substring refusal-prefix matching (`src/metrics.classify`), a cheap
  proxy, not an LLM judge panel.
