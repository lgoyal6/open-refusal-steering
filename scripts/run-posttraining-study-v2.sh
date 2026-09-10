#!/usr/bin/env bash
# One command for the pre-registered post-training v2 study.
#
# Verifies the frozen inputs against results/posttraining-v2-manifest.json and
# refuses to run if any hash has drifted, runs the tests, then runs every arm at
# every seed, the negative controls, and the retained training-path checks, and
# writes the four committed artefacts.
#
# Local, single host, Apple MPS. No remote compute, no paid API, cost 0.
# Expect several hours. The study caches each finished arm and each finished
# evaluation under .agent-work/, so re-running resumes instead of restarting.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}

say() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$1"
}

say "interpreter $(pwd)/${PYTHON}"
"${PYTHON}" --version
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'mps available', torch.backends.mps.is_available(), 'cuda available', torch.cuda.is_available())"

# The v1 study's splits are gitignored and its committed tests verify them, so
# re-derive them from the same pinned revisions before running the suite.
say "re-deriving the v1 study splits so its committed tests have their inputs"
"${PYTHON}" src/dpo_prepare.py > /dev/null

say "re-deriving the v2 splits and verifying them against the frozen manifest"
"${PYTHON}" src/pt2_prepare.py

say "running the test suite"
"${PYTHON}" -m pytest -q

say "running every arm at every seed, the controls, and the retained checks"
"${PYTHON}" src/pt2_study.py all

say "checking the four artefacts exist"
test -f results/posttraining-v2-manifest.json
test -f results/posttraining-v2.json
test -f results/posttraining-v2-generations.jsonl
test -f results/posttraining-v2-report.md
wc -l results/posttraining-v2-generations.jsonl

say "gate verdict and claim state"
"${PYTHON}" -c "import json; r = json.load(open('results/posttraining-v2.json')); print('gate', r['gate']['verdict']); print('failed clauses', r['gate']['failed_clauses']); print('controls valid', r['controls']['study_valid']); print('claim state', r['claim_state'])"

say "done"
