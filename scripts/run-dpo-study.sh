#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python src/dpo_prepare.py
.venv/bin/python -m pytest -q tests/test_metrics.py tests/test_dpo_prepare.py tests/test_dpo_study.py
.venv/bin/python src/dpo_study.py all
