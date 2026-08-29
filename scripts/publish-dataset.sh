#!/usr/bin/env bash
# Upload the prompt sets to Hugging Face, with a card generated from the run.
#
#   hf auth login                          # once
#   scripts/publish-dataset.sh [repo]
#
# The card is generated before anything uploads, and the results file it was
# generated from travels with it, so the table on the dataset page can be checked
# against the run rather than taken on trust.
set -euo pipefail

repo="${1:-lgoyal/open-refusal-steering}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hf="${HF:-hf}"

command -v "$hf" >/dev/null || { echo "install the CLI: pip install -U huggingface_hub" >&2; exit 1; }
"$hf" auth whoami >/dev/null 2>&1 || { echo "log in first: hf auth login" >&2; exit 1; }

staging="$(mktemp -d)"
echo "generating the card from results/sweep_partial_n100.csv"
python3 "$here/scripts/dataset-card.py" > "$staging/README.md"

# The prompt sets, their construction note, and their own licence file. The
# perplexity corpus ships too: the capability-tax column cannot be reproduced
# without the exact text it was measured on.
for f in over_refusal_100.jsonl contrastive_pairs.jsonl ppl_corpus.txt CONSTRUCTION.md LICENSE; do
  cp "$here/data/$f" "$staging/$f"
done
# The run the card's table was generated from, so the numbers are checkable.
cp "$here/results/sweep_partial_n100.csv" "$staging/sweep_partial_n100.csv"

echo
sed -n '1,40p' "$staging/README.md"
echo
echo "files:"
ls -1 "$staging"
echo
read -r -p "publish this to $repo? [y/N] " reply
[ "$reply" = "y" ] || { echo "stopped"; exit 0; }

"$hf" repo create "$repo" --repo-type dataset --exist-ok
"$hf" upload "$repo" "$staging" . --repo-type dataset \
  --commit-message "The prompt sets, and the run the card was generated from"

echo
echo "https://huggingface.co/datasets/$repo"
