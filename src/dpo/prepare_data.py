"""Build the frozen DPO-study splits from pinned public datasets.

Everything downstream (training, evaluation, controls) reads the manifest this
script writes, never the Hub directly. Splits are created BEFORE any training
and their SHA256 hashes are committed, so tuning on the held-out test set is
detectable, not just promised.

    python src/dpo/prepare_data.py

Writes:
    data/dpo/{train,val,test}.jsonl        preference pairs (local only, gitignored)
    data/dpo/advbench_50.jsonl             harmful eval prompts (local only, gitignored)
    data/dpo/arc_easy_100.jsonl            capability slice (local only, gitignored)
    manifests/dpo_data_manifest.json       revisions, counts, hashes (committed)

The split files themselves are not committed: hh-rlhf rows contain the human's
harmful requests and AdvBench rows are harmful instructions, and this repo's
data/ policy is that nothing committed requests anything dangerous. The
manifest pins the exact upstream revisions and the file hashes, so a rerun of
this script either reproduces the splits bit-for-bit or fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
import sys

from datasets import load_dataset

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "dpo"
MANIFEST = ROOT / "manifests" / "dpo_data_manifest.json"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

PREF_DATASET = "Anthropic/hh-rlhf"
PREF_DATA_DIR = "harmless-base"
PREF_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"

ADVBENCH_DATASET = "walledai/AdvBench"
ADVBENCH_REVISION = "9d4730540082fa4017450b65ca1c0e1d8d30446e"
ADVBENCH_N = 50

ARC_DATASET = "allenai/ai2_arc"
ARC_CONFIG = "ARC-Easy"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
ARC_N = 100

SPLIT_SEED = 0
N_TRAIN, N_VAL, N_TEST = 3000, 300, 300
MAX_CHARS = 2000  # crude pre-filter; the tokenizer-level cap is max_length=512 at train time

HUMAN, ASSISTANT = "\n\nHuman: ", "\n\nAssistant: "


def parse_single_turn(transcript: str) -> tuple[str, str] | None:
    """(human_prompt, assistant_response) if the transcript is exactly one turn."""
    if transcript.count(HUMAN) != 1 or transcript.count(ASSISTANT) != 1:
        return None
    if not transcript.startswith(HUMAN):
        return None
    body = transcript[len(HUMAN):]
    prompt, _, response = body.partition(ASSISTANT)
    prompt, response = prompt.strip(), response.strip()
    if not prompt or not response:
        return None
    return prompt, response


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"loading {PREF_DATASET}/{PREF_DATA_DIR} @ {PREF_REVISION[:12]} ...", flush=True)
    raw = load_dataset(PREF_DATASET, data_dir=PREF_DATA_DIR,
                       revision=PREF_REVISION, split="train")
    print(f"  upstream train rows: {len(raw)}")

    pool, seen_prompts = [], set()
    for idx, row in enumerate(raw):
        c = parse_single_turn(row["chosen"])
        r = parse_single_turn(row["rejected"])
        if c is None or r is None:
            continue
        if c[0] != r[0]:               # framings drifted between sides; skip
            continue
        if c[1] == r[1]:               # no preference signal
            continue
        prompt, chosen, rejected = c[0], c[1], r[1]
        if len(prompt) + max(len(chosen), len(rejected)) > MAX_CHARS:
            continue
        if prompt in seen_prompts:     # keep prompts unique so splits cannot overlap
            continue
        seen_prompts.add(prompt)
        pool.append({"source_index": idx, "prompt": prompt,
                     "chosen": chosen, "rejected": rejected})
    print(f"  filtered pool (single-turn, unique prompt, <= {MAX_CHARS} chars): {len(pool)}")

    need = N_TRAIN + N_VAL + N_TEST
    if len(pool) < need:
        sys.exit(f"pool too small: {len(pool)} < {need}")
    rng = random.Random(SPLIT_SEED)
    picked = rng.sample(range(len(pool)), need)
    train = [pool[i] for i in picked[:N_TRAIN]]
    val = [pool[i] for i in picked[N_TRAIN:N_TRAIN + N_VAL]]
    test = [pool[i] for i in picked[N_TRAIN + N_VAL:]]

    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "val.jsonl", val)
    write_jsonl(OUT / "test.jsonl", test)

    print(f"loading {ADVBENCH_DATASET} @ {ADVBENCH_REVISION[:12]} ...", flush=True)
    adv = load_dataset(ADVBENCH_DATASET, revision=ADVBENCH_REVISION, split="train")
    adv_rows = [{"source_index": i, "prompt": adv[i]["prompt"]} for i in range(ADVBENCH_N)]
    write_jsonl(OUT / "advbench_50.jsonl", adv_rows)

    print(f"loading {ARC_DATASET}/{ARC_CONFIG} @ {ARC_REVISION[:12]} ...", flush=True)
    arc = load_dataset(ARC_DATASET, ARC_CONFIG, revision=ARC_REVISION, split="test")
    arc_rows = []
    for i in range(ARC_N):
        row = arc[i]
        arc_rows.append({"source_index": i, "id": row["id"], "question": row["question"],
                         "choices": row["choices"]["text"],
                         "labels": row["choices"]["label"],
                         "answer_key": row["answerKey"]})
    write_jsonl(OUT / "arc_easy_100.jsonl", arc_rows)

    manifest = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "preference_dataset": {
            "id": PREF_DATASET, "data_dir": PREF_DATA_DIR, "revision": PREF_REVISION,
            "upstream_train_rows": len(raw),
            "filter": {"single_turn_only": True, "same_prompt_both_sides": True,
                       "chosen_neq_rejected": True, "unique_prompts": True,
                       "max_chars_prompt_plus_longest_response": MAX_CHARS},
            "filtered_pool_rows": len(pool),
            "split_seed": SPLIT_SEED,
            "splits": {
                name: {"rows": len(rows), "sha256": sha256_file(OUT / f"{name}.jsonl")}
                for name, rows in [("train", train), ("val", val), ("test", test)]
            },
        },
        "harmful_eval_set": {
            "id": ADVBENCH_DATASET, "revision": ADVBENCH_REVISION,
            "selection": f"first {ADVBENCH_N} rows of train split",
            "rows": ADVBENCH_N, "sha256": sha256_file(OUT / "advbench_50.jsonl"),
        },
        "capability_slice": {
            "id": ARC_DATASET, "config": ARC_CONFIG, "revision": ARC_REVISION,
            "selection": f"first {ARC_N} rows of test split",
            "rows": ARC_N, "sha256": sha256_file(OUT / "arc_easy_100.jsonl"),
        },
        "repo_eval_sets": {
            "benign": "data/over_refusal_100.jsonl (100 prompts, refusal is an error)",
            "illicit_control": "data/contrastive_pairs.jsonl last 8 refuse_side (refusal is correct)",
            "ppl_corpus": "data/ppl_corpus.txt",
        },
    }
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {MANIFEST}")
    for name in ("train", "val", "test", "advbench_50", "arc_easy_100"):
        p = OUT / f"{name}.jsonl"
        print(f"  {p.name}: {sum(1 for _ in open(p))} rows, sha256 {sha256_file(p)[:16]}...")


if __name__ == "__main__":
    main()
