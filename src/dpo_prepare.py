"""Freeze public preference and capability data before any model training."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dpo-study"
MANIFEST = ROOT / "results" / "dpo-study-manifest.json"

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
PREF = "Anthropic/hh-rlhf"
PREF_REV = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
ARC = "allenai/ai2_arc"
ARC_REV = "210d026faf9955653af8916fad021475a3f00453"
SEED = 20260908
COUNTS = {"train": 24, "validation": 8, "test": 8}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def split_pair(text: str) -> tuple[str, str] | None:
    human, assistant = "\n\nHuman: ", "\n\nAssistant: "
    if not text.startswith(human) or text.count(human) != 1 or text.count(assistant) != 1:
        return None
    prompt, sep, answer = text[len(human):].partition(assistant)
    if not sep or not prompt.strip() or not answer.strip():
        return None
    return prompt.strip(), answer.strip()


def assert_disjoint(parts: dict[str, list[dict]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in parts.items():
        for row in rows:
            key = hashlib.sha256(row["prompt"].encode()).hexdigest()
            if key in seen:
                raise ValueError(f"train/test overlap: prompt in {seen[key]} and {split}")
            seen[key] = split


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(PREF, data_dir="harmless-base", revision=PREF_REV, split="train")
    pool = []
    seen = set()
    for source_index, row in enumerate(raw):
        chosen, rejected = split_pair(row["chosen"]), split_pair(row["rejected"])
        if not chosen or not rejected or chosen[0] != rejected[0] or chosen[1] == rejected[1]:
            continue
        if chosen[0] in seen or len(chosen[0]) + max(len(chosen[1]), len(rejected[1])) > 1800:
            continue
        seen.add(chosen[0])
        pool.append({"source_index": source_index, "prompt": chosen[0],
                     "chosen": chosen[1], "rejected": rejected[1]})
    rng = random.Random(SEED)
    rng.shuffle(pool)
    parts, offset = {}, 0
    for name, count in COUNTS.items():
        parts[name] = pool[offset:offset + count]
        offset += count
    assert_disjoint(parts)
    for name, rows in parts.items():
        write_jsonl(OUT / f"{name}.jsonl", rows)

    arc_raw = load_dataset("allenai/ai2_arc", "ARC-Easy", revision=ARC_REV, split="test")
    arc = []
    for index in range(16):
        row = arc_raw[index]
        arc.append({"source_index": index, "question": row["question"],
                    "choices": row["choices"]["text"], "labels": row["choices"]["label"],
                    "answer": row["answerKey"]})
    write_jsonl(OUT / "arc-easy.jsonl", arc)
    files = {p.name: {"rows": sum(1 for _ in p.open()), "sha256": digest(p)}
             for p in sorted(OUT.glob("*.jsonl"))}
    manifest = {
        "created_before_training": True,
        "model": {"id": MODEL, "revision": MODEL_REV},
        "preference_dataset": {"id": PREF, "config": "harmless-base", "revision": PREF_REV},
        "capability_dataset": {"id": ARC, "config": "ARC-Easy", "revision": ARC_REV},
        "seed": SEED,
        "selection": "seeded shuffle after single-turn, same-prompt, unique-prompt, <=1800-char filter",
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
