"""Response-level degeneration and capability checks for steering sweeps.

Used by tests/test_degenerate_outputs.py and paired sweep analysis.
Reports text controls beside boolean adjudications without inferring safety.
"""
from __future__ import annotations

import re
from collections import Counter


def normalized_response(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def degeneration_flags(text: str, *, min_tokens: int = 12) -> dict[str, bool]:
    tokens = re.findall(r"\w+", normalized_response(text))
    repeated = bool(tokens) and max(Counter(tokens).values()) >= max(4, len(tokens) // 3)
    sentence = [normalized_response(x) for x in re.split(r"[.!?]+", text) if x.strip()]
    stock = len(sentence) >= 3 and len(set(sentence)) == 1
    return {"too_short": len(tokens) < min_tokens, "repeated_token": repeated,
            "repeated_sentence": stock}


def summarize(rows: list[dict]) -> dict:
    """Summarize paired baseline/intervention records without dropping failures."""
    if not rows:
        raise ValueError("at least one paired record is required")
    ids = [r.get("case_id") for r in rows]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("case_id values must be present and unique")
    required = {"baseline_text", "steered_text", "baseline_success", "steered_success"}
    if any(not required.issubset(r) for r in rows):
        raise ValueError("each row must include both texts and both adjudicated outcomes")
    if any(not isinstance(r[field], bool) for r in rows
           for field in ("baseline_success", "steered_success")):
        raise ValueError("adjudicated outcomes must be boolean")
    n = len(rows)
    return {
        "n_paired_cases": n,
        "baseline_success_rate": sum(bool(r["baseline_success"]) for r in rows) / n,
        "steered_success_rate": sum(bool(r["steered_success"]) for r in rows) / n,
        "baseline_degenerate_rate": sum(any(degeneration_flags(r["baseline_text"]).values()) for r in rows) / n,
        "steered_degenerate_rate": sum(any(degeneration_flags(r["steered_text"]).values()) for r in rows) / n,
        "capability_regressions": sum(bool(r["baseline_success"]) and not bool(r["steered_success"]) for r in rows),
        "capability_gains": sum(not bool(r["baseline_success"]) and bool(r["steered_success"]) for r in rows),
    }
