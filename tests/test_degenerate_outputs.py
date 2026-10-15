"""Check response degeneration and adjudication input handling.

These fixtures exercise eval_capability.py without model generation.
Invalid labels must fail before they can inflate success or gain counts.
"""
import pytest

from eval_capability import degeneration_flags, summarize


def test_repeated_stock_response_is_flagged_even_if_success_label_is_true():
    text = "I cannot help with that. " * 8
    assert degeneration_flags(text)["repeated_sentence"]
    out = summarize([{"case_id": "x", "baseline_text": "A varied, useful answer with enough detail.",
                      "steered_text": text, "baseline_success": False, "steered_success": True}])
    assert out["steered_success_rate"] == 1
    assert out["steered_degenerate_rate"] == 1


def test_missing_adjudication_and_duplicate_case_ids_are_rejected():
    with pytest.raises(ValueError):
        summarize([{"case_id": "x"}])
    row = {"case_id": "x", "baseline_text": "answer with adequate detail",
           "steered_text": "answer with adequate detail", "baseline_success": True,
           "steered_success": True}
    with pytest.raises(ValueError):
        summarize([row, dict(row)])


@pytest.mark.parametrize("field", ["baseline_success", "steered_success"])
@pytest.mark.parametrize("label", ["false", "true", 0, 1, None, [], {}])
def test_non_boolean_adjudication_is_rejected(field, label):
    row = {"case_id": "typed", "baseline_text": "A baseline answer.",
           "steered_text": "An intervention answer.", "baseline_success": False,
           "steered_success": False}
    row[field] = label
    with pytest.raises(ValueError, match="boolean"):
        summarize([row])
