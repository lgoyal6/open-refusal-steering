import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("dpo_prepare", Path(__file__).parents[1] / "src/dpo_prepare.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_overlap_detector_accepts_disjoint_splits():
    MOD.assert_disjoint({"train": [{"prompt": "a"}], "test": [{"prompt": "b"}]})


def test_overlap_detector_rejects_train_test_leakage():
    with pytest.raises(ValueError, match="train/test overlap"):
        MOD.assert_disjoint({"train": [{"prompt": "same"}], "test": [{"prompt": "same"}]})


def test_single_turn_parser_rejects_multiturn_transcript():
    assert MOD.split_pair("\n\nHuman: a\n\nAssistant: b\n\nHuman: c\n\nAssistant: d") is None
