import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("dpo_study", Path(__file__).parents[1] / "src/dpo_study.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_shuffled_control_changes_half_the_preferences():
    original = MOD.rows("train")
    shuffled = MOD.preference_dataset(True)
    changed = sum(row["chosen"] != original[i]["chosen"] for i, row in enumerate(shuffled))
    assert changed == len(original) // 2


def test_identical_pair_control_has_zero_margin():
    assert MOD.controls()["identical_pair_margin"] == 0.0


def test_frozen_manifest_matches_local_inputs():
    MOD.verify_manifest()
