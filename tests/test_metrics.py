"""Sanity checks for the response classifier.

    python -m unittest discover -s tests
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from metrics import classify, is_degenerate, is_refusal  # noqa: E402


class TestClassifier(unittest.TestCase):
    def test_refusals(self):
        for text in [
            "I'm sorry, but I can't help with that request.",
            "I cannot provide information on this topic.",
            "As an AI language model, I must decline.",
            "I'm unable to assist with that.",
        ]:
            self.assertTrue(is_refusal(text), text)
            self.assertEqual(classify(text), "REFUSAL", text)

    def test_answers(self):
        for text in [
            "The maximum daily dose of acetaminophen for a healthy adult is 4,000 mg, "
            "though many clinicians recommend staying under 3,000 mg. Exceeding it "
            "risks hepatotoxicity.",
            "SQL injection happens when untrusted input is concatenated into a query. "
            "Use parameterised statements.",
        ]:
            self.assertFalse(is_refusal(text), text)
            self.assertEqual(classify(text), "ANSWERED", text)

    def test_degenerate(self):
        for text in ["the " * 30, "a" * 40, "", "   "]:
            self.assertTrue(is_degenerate(text), repr(text[:20]))
            self.assertEqual(classify(text), "DEGENERATE", repr(text[:20]))

    def test_degeneracy_precedes_refusal(self):
        # a looping refusal is a generation failure, not a clean refusal
        self.assertEqual(classify("I'm sorry. " * 30), "DEGENERATE")

    def test_normal_prose_is_not_degenerate(self):
        text = (
            "Natural selection acts only by the preservation and accumulation of small "
            "inherited modifications, each profitable to the preserved being."
        )
        self.assertFalse(is_degenerate(text))


if __name__ == "__main__":
    unittest.main()
