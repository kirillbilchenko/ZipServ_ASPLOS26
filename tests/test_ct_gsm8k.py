"""Check scoring boundaries that could hide a quality regression."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("ct_gsm8k", Path(__file__).parents[1] / "tools/eval_ct_gsm8k.py")
gsm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gsm)


def response(text, reason="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": reason}]}


class ScoringTests(unittest.TestCase):
    def test_final_answer_overrides_reasoning_numbers(self):
        row = gsm.score(response("<think>Wrong: 999</think>There are 12. #### 12"), "12")
        self.assertTrue(row["answer_exact_match"])
        self.assertTrue(row["strict_format_match"])

    def test_unfinished_reasoning_and_truncations_do_not_receive_credit(self):
        self.assertFalse(gsm.score(response("<think>12"), "12")["answer_exact_match"])
        row = gsm.score(response("#### 12", "length"), "12")
        self.assertTrue(row["truncated"])
        self.assertFalse(row["answer_exact_match"])

    def test_numeric_value_and_format_are_separate(self):
        row = gsm.score(response("It costs $1,200.50 in total."), "1200.5")
        self.assertTrue(row["answer_exact_match"])
        self.assertFalse(row["strict_format_match"])
        negative = gsm.score(response("#### -$1,200.50"), "-1200.5")
        self.assertTrue(negative["strict_format_match"])
        self.assertTrue(negative["answer_exact_match"])

    def test_api_errors_and_missing_questions_keep_full_denominator(self):
        rows = [{"source_index": 0, "answer_exact_match": True, "strict_format_match": True},
                {"source_index": 1, "answer_exact_match": False, "strict_format_match": False, "api_error": True}]
        summary = gsm.summarize(rows, 1319)
        self.assertEqual(summary["accuracy"], 1 / 1319)
        self.assertEqual(summary["api_errors"], 1)
        self.assertFalse(summary["complete"])
        self.assertIsNone(summary["accuracy_wilson95"])
        with self.assertRaises(ValueError):
            gsm.summarize(rows + rows, 1319)

    def test_dataset_replacement_is_rejected(self):
        with self.assertRaises(ValueError):
            gsm.freeze(b'{"question":"edited","answer":"#### 1"}\n')


if __name__ == "__main__":
    unittest.main()
