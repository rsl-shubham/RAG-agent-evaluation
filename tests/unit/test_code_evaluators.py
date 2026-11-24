import unittest
from src.evaluators.code_based.metrics import (
    normalize_text,
    exact_match_evaluator,
    normalized_exact_match_evaluator,
    string_similarity_evaluator,
    numerical_fact_evaluator,
    source_validation_evaluator,
    abstention_evaluator,
    code_based_context_precision,
    code_based_context_recall,
)

class TestCodeEvaluators(unittest.TestCase):
    def test_normalize_text(self):
        text = "  Hello,   WORLD! 123...  "
        self.assertEqual(normalize_text(text), "hello world 123")

    def test_exact_match(self):
        res1 = exact_match_evaluator("20 days", "20 days")
        self.assertTrue(res1.passed)
        self.assertEqual(res1.score, 1.0)
        
        res2 = exact_match_evaluator("20 days", "20 Days")
        self.assertFalse(res2.passed)

    def test_normalized_exact_match(self):
        res = normalized_exact_match_evaluator("20 Days!", "20 days.")
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

    def test_string_similarity(self):
        res = string_similarity_evaluator("Employees get 20 days annual leave", "Employees receive 20 days paid annual leave")
        self.assertGreater(res.score, 0.4)

    def test_numerical_fact_evaluator(self):
        res = numerical_fact_evaluator("We offer 20 days off and 5,00,000 INR insurance.", ["20", "5,00,000"])
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

        res_fail = numerical_fact_evaluator("We offer 15 days off.", ["20"])
        self.assertFalse(res_fail.passed)
        self.assertEqual(res_fail.score, 0.0)

    def test_source_validation(self):
        res = source_validation_evaluator(["leave_policy.txt", "benefits.txt"], ["leave_policy.txt"])
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

    def test_abstention(self):
        res1 = abstention_evaluator("I could not find this information in the company documents.", should_abstain=True)
        self.assertTrue(res1.passed)

        res2 = abstention_evaluator("Employees receive 20 days off.", should_abstain=True)
        self.assertFalse(res2.passed)

    def test_context_precision_and_recall(self):
        ret = ["doc_a.txt", "doc_b.txt", "doc_c.txt"]
        exp = ["doc_a.txt"]
        p_res = code_based_context_precision(ret, exp)
        self.assertEqual(p_res.score, 1.0)  # Map score when first doc is relevant

        r_res = code_based_context_recall(ret, exp)
        self.assertTrue(r_res.passed)
        self.assertEqual(r_res.score, 1.0)

if __name__ == "__main__":
    unittest.main()
