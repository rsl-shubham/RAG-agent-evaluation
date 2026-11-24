import unittest
from src.evaluators.agent_evaluators.evaluators import (
    evaluate_routing,
    evaluate_tool_selection,
    evaluate_tool_execution,
    evaluate_trajectory,
)

class TestAgentEvaluators(unittest.TestCase):
    def test_evaluate_routing(self):
        res = evaluate_routing("policy_inquiry", "policy_inquiry")
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)
        
        res_fail = evaluate_routing("general_chitchat", "policy_inquiry")
        self.assertFalse(res_fail.passed)
        self.assertEqual(res_fail.score, 0.0)

    def test_evaluate_tool_selection(self):
        res = evaluate_tool_selection("search_company_policy", "search_company_policy")
        self.assertTrue(res.passed)
        
        res_none = evaluate_tool_selection(None, None)
        self.assertTrue(res_none.passed)

    def test_evaluate_trajectory(self):
        actual = ["router:policy_inquiry", "retrieval:search_company_policy", "assessment:sufficient", "generation:policy_answer"]
        expected = ["router", "retrieval", "generation"]
        res = evaluate_trajectory(actual, expected)
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 1.0)

if __name__ == "__main__":
    unittest.main()
