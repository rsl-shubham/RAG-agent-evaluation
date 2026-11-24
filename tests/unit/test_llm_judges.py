import unittest
from unittest.mock import MagicMock, patch
from src.evaluators.llm_judges.schemas import (
    AnswerRelevanceResult,
    FaithfulnessResult,
    ClaimAssessment,
    AnswerCorrectnessResult,
    ContextPrecisionResult,
    ChunkRelevanceAssessment,
    ContextRecallResult,
    FactSupportAssessment,
)
from src.evaluators.llm_judges.judges import (
    evaluate_answer_relevance,
    evaluate_faithfulness,
    evaluate_answer_correctness,
    evaluate_context_precision,
    evaluate_context_recall,
)

class TestLLMJudges(unittest.TestCase):
    @patch("src.evaluators.llm_judges.judges.get_judge_llm")
    def test_evaluate_answer_relevance(self, mock_get_llm):
        mock_res = AnswerRelevanceResult(
            score=5,
            verdict="relevant",
            reason="Directly answers the question.",
        )
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_res
        
        with patch("src.evaluators.llm_judges.judges.answer_relevance_prompt") as mock_prompt:
            mock_prompt.__or__.return_value = mock_chain
            res = evaluate_answer_relevance("What is the leave?", "20 days annual leave.")
            self.assertEqual(res.score, 1.0)
            self.assertTrue(res.passed)

    @patch("src.evaluators.llm_judges.judges.get_judge_llm")
    def test_evaluate_faithfulness(self, mock_get_llm):
        mock_res = FaithfulnessResult(
            claims=[
                ClaimAssessment(claim="20 days leave", verdict="supported", evidence="20 days", reason="matches"),
            ],
            overall_reason="Fully grounded",
        )
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_res
        
        with patch("src.evaluators.llm_judges.judges.faithfulness_prompt") as mock_prompt:
            mock_prompt.__or__.return_value = mock_chain
            res = evaluate_faithfulness("q", "20 days context", "20 days answer")
            self.assertEqual(res.score, 1.0)
            self.assertTrue(res.passed)

if __name__ == "__main__":
    unittest.main()
