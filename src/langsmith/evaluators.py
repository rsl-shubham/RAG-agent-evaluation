from typing import Dict, Any, List
from src.evaluators.code_based.metrics import (
    exact_match_evaluator,
    normalized_exact_match_evaluator,
    string_similarity_evaluator,
    semantic_similarity_evaluator,
    numerical_fact_evaluator,
    source_validation_evaluator,
    abstention_evaluator,
)
from src.evaluators.llm_judges.judges import (
    evaluate_answer_relevance,
    evaluate_faithfulness,
    evaluate_answer_correctness,
    evaluate_context_precision,
    evaluate_context_recall,
)

# --- LangSmith Code-Based Evaluators ---

def langsmith_normalized_exact_match(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith evaluator for normalized exact match."""
    actual = run.outputs.get("answer", "") if run.outputs else ""
    expected = example.outputs.get("reference", "") if example.outputs else ""
    res = normalized_exact_match_evaluator(actual, expected)
    return {
        "key": "normalized_exact_match",
        "score": res.score,
        "comment": res.explanation,
    }

def langsmith_string_similarity(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith evaluator for token string similarity."""
    actual = run.outputs.get("answer", "") if run.outputs else ""
    expected = example.outputs.get("reference", "") if example.outputs else ""
    res = string_similarity_evaluator(actual, expected)
    return {
        "key": "string_similarity",
        "score": res.score,
        "comment": res.explanation,
    }

def langsmith_source_validation(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith evaluator for retrieved source matching."""
    actual_sources = run.outputs.get("sources", []) if run.outputs else []
    expected_sources = example.outputs.get("expected_sources", []) if example.outputs else []
    res = source_validation_evaluator(actual_sources, expected_sources)
    return {
        "key": "source_recall",
        "score": res.score,
        "comment": res.explanation,
    }

def langsmith_abstention(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith evaluator for abstention detection."""
    actual = run.outputs.get("answer", "") if run.outputs else ""
    should_abstain = example.outputs.get("should_abstain", False) if example.outputs else False
    res = abstention_evaluator(actual, should_abstain=should_abstain)
    return {
        "key": "correct_abstention",
        "score": res.score,
        "comment": res.explanation,
    }

# --- LangSmith LLM Judges ---

def langsmith_llm_relevance(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith LLM judge for answer relevance."""
    question = example.inputs.get("question", "") if example.inputs else ""
    answer = run.outputs.get("answer", "") if run.outputs else ""
    res = evaluate_answer_relevance(question, answer)
    return {
        "key": "llm_answer_relevance",
        "score": res.score,
        "comment": res.explanation,
    }

def langsmith_llm_faithfulness(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith LLM judge for faithfulness against retrieved context."""
    question = example.inputs.get("question", "") if example.inputs else ""
    answer = run.outputs.get("answer", "") if run.outputs else ""
    context = run.outputs.get("context", "") if run.outputs else ""
    res = evaluate_faithfulness(question, context, answer)
    return {
        "key": "llm_faithfulness",
        "score": res.score,
        "comment": res.explanation,
    }

def langsmith_llm_correctness(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith LLM judge for answer correctness against ground truth."""
    question = example.inputs.get("question", "") if example.inputs else ""
    answer = run.outputs.get("answer", "") if run.outputs else ""
    reference = example.outputs.get("reference", "") if example.outputs else ""
    res = evaluate_answer_correctness(question, reference, answer)
    return {
        "key": "llm_correctness",
        "score": res.score,
        "comment": res.explanation,
    }

def langsmith_llm_context_precision(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith LLM judge for retrieved context precision."""
    question = example.inputs.get("question", "") if example.inputs else ""
    retrieved_docs = run.outputs.get("retrieved_documents", []) if run.outputs else []
    res = evaluate_context_precision(question, retrieved_docs)
    return {
        "key": "llm_context_precision_map",
        "score": res.score,
        "comment": res.explanation,
    }

def langsmith_llm_context_recall(run: Any, example: Any) -> Dict[str, Any]:
    """LangSmith LLM judge for context recall against reference answer."""
    question = example.inputs.get("question", "") if example.inputs else ""
    reference = example.outputs.get("reference", "") if example.outputs else ""
    context = run.outputs.get("context", "") if run.outputs else ""
    res = evaluate_context_recall(question, reference, context)
    return {
        "key": "llm_context_recall",
        "score": res.score,
        "comment": res.explanation,
    }
