from typing import List, Dict, Any, Optional
from src.config.models import get_judge_llm
from src.evaluators.base import EvaluationResult
from src.evaluators.llm_judges.schemas import (
    AnswerRelevanceResult,
    FaithfulnessResult,
    AnswerCorrectnessResult,
    ContextPrecisionResult,
    ContextRecallResult,
    HallucinationResult,
)
from src.evaluators.llm_judges.prompts import (
    answer_relevance_prompt,
    faithfulness_prompt,
    answer_correctness_prompt,
    context_precision_prompt,
    context_recall_prompt,
    hallucination_prompt,
)

def evaluate_answer_relevance(
    question: str,
    answer: str,
    judge_model: Optional[str] = None,
) -> EvaluationResult:
    """Judge relevance of generated answer to question (1-5 scale normalized to 0.0-1.0)."""
    judge_llm = get_judge_llm(model=judge_model).with_structured_output(AnswerRelevanceResult)
    chain = answer_relevance_prompt | judge_llm
    
    res: AnswerRelevanceResult = chain.invoke({"question": question, "answer": answer})
    normalized_score = round((res.score - 1) / 4.0, 4)
    
    return EvaluationResult(
        evaluator_name="llm_answer_relevance",
        score=normalized_score,
        passed=res.score >= 4,
        explanation=f"Relevance: {res.score}/5 ({res.verdict}). {res.reason}",
        details={"raw_score": res.score, "verdict": res.verdict, "reason": res.reason},
    )

def evaluate_faithfulness(
    question: str,
    context: str,
    answer: str,
    judge_model: Optional[str] = None,
) -> EvaluationResult:
    """Judge claim-level faithfulness against retrieved context."""
    judge_llm = get_judge_llm(model=judge_model).with_structured_output(FaithfulnessResult)
    chain = faithfulness_prompt | judge_llm
    
    res: FaithfulnessResult = chain.invoke({
        "question": question,
        "context": context,
        "answer": answer,
    })
    
    claims = res.claims
    if not claims:
        return EvaluationResult(
            evaluator_name="llm_faithfulness",
            score=1.0,
            passed=True,
            explanation="No factual claims identified in answer to evaluate.",
            details={"claims": [], "overall_reason": res.overall_reason},
        )
        
    supported_count = sum(1 for c in claims if c.verdict == "supported")
    contradicted_count = sum(1 for c in claims if c.verdict == "contradicted")
    score = round(supported_count / len(claims), 4)
    
    passed = (score == 1.0) and (contradicted_count == 0)
    return EvaluationResult(
        evaluator_name="llm_faithfulness",
        score=score,
        passed=passed,
        explanation=f"{supported_count}/{len(claims)} claims supported by context ({res.overall_reason}).",
        details={
            "total_claims": len(claims),
            "supported_claims": supported_count,
            "contradicted_claims": contradicted_count,
            "claims": [c.model_dump() for c in claims],
            "overall_reason": res.overall_reason,
        },
    )

def evaluate_answer_correctness(
    question: str,
    reference_answer: str,
    generated_answer: str,
    judge_model: Optional[str] = None,
) -> EvaluationResult:
    """Judge factual correctness against human reference answer."""
    judge_llm = get_judge_llm(model=judge_model).with_structured_output(AnswerCorrectnessResult)
    chain = answer_correctness_prompt | judge_llm
    
    res: AnswerCorrectnessResult = chain.invoke({
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,
    })
    
    normalized_score = round((res.score - 1) / 4.0, 4)
    return EvaluationResult(
        evaluator_name="llm_answer_correctness",
        score=normalized_score,
        passed=res.score >= 4,
        explanation=f"Correctness: {res.score}/5 ({res.verdict}). {res.reason}",
        details={
            "raw_score": res.score,
            "verdict": res.verdict,
            "missing_facts": res.missing_facts,
            "incorrect_claims": res.incorrect_claims,
            "extra_unverified_claims": res.extra_unverified_claims,
            "reason": res.reason,
        },
    )

def evaluate_context_precision(
    question: str,
    chunks: List[Dict[str, Any]],
    judge_model: Optional[str] = None,
) -> EvaluationResult:
    """
    Evaluate chunk-level relevance and calculate basic Precision@K and Mean Average Precision (MAP).
    Handles missing or out-of-order chunk indices safely in Python.
    """
    if not chunks:
        return EvaluationResult(
            evaluator_name="llm_context_precision",
            score=0.0,
            passed=False,
            explanation="No retrieved chunks provided.",
            details={},
        )
        
    formatted_chunks = []
    for idx, c in enumerate(chunks):
        content = c.get("content", "")
        source = c.get("source", "Unknown")
        formatted_chunks.append(f"--- Chunk Index {idx} (Source: {source}) ---\n{content}")
    chunks_text = "\n\n".join(formatted_chunks)
    
    judge_llm = get_judge_llm(model=judge_model).with_structured_output(ContextPrecisionResult)
    chain = context_precision_prompt | judge_llm
    
    res: ContextPrecisionResult = chain.invoke({
        "question": question,
        "chunks_text": chunks_text,
    })
    
    # Map returned assessments by chunk_index
    assessment_map = {a.chunk_index: a.is_relevant for a in res.assessments}
    
    # Validate each index from 0 to len(chunks)-1
    relevance_flags = []
    for idx in range(len(chunks)):
        # Default to False if judge skipped an index
        relevance_flags.append(1 if assessment_map.get(idx, False) else 0)
        
    total_relevant = sum(relevance_flags)
    precision_at_k = total_relevant / len(chunks)
    
    # Calculate ranking-aware Average Precision (AP)
    running_relevant = 0
    precisions = []
    for rank, flag in enumerate(relevance_flags, start=1):
        if flag == 1:
            running_relevant += 1
            precisions.append(running_relevant / rank)
            
    map_score = sum(precisions) / total_relevant if total_relevant > 0 else 0.0
    
    score = round(map_score, 4)
    return EvaluationResult(
        evaluator_name="llm_context_precision",
        score=score,
        passed=score >= 0.5,
        explanation=f"Context Precision MAP: {score:.4f} (P@{len(chunks)}: {precision_at_k:.2f}). {res.summary_reason}",
        details={
            "precision_at_k": round(precision_at_k, 4),
            "map_score": score,
            "relevance_flags": relevance_flags,
            "assessments": [a.model_dump() for a in res.assessments],
            "summary_reason": res.summary_reason,
        },
    )

def evaluate_context_recall(
    question: str,
    reference_answer: str,
    context: str,
    judge_model: Optional[str] = None,
) -> EvaluationResult:
    """
    Identify reference atomic facts and evaluate retrieved context support in Python.
    """
    if not reference_answer or not context:
        return EvaluationResult(
            evaluator_name="llm_context_recall",
            score=0.0,
            passed=False,
            explanation="Empty reference answer or retrieved context.",
            details={},
        )
        
    judge_llm = get_judge_llm(model=judge_model).with_structured_output(ContextRecallResult)
    chain = context_recall_prompt | judge_llm
    
    res: ContextRecallResult = chain.invoke({
        "question": question,
        "reference_answer": reference_answer,
        "context": context,
    })
    
    facts = res.atomic_facts
    if not facts:
        return EvaluationResult(
            evaluator_name="llm_context_recall",
            score=1.0,
            passed=True,
            explanation="No required atomic facts identified.",
            details={"atomic_facts": []},
        )
        
    supported_count = sum(1 for f in facts if f.is_supported)
    recall_score = round(supported_count / len(facts), 4)
    
    return EvaluationResult(
        evaluator_name="llm_context_recall",
        score=recall_score,
        passed=recall_score >= 0.8,
        explanation=f"Context Recall: {supported_count}/{len(facts)} reference facts supported ({recall_score * 100:.1f}%). {res.summary_reason}",
        details={
            "total_facts": len(facts),
            "supported_facts": supported_count,
            "atomic_facts": [f.model_dump() for f in facts],
            "summary_reason": res.summary_reason,
        },
    )

def evaluate_hallucination(
    question: str,
    context: str,
    answer: str,
    judge_model: Optional[str] = None,
) -> EvaluationResult:
    """Detect whether answer contains fabricated or unsupported claims."""
    judge_llm = get_judge_llm(model=judge_model).with_structured_output(HallucinationResult)
    chain = hallucination_prompt | judge_llm
    
    res: HallucinationResult = chain.invoke({
        "question": question,
        "context": context,
        "answer": answer,
    })
    
    passed = not res.has_hallucinations
    score = 1.0 if passed else 0.0
    
    return EvaluationResult(
        evaluator_name="llm_hallucination_detection",
        score=score,
        passed=passed,
        explanation=f"Hallucination severity: {res.severity}. {res.reason}",
        details={
            "has_hallucinations": res.has_hallucinations,
            "hallucinated_claims": res.hallucinated_claims,
            "severity": res.severity,
            "reason": res.reason,
        },
    )
