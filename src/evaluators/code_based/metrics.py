import math
import re
import string
from typing import List, Optional, Dict, Any, Union
from src.evaluators.base import EvaluationResult

def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def exact_match_evaluator(
    actual_answer: str,
    reference_answer: str,
) -> EvaluationResult:
    """Check literal exact match (case-sensitive, whitespace-preserving)."""
    if not reference_answer or not actual_answer:
        return EvaluationResult(
            evaluator_name="exact_match",
            score=0.0,
            passed=False,
            explanation="Empty actual or reference answer.",
            details={"actual": actual_answer, "reference": reference_answer},
        )
    is_match = (actual_answer.strip() == reference_answer.strip())
    score = 1.0 if is_match else 0.0
    return EvaluationResult(
        evaluator_name="exact_match",
        score=score,
        passed=is_match,
        explanation="Exact string match." if is_match else "Strings do not match exactly.",
        details={"actual": actual_answer, "reference": reference_answer},
    )

def normalized_exact_match_evaluator(
    actual_answer: str,
    reference_answer: str,
) -> EvaluationResult:
    """Check case-insensitive, punctuation-stripped exact match."""
    norm_actual = normalize_text(actual_answer)
    norm_ref = normalize_text(reference_answer)
    is_match = (norm_actual == norm_ref) and (norm_actual != "")
    score = 1.0 if is_match else 0.0
    return EvaluationResult(
        evaluator_name="normalized_exact_match",
        score=score,
        passed=is_match,
        explanation="Normalized exact match achieved." if is_match else "Normalized text does not match reference.",
        details={"normalized_actual": norm_actual, "normalized_reference": norm_ref},
    )

def string_similarity_evaluator(
    actual_answer: str,
    reference_answer: str,
    threshold: float = 0.7,
) -> EvaluationResult:
    """
    Token-based Jaccard similarity and character n-gram overlap between actual and reference answers.
    """
    tokens_actual = set(normalize_text(actual_answer).split())
    tokens_ref = set(normalize_text(reference_answer).split())
    
    if not tokens_actual or not tokens_ref:
        return EvaluationResult(
            evaluator_name="string_similarity",
            score=0.0,
            passed=False,
            explanation="Empty tokens in answer or reference.",
            details={"token_jaccard": 0.0},
        )
        
    intersection = tokens_actual.intersection(tokens_ref)
    union = tokens_actual.union(tokens_ref)
    jaccard = len(intersection) / len(union) if union else 0.0
    
    score = round(jaccard, 4)
    passed = score >= threshold
    return EvaluationResult(
        evaluator_name="string_similarity",
        score=score,
        passed=passed,
        explanation=f"Token Jaccard similarity is {score:.4f} (threshold: {threshold}).",
        details={"tokens_actual": len(tokens_actual), "tokens_ref": len(tokens_ref), "overlap": len(intersection)},
    )

def semantic_similarity_evaluator(
    actual_answer: str,
    reference_answer: str,
    threshold: float = 0.82,
    embeddings_model=None,
) -> EvaluationResult:
    """
    Compute cosine similarity between embeddings of actual and reference answers.
    """
    from src.config.models import get_embeddings
    emb = embeddings_model or get_embeddings()
    
    vec_actual = emb.embed_query(actual_answer)
    vec_ref = emb.embed_query(reference_answer)
    
    dot_product = sum(a * b for a, b in zip(vec_actual, vec_ref))
    mag_a = math.sqrt(sum(a * a for a in vec_actual))
    mag_b = math.sqrt(sum(b * b for b in vec_ref))
    
    if mag_a == 0 or mag_b == 0:
        sim = 0.0
    else:
        sim = dot_product / (mag_a * mag_b)
        
    score = round(max(0.0, min(1.0, sim)), 4)
    passed = score >= threshold
    return EvaluationResult(
        evaluator_name="semantic_similarity",
        score=score,
        passed=passed,
        explanation=f"Embedding cosine similarity is {score:.4f} (threshold: {threshold}).",
        details={"similarity": score, "threshold": threshold},
    )

def keyword_coverage_evaluator(
    actual_answer: str,
    expected_keywords: List[str],
    threshold: float = 0.8,
) -> EvaluationResult:
    """
    Measure what percentage of mandatory keywords/terms are covered in the generated answer.
    """
    if not expected_keywords:
        return EvaluationResult(
            evaluator_name="keyword_coverage",
            score=None,
            passed=None,
            explanation="No expected keywords provided; evaluator not applicable.",
            details={},
        )
        
    norm_actual = normalize_text(actual_answer)
    found = []
    missing = []
    for kw in expected_keywords:
        norm_kw = normalize_text(kw)
        if norm_kw in norm_actual:
            found.append(kw)
        else:
            missing.append(kw)
            
    coverage = len(found) / len(expected_keywords)
    score = round(coverage, 4)
    passed = score >= threshold
    return EvaluationResult(
        evaluator_name="keyword_coverage",
        score=score,
        passed=passed,
        explanation=f"Found {len(found)}/{len(expected_keywords)} expected keywords ({score * 100:.1f}%).",
        details={"found_keywords": found, "missing_keywords": missing, "total_expected": len(expected_keywords)},
    )

def numerical_fact_evaluator(
    actual_answer: str,
    expected_numbers: List[Union[int, float, str]],
) -> EvaluationResult:
    """
    Verify that specific numerical quantities (e.g. '20', '15,000', '5,00,000') exist accurately in the response.
    """
    if not expected_numbers:
        return EvaluationResult(
            evaluator_name="numerical_fact_validation",
            score=None,
            passed=None,
            explanation="No expected numbers provided.",
            details={},
        )
        
    # Clean text numbers (remove commas in 15,000 / 5,00,000 for standard regex matching)
    cleaned_text = actual_answer.replace(",", "")
    
    found = []
    missing = []
    for num in expected_numbers:
        clean_num = str(num).replace(",", "")
        pattern = rf"\b{re.escape(clean_num)}\b"
        if re.search(pattern, cleaned_text):
            found.append(num)
        else:
            missing.append(num)
            
    score = len(found) / len(expected_numbers)
    passed = len(missing) == 0
    return EvaluationResult(
        evaluator_name="numerical_fact_validation",
        score=round(score, 4),
        passed=passed,
        explanation=f"Found {len(found)} of {len(expected_numbers)} expected numerical facts." if passed else f"Missing expected numbers: {missing}",
        details={"found": found, "missing": missing},
    )

def source_validation_evaluator(
    actual_sources: List[str],
    expected_sources: List[str],
) -> EvaluationResult:
    """
    Validate that retrieved sources match expected document files.
    """
    if not expected_sources:
        return EvaluationResult(
            evaluator_name="source_validation",
            score=None,
            passed=None,
            explanation="No expected sources defined for this test case.",
            details={"actual_sources": actual_sources},
        )
        
    set_actual = set(actual_sources or [])
    set_expected = set(expected_sources)
    
    overlap = set_actual.intersection(set_expected)
    recall = len(overlap) / len(set_expected) if set_expected else 1.0
    precision = len(overlap) / len(set_actual) if set_actual else 0.0
    
    score = round(recall, 4)
    passed = len(overlap) == len(set_expected)
    return EvaluationResult(
        evaluator_name="source_validation",
        score=score,
        passed=passed,
        explanation=f"Source recall: {recall:.2f}, precision: {precision:.2f}. Expected: {list(set_expected)}, got: {list(set_actual)}.",
        details={"expected": list(set_expected), "actual": list(set_actual), "overlap": list(overlap)},
    )

def abstention_evaluator(
    actual_answer: str,
    should_abstain: bool,
    abstention_indicators: Optional[List[str]] = None,
) -> EvaluationResult:
    """
    Evaluate whether the assistant appropriately abstains when requested information is absent.
    """
    indicators = abstention_indicators or [
        "not available in the provided",
        "could not find this information",
        "does not contain",
        "not mentioned",
        "cannot find",
        "no information",
        "unsupported",
    ]
    
    norm_answer = actual_answer.lower()
    abstained = any(ind in norm_answer for ind in indicators)
    
    if should_abstain:
        passed = abstained
        score = 1.0 if passed else 0.0
        exp = "Assistant correctly abstained from answering." if passed else "Assistant failed to abstain when information was unavailable."
    else:
        # Question was supposed to be answerable
        passed = not abstained
        score = 1.0 if passed else 0.0
        exp = "Assistant provided an answer as expected." if passed else "Assistant improperly abstained on an answerable question."
        
    return EvaluationResult(
        evaluator_name="abstention_evaluation",
        score=score,
        passed=passed,
        explanation=exp,
        details={"should_abstain": should_abstain, "detected_abstention": abstained},
    )

def code_based_context_precision(
    retrieved_sources: List[str],
    expected_sources: List[str],
) -> EvaluationResult:
    """
    Deterministic document-level context precision and ranked precision.
    """
    if not expected_sources or not retrieved_sources:
        return EvaluationResult(
            evaluator_name="code_context_precision",
            score=0.0,
            passed=False,
            explanation="Empty retrieved sources or expected sources.",
            details={},
        )
        
    exp_set = set(expected_sources)
    relevant_flags = [1 if s in exp_set else 0 for s in retrieved_sources]
    
    # Precision at K
    total_relevant = sum(relevant_flags)
    precision = total_relevant / len(retrieved_sources)
    
    # Average precision (ranking aware)
    running_relevant = 0
    precisions_at_k = []
    for k, flag in enumerate(relevant_flags, start=1):
        if flag == 1:
            running_relevant += 1
            precisions_at_k.append(running_relevant / k)
            
    map_score = sum(precisions_at_k) / total_relevant if total_relevant > 0 else 0.0
    
    return EvaluationResult(
        evaluator_name="code_context_precision",
        score=round(map_score, 4),
        passed=map_score > 0.5,
        explanation=f"Mean Average Precision of retrieved document sources is {map_score:.4f} (P@{len(retrieved_sources)}: {precision:.4f}).",
        details={
            "retrieved_sources": retrieved_sources,
            "expected_sources": expected_sources,
            "relevant_flags": relevant_flags,
            "precision_at_k": round(precision, 4),
            "map_score": round(map_score, 4),
        },
    )

def code_based_context_recall(
    retrieved_sources: List[str],
    expected_sources: List[str],
) -> EvaluationResult:
    """
    Deterministic source-level context recall.
    """
    if not expected_sources:
        return EvaluationResult(
            evaluator_name="code_context_recall",
            score=None,
            passed=None,
            explanation="No expected sources defined.",
            details={},
        )
        
    ret_set = set(retrieved_sources or [])
    exp_set = set(expected_sources)
    overlap = ret_set.intersection(exp_set)
    recall = len(overlap) / len(exp_set)
    
    score = round(recall, 4)
    passed = score >= 1.0
    return EvaluationResult(
        evaluator_name="code_context_recall",
        score=score,
        passed=passed,
        explanation=f"Retrieved {len(overlap)} of {len(exp_set)} expected document sources ({score * 100:.1f}%).",
        details={"retrieved": list(ret_set), "expected": list(exp_set), "overlap": list(overlap)},
    )
