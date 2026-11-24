import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.evaluators.code_based.metrics import (
    exact_match_evaluator,
    normalized_exact_match_evaluator,
    string_similarity_evaluator,
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

def run_deep_dive():
    print("=" * 80)
    print(" LIVE DEMO: Code-Based Evaluators vs. LLM-as-a-Judge Deep Dive ")
    print("=" * 80)

    # -------------------------------------------------------------
    # CASE 1: Why Exact Match Fails & Why Numerical + LLM Judge Wins
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" SCENARIO 1: Semantic Equivalence vs Exact Match Failure")
    print("#" * 80)
    q1 = "How many days of paid annual leave do employees receive per year?"
    ref1 = "Employees receive 20 days of paid annual leave per calendar year."
    gen1 = "Full-time team members are entitled to twenty (20) days of paid annual vacation each year."

    print(f"Question:        {q1}")
    print(f"Reference:       {ref1}")
    print(f"Generated Answer:{gen1}")
    print("\n--- Running Evaluators ---")

    em = exact_match_evaluator(gen1, ref1)
    nem = normalized_exact_match_evaluator(gen1, ref1)
    ss = string_similarity_evaluator(gen1, ref1)
    num = numerical_fact_evaluator(gen1, ["20"])
    rel = evaluate_answer_relevance(q1, gen1)
    corr = evaluate_answer_correctness(q1, ref1, gen1)

    print(f"1. Exact Match (Raw):            score={em.score}   -> {'FAIL' if not em.passed else 'PASS'} (Reason: {em.explanation})")
    print(f"2. Normalized Exact Match:       score={nem.score}   -> {'FAIL' if not nem.passed else 'PASS'} (Reason: {nem.explanation})")
    print(f"3. Token String Jaccard:         score={ss.score:.3f} -> {'PASS' if ss.passed else 'POOR OVERLAP'} (Reason: {ss.explanation})")
    print(f"4. Deterministic Numerical Fact: score={num.score}   -> {'PASS' if num.passed else 'FAIL'} (Found '20')")
    print(f"5. LLM Judge - Answer Relevance: score={rel.score}   -> {'PASS' if rel.passed else 'FAIL'} (Verdict: {rel.details.get('verdict')})")
    print(f"6. LLM Judge - Correctness:      score={corr.score}  -> {'PASS' if corr.passed else 'FAIL'} (Semantic alignment verified)")
    print("\n💡 KEY TAKEAWAY FOR AUDIENCE:")
    print("   Deterministic exact match assigns 0.0 despite the answer being 100% correct.")
    print("   Pairing Regex Numerical Fact Validation with LLM-as-a-Judge Correctness provides the true quality signal.")

    # -------------------------------------------------------------
    # CASE 2: Faithfulness & Hallucination Claim-Level Breakdown
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" SCENARIO 2: Hallucination Detection & Claim-Level Faithfulness")
    print("#" * 80)
    q2 = "What is the internet reimbursement policy?"
    ctx2 = "Remote and hybrid employees receive an Internet Reimbursement allowance of up to INR 1,500 per month for home broadband connectivity. Invoices must be submitted between the 1st and 10th."
    gen2_faithful = "Remote employees can claim up to INR 1,500 per month for internet."
    gen2_hallucinated = "Remote employees get INR 1,500 per month for internet, plus INR 2,500 monthly for free Starbucks coffee."

    print(f"Context: {ctx2}")
    print(f"\nEvaluating Hallucinated Answer: \"{gen2_hallucinated}\"")
    
    faith_res = evaluate_faithfulness(q2, ctx2, gen2_hallucinated)
    print(f"\nFaithfulness Score: {faith_res.score} -> {'PASS' if faith_res.passed else 'FLAGGED AS UNFAITHFUL'}")
    print(f"Overall Reason: {faith_res.explanation}")
    print("\nClaim-Level Breakdown (Extracted by LLM Judge):")
    for idx, c in enumerate(faith_res.details.get("claims", []), start=1):
        verdict = c["verdict"]
        print(f"  Claim {idx}: \"{c['claim']}\"")
        print(f"     Verdict:  [{verdict.upper()}]")
        print(f"     Evidence: \"{c['evidence']}\"")
        print(f"     Reason:   {c['reason']}")

    # -------------------------------------------------------------
    # CASE 3: Chunk-Level Context Precision (MAP) & Context Recall
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(" SCENARIO 3: Retrieval Evaluation (Context Precision MAP & Context Recall)")
    print("#" * 80)
    q3 = "What are the rules and carry-forward limits for annual leave?"
    ref3 = "Employees receive 20 days of paid annual leave. A maximum of 5 unused days can be carried forward, and any excess above 5 days is forfeited on December 31st."
    
    mock_retrieved_chunks = [
        {"content": "Full-time employees receive 20 days of paid annual leave per calendar year.", "source": "leave_policy.txt"},
        {"content": "A maximum of 5 unused annual leave days can be carried forward to the following calendar year. Excess is forfeited.", "source": "leave_policy.txt"},
        {"content": "Standard hardware refresh cycle is 36 months for Apple MacBook Pro laptops.", "source": "it_support_policy.txt"},
    ]
    
    prec_res = evaluate_context_precision(q3, mock_retrieved_chunks)
    rec_res = evaluate_context_recall(q3, ref3, "\n\n".join(c["content"] for c in mock_retrieved_chunks))

    print(f"Retrieval Context Precision MAP: score={prec_res.score} -> {prec_res.explanation}")
    print(f"Precision Details: P@3={prec_res.details.get('precision_at_k')} | Relevance Flags={prec_res.details.get('relevance_flags')}")
    print(f"\nRetrieval Context Recall:       score={rec_res.score} -> {rec_res.explanation}")
    print("Atomic Facts Checked by Judge:")
    for f in rec_res.details.get("atomic_facts", []):
        status = "SUPPORTED" if f["is_supported"] else "MISSING"
        print(f"  * [{status}] \"{f['atomic_fact']}\"")

    print("\n" + "=" * 80)
    print(" Deep Dive Demo Execution Complete!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_deep_dive()
