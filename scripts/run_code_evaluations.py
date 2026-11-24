import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.rag.pipeline import answer_question
from src.evaluators.code_based.metrics import (
    exact_match_evaluator,
    normalized_exact_match_evaluator,
    string_similarity_evaluator,
    keyword_coverage_evaluator,
    numerical_fact_evaluator,
    source_validation_evaluator,
    abstention_evaluator,
)
from src.evaluators.datasets import load_dataset
from src.monitoring.metrics import aggregate_evaluations

def main():
    print("=" * 70)
    print(" Running Code-Based Deterministic Evaluations ")
    print("=" * 70)
    
    # Run against answer_quality dataset
    dataset = load_dataset("answer_quality")
    results = []
    
    for case in dataset:
        cid = case["id"]
        q = case["inputs"]["question"]
        ref = case.get("reference", "")
        exp_sources = case.get("expected_sources", [])
        exp_numbers = case.get("expected_numbers", [])
        
        print(f"\n[Case {cid}]: {q}")
        rag_res = answer_question(q)
        actual = rag_res["answer"]
        actual_sources = rag_res["sources"]
        
        # Evaluate metrics
        em = exact_match_evaluator(actual, ref)
        nem = normalized_exact_match_evaluator(actual, ref)
        ss = string_similarity_evaluator(actual, ref)
        num = numerical_fact_evaluator(actual, exp_numbers)
        src = source_validation_evaluator(actual_sources, exp_sources)
        abst = abstention_evaluator(actual, should_abstain=False)
        
        case_eval = {
            "id": cid,
            "question": q,
            "actual_answer": actual,
            "reference": ref,
            "evaluations": {
                "exact_match": em.model_dump(),
                "normalized_exact_match": nem.model_dump(),
                "string_similarity": ss.model_dump(),
                "numerical_validation": num.model_dump(),
                "source_validation": src.model_dump(),
                "abstention": abst.model_dump(),
            }
        }
        results.append(case_eval)
        print(f"  String Sim: {ss.score:.3f} | Numerical: {num.score} | Source Recall: {src.score}")
        
    summary = aggregate_evaluations(results)
    print("\n" + "=" * 70)
    print(" EVALUATION SUMMARY ")
    print("=" * 70)
    for m, stat in summary.get("metrics_summary", {}).items():
        print(f"{m:<25}: Mean Score={stat['mean_score']} | Pass Rate={stat['pass_rate']}")
        
    out_file = Path("reports/code_evaluation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved detailed code evaluation results to {out_file}")

if __name__ == "__main__":
    main()
