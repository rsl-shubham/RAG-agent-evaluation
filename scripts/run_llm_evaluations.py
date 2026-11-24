import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.rag.pipeline import answer_question
from src.evaluators.llm_judges.judges import (
    evaluate_answer_relevance,
    evaluate_faithfulness,
    evaluate_answer_correctness,
    evaluate_context_precision,
    evaluate_context_recall,
)
from src.evaluators.datasets import load_dataset
from src.monitoring.metrics import aggregate_evaluations

def main():
    print("=" * 70)
    print(" Running LLM-as-a-Judge Evaluation Suite ")
    print("=" * 70)
    
    dataset = load_dataset("answer_quality")[:4]  # Run on representative test sample
    results = []
    
    for case in dataset:
        cid = case["id"]
        q = case["inputs"]["question"]
        ref = case.get("reference", "")
        
        print(f"\n[Evaluating {cid}]: {q}")
        rag_res = answer_question(q)
        actual = rag_res["answer"]
        context = rag_res["context"]
        chunks = rag_res["retrieved_documents"]
        
        rel_eval = evaluate_answer_relevance(q, actual)
        faith_eval = evaluate_faithfulness(q, context, actual)
        corr_eval = evaluate_answer_correctness(q, ref, actual)
        prec_eval = evaluate_context_precision(q, chunks)
        rec_eval = evaluate_context_recall(q, ref, context)
        
        print(f"  * Relevance:   {rel_eval.score} ({rel_eval.explanation})")
        print(f"  * Faithfulness:{faith_eval.score} ({faith_eval.explanation})")
        print(f"  * Correctness: {corr_eval.score} ({corr_eval.explanation})")
        print(f"  * Context Prec:{prec_eval.score} ({prec_eval.explanation})")
        print(f"  * Context Rec: {rec_eval.score} ({rec_eval.explanation})")
        
        results.append({
            "id": cid,
            "question": q,
            "actual_answer": actual,
            "reference": ref,
            "evaluations": {
                "llm_relevance": rel_eval.model_dump(),
                "llm_faithfulness": faith_eval.model_dump(),
                "llm_correctness": corr_eval.model_dump(),
                "llm_context_precision": prec_eval.model_dump(),
                "llm_context_recall": rec_eval.model_dump(),
            }
        })
        
    summary = aggregate_evaluations(results)
    out_file = Path("reports/llm_evaluation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"LLM-as-a-Judge evaluations finished! Results saved to {out_file}")

if __name__ == "__main__":
    main()
