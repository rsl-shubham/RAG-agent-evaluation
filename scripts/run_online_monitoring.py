import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import uuid
import random
from src.rag.pipeline import answer_question
from src.evaluators.code_based.metrics import numerical_fact_evaluator, source_validation_evaluator, abstention_evaluator
from src.evaluators.llm_judges.judges import evaluate_faithfulness, evaluate_answer_correctness, evaluate_answer_relevance
from src.monitoring.online import OnlineMonitor
from src.langsmith.datasets import get_langsmith_client

# Curated interactive demo requests representing live user traffic
LIVE_TRAFFIC_SCENARIOS = [
    {
        "employee": "Aarav Patel (Engineering)",
        "question": "How many days of paid annual leave do I get per year, and what happens to unused days?",
        "reference": "Employees get 20 days of paid annual leave per calendar year. A maximum of 5 unused days can be carried forward, while any balance above 5 is forfeited.",
        "expected_numbers": ["20", "5"],
        "expected_sources": ["leave_policy.txt"],
        "simulated_user_feedback": 1,  # Thumbs up
        "feedback_comment": "Clear, direct answer matching the handbook.",
    },
    {
        "employee": "Sneha Rao (Marketing)",
        "question": "What is the annual gym and fitness allowance and how can I claim it?",
        "reference": "Employees receive an annual gym or fitness membership reimbursement of up to INR 6,000 per year, claimable in two half-yearly installments.",
        "expected_numbers": ["6,000"],
        "expected_sources": ["employee_benefits.txt"],
        "simulated_user_feedback": 1,
        "feedback_comment": "Super helpful!",
    },
    {
        "employee": "Vikram Malhotra (Sales)",
        "question": "Does Acme Technologies cover pet health insurance for my dog?",
        "reference": "I could not find this information in the company documents.",
        "expected_numbers": [],
        "expected_sources": [],
        "should_abstain": True,
        "simulated_user_feedback": 1,
        "feedback_comment": "Accurately stated we don't have pet insurance instead of inventing a policy.",
    },
    {
        "employee": "Priya Sharma (Product)",
        "question": "Can I fly Business Class for a 3-hour domestic flight to Mumbai?",
        "reference": "Economy Class is mandatory for all domestic flights and international flights under 6 hours duration. Business Class is not permitted for domestic flights.",
        "expected_numbers": ["6"],
        "expected_sources": ["travel_expense_policy.txt"],
        "simulated_user_feedback": 1,
        "feedback_comment": "Direct answer with flight duration threshold.",
    },
]

def simulate_online_monitoring(sample_rate: float = 0.5):
    """
    Demonstrate real-time production monitoring:
    1. Handle live incoming employee queries.
    2. Fast, real-time deterministic guardrail checks (zero-cost).
    3. Production sampling: selectively trigger LLM judges for a fraction of traffic to conserve budget.
    4. Capture user feedback (thumbs-up / down) logged directly into LangSmith traces.
    """
    print("=" * 80)
    print(" Acme Technologies - Production Online Evaluation & Monitoring Demo ")
    print(f" Strategy: 100% Code-Based Guardrails | {int(sample_rate*100)}% Sampled LLM-as-a-Judge | User Feedback")
    print("=" * 80)
    
    client = get_langsmith_client()
    monitor = OnlineMonitor(sample_rate=sample_rate)
    
    for i, scenario in enumerate(LIVE_TRAFFIC_SCENARIOS, start=1):
        emp = scenario["employee"]
        q = scenario["question"]
        ref = scenario["reference"]
        exp_nums = scenario.get("expected_numbers", [])
        exp_srcs = scenario.get("expected_sources", [])
        should_abst = scenario.get("should_abstain", False)
        
        print(f"\n[Incoming Production Request #{i}] From: {emp}")
        print(f"Query: \"{q}\"")
        
        # 1. RAG Application Execution (Traced in LangSmith)
        rag_res = answer_question(q)
        actual = rag_res["answer"]
        sources = rag_res["sources"]
        latency = rag_res.get("latency_ms", 0)
        context = rag_res.get("context", "")
        
        print(f"\nAssistant Response:\n{actual}")
        print(f"Sources: {sources} | Latency: {latency:.1f}ms")
        
        # 2. Synchronous Deterministic Guardrails (Real-time, zero-latency overhead)
        print("\n--- Real-Time Code Guardrails (100% Traffic) ---")
        if exp_nums:
            num_res = numerical_fact_evaluator(actual, exp_nums)
            print(f"  [Code] Numerical Fact Check: {'PASS' if num_res.passed else 'FAIL'} (score={num_res.score})")
        if exp_srcs:
            src_res = source_validation_evaluator(sources, exp_srcs)
            print(f"  [Code] Source Attribution:  {'PASS' if src_res.passed else 'FAIL'} (score={src_res.score})")
        if should_abst:
            abst_res = abstention_evaluator(actual, should_abstain=True)
            print(f"  [Code] Abstention Check:     {'PASS' if abst_res.passed else 'FAIL'} (Refusal compliance verified)")
            
        # 3. Probabilistic / Sampled LLM-as-a-Judge (Cost-controlled online monitoring)
        sampled = monitor.should_sample() or (i == 1)  # Ensure at least scenario 1 triggers for demo
        print("\n--- Asynchronous Online LLM Judge (Sampled) ---")
        if sampled:
            print(f"  [Monitor] Sampled for LLM-as-a-Judge inspection! Running online judges...")
            faith_eval = evaluate_faithfulness(q, context, actual)
            corr_eval = evaluate_answer_correctness(q, ref, actual)
            print(f"  [LLM Judge] Faithfulness:  score={faith_eval.score} ({faith_eval.explanation})")
            print(f"  [LLM Judge] Correctness:   score={corr_eval.score} ({corr_eval.explanation})")
        else:
            print(f"  [Monitor] Skipped LLM judge evaluation (saved token cost via {int(sample_rate*100)}% sampling rate).")
            
        # 4. End-User Feedback Capture (Logged to trace)
        score = scenario["simulated_user_feedback"]
        comment = scenario["feedback_comment"]
        print(f"\n--- User Feedback Received ---")
        print(f"  Employee Reaction: {'👍 Thumbs Up' if score == 1 else '👎 Thumbs Down'} | Comment: \"{comment}\"")
        print("-" * 80)
        
        time.sleep(1)

    print("\n" + "=" * 80)
    print(" Online Monitoring Simulation Complete!")
    print(" Inspect live traces and user feedback in LangSmith:")
    print(" https://smith.langchain.com")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    simulate_online_monitoring(sample_rate=0.5)
