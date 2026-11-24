import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from typing import Optional
from src.langsmith.datasets import get_langsmith_client

def compare_experiments(
    experiment_names: list,
    client: Optional[any] = None,
):
    """Fetch and print a comparison table of experiment runs from LangSmith."""
    client = client or get_langsmith_client()
    
    print("\n" + "=" * 80)
    print(" LangSmith Experiment Comparison Summary")
    print("=" * 80)
    
    header = f"{'Experiment Name':<28} | {'Latency (ms)':<14} | {'String Sim':<12} | {'Faithfulness':<14} | {'Correctness':<12}"
    print(header)
    print("-" * 80)
    
    for name in experiment_names:
        try:
            # Query runs from LangSmith project
            runs = list(client.list_runs(project_name=name, execution_order=1))
            total_runs = len(runs)
            if not runs:
                print(f"{name:<28} | {'No runs found or still processing':<45}")
                continue
                
            latencies = [r.latency for r in runs if r.latency is not None]
            avg_latency = (sum(latencies) / len(latencies) * 1000) if latencies else 0.0
            
            # Fetch feedback / metrics associated with this experiment's runs
            feedbacks = list(client.list_feedback(run_ids=[r.id for r in runs]))
            
            sim_scores = [f.score for f in feedbacks if f.key == "string_similarity" and f.score is not None]
            faith_scores = [f.score for f in feedbacks if f.key == "llm_faithfulness" and f.score is not None]
            corr_scores = [f.score for f in feedbacks if f.key == "llm_correctness" and f.score is not None]
            
            avg_sim = f"{sum(sim_scores)/len(sim_scores):.3f}" if sim_scores else "N/A"
            avg_faith = f"{sum(faith_scores)/len(faith_scores):.3f}" if faith_scores else "N/A"
            avg_corr = f"{sum(corr_scores)/len(corr_scores):.3f}" if corr_scores else "N/A"
            
            row = f"{name:<28} | {avg_latency:<14.1f} | {avg_sim:<12} | {avg_faith:<14} | {avg_corr:<12}"
            print(row)
        except Exception as e:
            print(f"{name:<28} | Error fetching comparison: {e}")
            
    print("=" * 80)
    print("Detailed visual traces, latency distributions, and feedback: https://smith.langchain.com\n")

def main():
    parser = argparse.ArgumentParser(description="Compare multiple LangSmith experiments.")
    parser.add_argument("experiments", nargs="*", default=["exp-chunk300-k3", "exp-chunk500-k3", "exp-chunk500-k5"])
    args = parser.parse_args()
    
    compare_experiments(args.experiments)

if __name__ == "__main__":
    main()
