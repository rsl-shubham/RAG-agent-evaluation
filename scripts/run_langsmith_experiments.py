import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.experiments.runner import run_rag_experiment, ExperimentConfig

EXPERIMENT_SUITE = [
    ExperimentConfig(
        name="exp-chunk300-k3",
        chunk_size=300,
        chunk_overlap=30,
        retrieval_k=3,
        collection_name="acme_c300_o30",
        dataset_name="acme-regression",
        description="Small chunk size (300 chars), top-3 retrieval",
    ),
    ExperimentConfig(
        name="exp-chunk500-k3",
        chunk_size=500,
        chunk_overlap=50,
        retrieval_k=3,
        collection_name="acme_c500_o50",
        dataset_name="acme-regression",
        description="Standard baseline chunk size (500 chars), top-3 retrieval",
    ),
    ExperimentConfig(
        name="exp-chunk500-k5",
        chunk_size=500,
        chunk_overlap=50,
        retrieval_k=5,
        collection_name="acme_c500_o50",
        dataset_name="acme-regression",
        description="Standard chunk size (500 chars), broader top-5 retrieval",
    ),
]

def main():
    parser = argparse.ArgumentParser(description="Run LangSmith RAG experiments.")
    parser.add_argument("--experiment", type=str, default="all", help="Specific experiment name or 'all'")
    parser.add_argument("--dataset", type=str, default="acme-regression", help="LangSmith dataset name to test against")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" Running LangSmith Production Experiments ")
    print("=" * 60)
    
    selected = [e for e in EXPERIMENT_SUITE if args.experiment == "all" or e.name == args.experiment]
    if not selected:
        print(f"Unknown experiment '{args.experiment}'. Available: {[e.name for e in EXPERIMENT_SUITE]}")
        return
        
    for cfg in selected:
        cfg.dataset_name = args.dataset
        run_rag_experiment(cfg)
        
    print("\nAll requested experiments executed successfully in LangSmith!")

if __name__ == "__main__":
    main()
