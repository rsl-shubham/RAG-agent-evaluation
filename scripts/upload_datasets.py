import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.langsmith.datasets import upload_all_datasets, upload_dataset_to_langsmith
from src.evaluators.datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Upload evaluation datasets to LangSmith.")
    parser.add_argument("--category", type=str, default="all", help="Dataset category to upload or 'all'")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset in LangSmith")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" Uploading Acme Evaluation Datasets to LangSmith ")
    print("=" * 60)
    
    if args.category == "all":
        upload_all_datasets(overwrite=args.overwrite)
    else:
        items = load_dataset(args.category)
        upload_dataset_to_langsmith(args.category, items, overwrite=args.overwrite)
        
    print("\nUpload completed successfully!")

if __name__ == "__main__":
    main()
