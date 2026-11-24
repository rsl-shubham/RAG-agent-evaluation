import json
from pathlib import Path
from typing import List, Dict, Any
from src.config.settings import EVALUATION_DIR

DATASET_FILES = {
    "retrieval": EVALUATION_DIR / "retrieval" / "retrieval_dataset.json",
    "answer_quality": EVALUATION_DIR / "answer_quality" / "answer_quality_dataset.json",
    "unanswerable": EVALUATION_DIR / "unanswerable" / "unanswerable_dataset.json",
    "faithfulness": EVALUATION_DIR / "adversarial" / "faithfulness_dataset.json",
    "adversarial": EVALUATION_DIR / "adversarial" / "adversarial_dataset.json",
    "regression": EVALUATION_DIR / "regression" / "regression_dataset.json",
}

def load_dataset(category: str) -> List[Dict[str, Any]]:
    """Load a specific dataset category."""
    if category not in DATASET_FILES:
        raise ValueError(f"Unknown dataset category: {category}. Choose from {list(DATASET_FILES.keys())}")
    file_path = DATASET_FILES[category]
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_all_datasets() -> Dict[str, List[Dict[str, Any]]]:
    """Load all evaluation datasets into a dictionary."""
    return {cat: load_dataset(cat) for cat in DATASET_FILES}

def get_total_test_cases() -> int:
    """Return total count of all curated test cases."""
    all_data = load_all_datasets()
    return sum(len(items) for items in all_data.values())
