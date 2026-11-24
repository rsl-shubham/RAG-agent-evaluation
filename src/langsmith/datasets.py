import os
from typing import Dict, Any, List, Optional
from langsmith import Client
from src.config.settings import settings
from src.evaluators.datasets import load_all_datasets

DATASET_NAME_PREFIX = "acme"

def get_langsmith_client() -> Client:
    """Return an authenticated LangSmith client."""
    return Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key,
    )

def upload_dataset_to_langsmith(
    dataset_category: str,
    items: List[Dict[str, Any]],
    client: Optional[Client] = None,
    overwrite: bool = False,
) -> str:
    """
    Upload a local dataset category to LangSmith idempotently.
    Avoids duplicate dataset creation.
    """
    client = client or get_langsmith_client()
    dataset_name = f"{DATASET_NAME_PREFIX}-{dataset_category.replace('_', '-')}"
    
    # Check if dataset already exists
    existing = None
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        pass
        
    if existing:
        if not overwrite:
            print(f"Dataset '{dataset_name}' already exists in LangSmith (ID: {existing.id}). Skipping re-upload.")
            return str(existing.id)
        else:
            print(f"Dataset '{dataset_name}' exists. Deleting to overwrite...")
            client.delete_dataset(dataset_id=existing.id)
            
    # Create dataset
    description = f"Acme Technologies Evaluation Dataset for {dataset_category}"
    ds = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )
    
    # Format and upload examples
    inputs = []
    outputs = []
    metadata = []
    
    for item in items:
        # User query
        q = item.get("inputs", {}).get("question", "")
        inputs.append({"question": q})
        
        # Reference answer / expected facts
        out = {
            "reference": item.get("reference", ""),
            "expected_sources": item.get("expected_sources", []),
            "expected_facts": item.get("expected_facts", []),
            "should_abstain": item.get("should_abstain", False),
        }
        outputs.append(out)
        
        meta = {
            "id": item.get("id"),
            "tags": item.get("tags", []),
            "scenario": item.get("scenario", ""),
        }
        metadata.append(meta)
        
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_id=ds.id,
    )
    
    print(f"Successfully uploaded {len(inputs)} examples to LangSmith dataset '{dataset_name}' (ID: {ds.id})")
    return str(ds.id)

def upload_all_datasets(overwrite: bool = False) -> Dict[str, str]:
    """Upload all local evaluation datasets to LangSmith."""
    client = get_langsmith_client()
    all_data = load_all_datasets()
    results = {}
    for cat, items in all_data.items():
        ds_id = upload_dataset_to_langsmith(cat, items, client=client, overwrite=overwrite)
        results[cat] = ds_id
    return results
