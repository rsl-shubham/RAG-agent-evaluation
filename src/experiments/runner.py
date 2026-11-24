from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langsmith import evaluate
from src.config.settings import settings
from src.retrieval.vector_store import index_documents
from src.rag.pipeline import answer_question
from src.langsmith.evaluators import (
    langsmith_string_similarity,
    langsmith_source_validation,
    langsmith_llm_relevance,
    langsmith_llm_faithfulness,
    langsmith_llm_correctness,
)

class ExperimentConfig(BaseModel):
    name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    collection_name: str
    dataset_name: str = "acme-regression"
    description: str = ""

def run_rag_experiment(
    config: ExperimentConfig,
    evaluators: Optional[list] = None,
    max_concurrency: int = 2,
) -> Any:
    """
    Execute a LangSmith evaluation experiment with a specific RAG configuration.
    Ensures that the vector store is indexed to the specified chunk parameters before running.
    """
    print(f"\n{'=' * 60}")
    print(f" Running LangSmith Experiment: {config.name}")
    print(f" Config: chunk_size={config.chunk_size}, overlap={config.chunk_overlap}, k={config.retrieval_k}")
    print(f" Target Collection: {config.collection_name}")
    print(f"{'=' * 60}")
    
    # Re-index / index documents into the specific collection for this experiment
    index_documents(
        collection_name=config.collection_name,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        force_reindex=False,
    )
    
    # Define target RAG function consuming the dataset example input
    def target_rag_app(inputs: dict) -> dict:
        question = inputs["question"]
        res = answer_question(
            question=question,
            k=config.retrieval_k,
            collection_name=config.collection_name,
        )
        return res
        
    eval_list = evaluators or [
        langsmith_string_similarity,
        langsmith_source_validation,
        langsmith_llm_relevance,
        langsmith_llm_faithfulness,
        langsmith_llm_correctness,
    ]
    
    experiment_results = evaluate(
        target_rag_app,
        data=config.dataset_name,
        evaluators=eval_list,
        experiment_prefix=config.name,
        max_concurrency=max_concurrency,
        metadata={
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "retrieval_k": config.retrieval_k,
            "collection_name": config.collection_name,
            "app_version": settings.app_version,
            "llm_model": settings.llm_model,
        },
    )
    
    print(f"\nExperiment '{config.name}' completed!")
    return experiment_results
