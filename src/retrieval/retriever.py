from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from src.retrieval.vector_store import get_vector_store

class RetrievedChunk(BaseModel):
    content: str
    source: str
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    category: Optional[str] = None
    version: Optional[str] = None
    department: Optional[str] = None
    score: float = Field(description="Distance or similarity score")

class RetrievalResult(BaseModel):
    query: str
    k: int
    chunks: List[RetrievedChunk]
    sources: List[str]
    document_versions: Dict[str, str]

def retrieve_documents(
    query: str,
    k: int = 3,
    collection_name: Optional[str] = None,
    filter_metadata: Optional[Dict[str, Any]] = None,
    score_threshold: Optional[float] = None,
) -> RetrievalResult:
    """
    Retrieve documents matching query with full metadata, scores, and document versions.
    """
    store = get_vector_store(collection_name=collection_name)
    
    if filter_metadata:
        raw_results = store.similarity_search_with_score(query, k=k, filter=filter_metadata)
    else:
        raw_results = store.similarity_search_with_score(query, k=k)
        
    chunks = []
    sources = set()
    doc_versions = {}
    
    for doc, distance in raw_results:
        if score_threshold is not None and distance > score_threshold:
            continue
            
        src = doc.metadata.get("source", "Unknown")
        ver = doc.metadata.get("version", "1.0")
        sources.add(src)
        doc_versions[src] = ver
        
        chunks.append(
            RetrievedChunk(
                content=doc.page_content,
                source=src,
                chunk_id=doc.metadata.get("chunk_id"),
                chunk_index=doc.metadata.get("chunk_index"),
                category=doc.metadata.get("category"),
                version=ver,
                department=doc.metadata.get("department"),
                score=float(distance),
            )
        )
        
    return RetrievalResult(
        query=query,
        k=k,
        chunks=chunks,
        sources=sorted(list(sources)),
        document_versions=doc_versions,
    )
