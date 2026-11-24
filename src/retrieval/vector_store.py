import time
from typing import List, Optional, Tuple
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config.settings import settings
from src.config.models import get_embeddings
from src.ingestion.loader import load_documents, split_documents

def get_vector_store(
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    embeddings_model=None,
) -> Chroma:
    """Instantiate a Chroma vector store connection."""
    persist_dir = str(persist_directory or settings.chroma_db_dir)
    coll_name = collection_name or settings.collection_name
    emb = embeddings_model or get_embeddings()
    
    return Chroma(
        collection_name=coll_name,
        embedding_function=emb,
        persist_directory=persist_dir,
    )

def index_documents(
    documents: Optional[List[Document]] = None,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embeddings_model=None,
    force_reindex: bool = False,
    batch_size: int = 20,
) -> Tuple[Chroma, int]:
    """
    Ingest documents into ChromaDB with deterministic IDs to avoid duplicate records.
    Batches additions to avoid hitting rate limits.
    """
    persist_dir = str(persist_directory or settings.chroma_db_dir)
    coll_name = collection_name or settings.collection_name
    emb = embeddings_model or get_embeddings()
    
    if force_reindex and Path(persist_dir).exists():
        try:
            temp_store = Chroma(collection_name=coll_name, embedding_function=emb, persist_directory=persist_dir)
            temp_store.delete_collection()
        except Exception:
            pass

    vector_store = Chroma(
        collection_name=coll_name,
        embedding_function=emb,
        persist_directory=persist_dir,
    )
    
    raw_docs = documents if documents is not None else load_documents()
    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    
    existing_ids = set()
    try:
        existing_data = vector_store.get(ids=ids)
        if existing_data and "ids" in existing_data:
            existing_ids = set(existing_data["ids"])
    except Exception:
        existing_ids = set()
        
    new_chunks = []
    new_ids = []
    for chunk, cid in zip(chunks, ids):
        if cid not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(cid)
            
    if new_chunks:
        # Batch addition with pause to stay well within Gemini RPM quota
        for i in range(0, len(new_chunks), batch_size):
            batch_c = new_chunks[i:i + batch_size]
            batch_i = new_ids[i:i + batch_size]
            vector_store.add_documents(documents=batch_c, ids=batch_i)
            if i + batch_size < len(new_chunks):
                time.sleep(2)
        print(f"Indexed {len(new_chunks)} new chunks (skipped {len(chunks) - len(new_chunks)} existing) into '{coll_name}'.")
    else:
        print(f"All {len(chunks)} chunks already indexed in '{coll_name}'. No new records added.")
        
    return vector_store, len(chunks)
