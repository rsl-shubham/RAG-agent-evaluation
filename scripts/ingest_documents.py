import argparse
from src.retrieval.vector_store import index_documents
from src.config.settings import settings

def main():
    parser = argparse.ArgumentParser(description="Ingest company knowledge base documents into ChromaDB.")
    parser.add_argument("--chunk-size", type=int, default=settings.default_chunk_size, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=settings.default_chunk_overlap, help="Chunk overlap in characters")
    parser.add_argument("--collection", type=str, default=settings.collection_name, help="Chroma collection name")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindexing by clearing existing collection")
    
    args = parser.parse_args()
    
    print(f"Starting document ingestion...")
    print(f"Collection: {args.collection}")
    print(f"Chunk size: {args.chunk_size}, Chunk overlap: {args.chunk_overlap}")
    print(f"Force reindex: {args.force_reindex}")
    
    store, total = index_documents(
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_reindex=args.force_reindex,
    )
    
    print(f"Ingestion finished successfully! Total chunks managed: {total}")

if __name__ == "__main__":
    main()
