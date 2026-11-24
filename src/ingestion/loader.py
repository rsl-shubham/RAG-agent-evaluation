import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import DOCUMENTS_DIR

def parse_document_header(content: str) -> Dict[str, Any]:
    """Extract metadata header comments if present."""
    metadata = {}
    lines = content.splitlines()
    for line in lines[:5]:
        if line.startswith("# Metadata:") or line.startswith("# metadata:"):
            raw_meta = line.split(":", 1)[1].strip()
            pairs = [p.strip() for p in raw_meta.split(",") if "=" in p]
            for pair in pairs:
                k, v = pair.split("=", 1)
                metadata[k.strip()] = v.strip()
    return metadata

def load_documents(directory: Optional[Path] = None) -> List[Document]:
    """Load text and markdown documents from directory with rich metadata and stable doc IDs."""
    doc_dir = directory or DOCUMENTS_DIR
    documents = []
    
    for file_path in sorted(doc_dir.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        parsed_meta = parse_document_header(content)
        
        # Compute stable doc hash based on relative path and content
        doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        
        metadata = {
            "source": file_path.name,
            "category": parsed_meta.get("category", "General"),
            "version": parsed_meta.get("version", "1.0"),
            "department": parsed_meta.get("department", "Company"),
            "effective_date": parsed_meta.get("effective_date", "2026-01-01"),
            "doc_id": f"doc_{file_path.stem}_{doc_hash}",
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
        
    return documents

def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split documents into chunks with stable, deterministic chunk IDs."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks):
            chunk_content = chunk.page_content.strip()
            chunk_hash = hashlib.sha256(chunk_content.encode("utf-8")).hexdigest()[:8]
            chunk_id = f"{doc.metadata.get('source', 'doc')}_c{idx:03d}_{chunk_hash}"
            
            chunk.metadata = {
                **doc.metadata,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "chunk_size": len(chunk_content),
                "total_chunks": len(doc_chunks),
            }
            chunks.append(chunk)
            
    return chunks
