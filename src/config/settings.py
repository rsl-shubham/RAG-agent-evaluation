import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EVALUATION_DIR = DATA_DIR / "evaluation"
REPORTS_DIR = BASE_DIR / "reports"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

class Settings(BaseModel):
    # Google GenAI Models
    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"))
    judge_model: str = Field(default_factory=lambda: os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"))
    
    # LangSmith Configuration
    langsmith_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY"))
    langsmith_tracing: bool = Field(default_factory=lambda: os.getenv("LANGSMITH_TRACING", "true").lower() in ("true", "1", "yes"))
    langsmith_project: str = Field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "acme-rag-development"))
    langsmith_endpoint: str = Field(default_factory=lambda: os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))

    # RAG Index & Retrieval Defaults
    collection_name: str = "company_knowledge"
    chroma_db_dir: Path = CHROMA_DB_DIR
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50
    default_retrieval_k: int = 3
    
    # App Metadata
    app_version: str = "1.0.0"
    prompt_version: str = "v1"

settings = Settings()
