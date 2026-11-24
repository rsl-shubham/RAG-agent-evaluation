# Re-exporting for backward compatibility with existing scripts
from src.config.models import llm, embeddings
from src.config.settings import settings

__all__ = ["llm", "embeddings", "settings"]
