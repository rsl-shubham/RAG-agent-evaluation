from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.config.settings import settings

def get_llm(model: str = None, temperature: float = 0.0):
    return ChatGoogleGenerativeAI(
        model=model or settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )

def get_judge_llm(model: str = None, temperature: float = 0.0):
    return ChatGoogleGenerativeAI(
        model=model or settings.judge_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )

def get_embeddings(model: str = None):
    return GoogleGenerativeAIEmbeddings(
        model=model or settings.embedding_model,
        google_api_key=settings.google_api_key,
    )

# Backward-compatibility singletons
llm = get_llm()
embeddings = get_embeddings()
