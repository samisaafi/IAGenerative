from dataclasses import dataclass

@dataclass
class RAGChatbotConfig:
    # LM Studio
    LM_STUDIO_API_BASE: str = "http://localhost:1234/v1"
    LM_STUDIO_API_KEY: str = "lm-studio"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector DB
    PERSIST_DIRECTORY: str = "./chroma_db"

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retrieval
    TOP_K_RESULTS: int = 3
