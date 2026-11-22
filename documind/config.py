from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    # Project Information
    PROJECT_NAME: str = "DocuMind"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # OpenAI & LLM Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # File and Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    BM25_INDEX_PATH: Path = DATA_DIR / "bm25_index.pkl"

    # Hybrid Retrieval Tuning
    # hybrid_alpha: 1.0 = pure dense vector, 0.0 = pure BM25, 0.5 = balanced blend
    HYBRID_ALPHA: float = 0.5
    RETRIEVAL_TOP_K: int = 10
    MMR_DIVERSITY_LAMBDA: float = 0.7
    MMR_FINAL_K: int = 5

    # Document Chunking
    CHUNK_SIZE_CHARS: int = 1200
    CHUNK_OVERLAP_CHARS: int = 200

    # API Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        """Create necessary runtime directories if they do not exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
