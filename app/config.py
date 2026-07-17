"""
Centralized configuration, loaded from .env via pydantic-settings.
Import `settings` anywhere you need a config value — never read
os.environ directly elsewhere in the app.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Embedding microservice (manually started, port varies)
    EMBEDDING_SERVICE_URL: str = "http://127.0.0.1:8001"

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "video-qa"
    PINECONE_DIMENSION: int = 384

    # OpenRouter
    OPENROUTER_API_KEY: str
    DEFAULT_AI_MODEL: str = "openai/gpt-4o-mini"

    # Whisper
    WHISPER_MODEL_SIZE: str = "small"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"

    # Chunking
    CHUNK_MAX_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 50

    # Retrieval
    RETRIEVAL_TOP_K: int = 5

    # Temp file storage (cleaned up after each upload)
    TEMP_DIR: str = "temp_uploads"

    # Storage for uploaded files
    UPLOAD_STORAGE_DIR: str = "storage/uploads"

    # SQLite Database URL
    DATABASE_URL: str = "sqlite:///./knowledge.db"


settings = Settings()