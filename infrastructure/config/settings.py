import os
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Validates and manages the runtime environment configuration parameters
    for the vector engine. Automatically parses system variables and local .env files.
    """

    # Core Application Configuration
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"

    # Nomic AI Platform / Local Model Settings
    NOMIC_API_KEY: Optional[str] = None
    NOMIC_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIMENSION: int = 768

    # Compute Hardware Configuration
    # Defaults to "cuda" if hardware supports it, allowing runtime override via ENV
    COMPUTE_DEVICE: str = "cuda"

    # Nomic Task Prefixes (Crucial for local retrieval accuracy)
    DOC_PREFIX: str = "search_document: "
    QUERY_PREFIX: str = "search_query: "

    # LanceDB Vector Database Settings
    # Defaults to a local directory named 'data/academic_lancedb' inside the project root
    LANCE_DB_URI: str = str(Path(__file__).resolve().parents[2] / "data" / "academic_lancedb")

    # Pydantic v2 configuration settings block
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Gracefully ignore extra env variables not defined here
    )

    def model_post_init(self, __context) -> None:
        """Dynamically check for CUDA availability if configured to use it."""
        if self.COMPUTE_DEVICE == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    # Fallback gracefully to CPU in environments without a GPU
                    self.COMPUTE_DEVICE = "cpu"
            except ImportError:
                self.COMPUTE_DEVICE = "cpu"


# Instantiate a singleton configuration instance to be shared across the microservice
settings = Settings()