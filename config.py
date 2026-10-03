"""Central, environment-based configuration for the application."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    knowledge_base_dir: Path = Path("knowledge_base")
    index_dir: Path = Path("data/vector_store")
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    retrieval_top_k: int = 5
    retrieval_threshold: float = 0.42
    chunk_size: int = 900
    chunk_overlap: int = 140
    min_chunk_length: int = 60
    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        settings = cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model).strip(),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            llm_api_url=os.getenv("LLM_API_URL", "").strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
            retrieval_threshold=float(os.getenv("RETRIEVAL_THRESHOLD", "0.42")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "140")),
            min_chunk_length=int(os.getenv("MIN_CHUNK_LENGTH", "60")),
        )
        if not settings.embedding_model or settings.retrieval_top_k < 1 or not -1 <= settings.retrieval_threshold <= 1:
            raise ValueError("تنظیمات retrieval نامعتبر است.")
        if not (100 <= settings.chunk_size <= 10_000 and 0 <= settings.chunk_overlap < settings.chunk_size
                and 1 <= settings.min_chunk_length <= settings.chunk_size and 1 <= settings.llm_timeout_seconds <= 300):
            raise ValueError("تنظیمات chunking یا timeout نامعتبر است.")
        return settings
