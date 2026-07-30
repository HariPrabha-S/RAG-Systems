import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {key}={raw!r} is not a valid integer")


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Environment variable {key}={raw!r} is not a valid float")


@dataclass(frozen=True)
class Settings:
    # Ollama
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.2"))

    # Embedding
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"))
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN", ""))

    # Paths
    pdf_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env("PDF_DIR", "data/raw/pdfs"))
    index_path: Path = field(default_factory=lambda: PROJECT_ROOT / _env("INDEX_PATH", "vectorstore/faiss_index"))
    metadata_path: Path = field(default_factory=lambda: PROJECT_ROOT / _env("METADATA_PATH", "vectorstore/faiss_index/metadata.json"))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    similarity_threshold: float = field(default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.3))

    # Chunking
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 64))

    # HNSW
    hnsw_m: int = field(default_factory=lambda: _env_int("HNSW_M", 32))
    hnsw_ef_construction: int = field(default_factory=lambda: _env_int("HNSW_EF_CONSTRUCTION", 200))
    hnsw_ef_search: int = field(default_factory=lambda: _env_int("HNSW_EF_SEARCH", 128))


settings = Settings()
