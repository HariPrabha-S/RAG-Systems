import json
from pathlib import Path

import faiss
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_hnsw_index(
    embeddings: np.ndarray,
    m: int = 32,
    ef_construction: int = 200,
) -> faiss.IndexHNSWFlat:
    """Build a FAISS HNSW index from embeddings."""
    dim = embeddings.shape[1]
    logger.info("Building HNSW index: dim=%d, M=%d, efConstruction=%d", dim, m, ef_construction)

    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)

    logger.info("Index built with %d vectors", index.ntotal)
    return index


def save_index(index: faiss.IndexHNSWFlat, index_path: Path) -> None:
    """Persist FAISS index to disk."""
    index_path = Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    faiss_file = index_path / "index.faiss"
    faiss.write_index(index, str(faiss_file))
    logger.info("Index saved to %s", faiss_file)


def load_index(index_path: Path, ef_search: int = 128) -> faiss.IndexHNSWFlat:
    """Load a FAISS index from disk."""
    faiss_file = Path(index_path) / "index.faiss"
    if not faiss_file.exists():
        raise FileNotFoundError(f"FAISS index not found at {faiss_file}. Run indexer first.")

    index = faiss.read_index(str(faiss_file))
    index.hnsw.efSearch = ef_search
    logger.info("Loaded index with %d vectors (efSearch=%d)", index.ntotal, ef_search)
    return index


def save_metadata(metadata: list[dict], metadata_path: Path) -> None:
    """Save chunk metadata as JSON."""
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata saved: %d entries to %s", len(metadata), metadata_path)


def load_metadata(metadata_path: Path) -> list[dict]:
    """Load chunk metadata from JSON."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run indexer first.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise ValueError(f"Corrupt metadata file: expected list, got {type(metadata).__name__}")

    logger.info("Loaded metadata: %d entries", len(metadata))
    return metadata
