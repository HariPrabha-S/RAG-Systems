import json
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pytest

from src.embeddings.embedding_model import generate_embeddings
from src.vectorstore.faiss_manager import build_hnsw_index, save_index, save_metadata
from src.retrieval.retriever import Retriever
from src.utils.config import Settings


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

SAMPLE_CHUNKS = [
    {"id": "doc1::p1::c0", "text": "Fraud detection uses machine learning to identify suspicious transactions.", "document": "fraud.pdf", "page": 1, "chunk_index": 0},
    {"id": "doc1::p2::c1", "text": "Banks invest in real-time monitoring systems for compliance.", "document": "fraud.pdf", "page": 2, "chunk_index": 1},
    {"id": "doc2::p1::c2", "text": "Neural networks can classify images with high accuracy.", "document": "ml_guide.pdf", "page": 1, "chunk_index": 2},
    {"id": "doc2::p3::c3", "text": "Financial institutions face regulatory requirements for reporting.", "document": "fraud.pdf", "page": 3, "chunk_index": 3},
]


@pytest.fixture(scope="module")
def index_dir():
    """Build a temporary FAISS index for testing."""
    texts = [c["text"] for c in SAMPLE_CHUNKS]
    embeddings = generate_embeddings(texts, MODEL_NAME, normalize=True)
    index = build_hnsw_index(embeddings, m=16, ef_construction=40)

    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = Path(tmpdir)
        save_index(index, idx_path)
        meta_path = idx_path / "metadata.json"
        save_metadata(SAMPLE_CHUNKS, meta_path)
        yield idx_path, meta_path


@pytest.fixture
def retriever(index_dir):
    idx_path, meta_path = index_dir
    s = Settings.__new__(Settings)
    object.__setattr__(s, "index_path", idx_path)
    object.__setattr__(s, "metadata_path", meta_path)
    object.__setattr__(s, "hnsw_ef_search", 32)
    object.__setattr__(s, "embedding_model", MODEL_NAME)
    object.__setattr__(s, "top_k", 3)
    object.__setattr__(s, "similarity_threshold", 0.0)
    return Retriever(s)


def test_retrieve_returns_results(retriever):
    results = retriever.retrieve("How does fraud detection work?")
    assert len(results) > 0
    assert all(hasattr(r, "text") for r in results)
    assert all(hasattr(r, "score") for r in results)


def test_retrieve_relevance_ordering(retriever):
    results = retriever.retrieve("fraud detection machine learning")
    if len(results) >= 2:
        assert results[0].score >= results[1].score


def test_retrieve_mmr_returns_results(retriever):
    results = retriever.retrieve_mmr("banking compliance monitoring", candidates=4)
    assert len(results) > 0


def test_threshold_filters_results(retriever):
    results = retriever.retrieve("completely unrelated quantum physics topic", threshold=0.99)
    assert len(results) == 0 or all(r.score >= 0.99 for r in results)
