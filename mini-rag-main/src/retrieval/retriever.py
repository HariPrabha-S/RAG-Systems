import time
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import CrossEncoder
from src.embeddings.embedding_model import generate_embeddings
from src.vectorstore.faiss_manager import load_index, load_metadata
from src.utils.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    document: str
    page: int
    chunk_id: str


class Retriever:
    """Retrieve relevant chunks from a FAISS index."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.index = load_index(settings.index_path, settings.hnsw_ef_search)
        self.metadata = load_metadata(settings.metadata_path)
        self.bge_reranker = CrossEncoder("BAAI/bge-reranker-base")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a query with similarity filtering."""
        top_k = top_k or self.settings.top_k
        threshold = threshold if threshold is not None else self.settings.similarity_threshold

        start = time.perf_counter()
        query_embedding = generate_embeddings(
            [query], self.settings.embedding_model, batch_size=1, normalize=True
        )

        distances, indices = self.index.search(query_embedding, top_k)
        elapsed = time.perf_counter() - start
        logger.info("Retrieval completed in %.3fs", elapsed)

        results: list[RetrievedChunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            score = 1.0 / (1.0 + dist)
            if score < threshold:
                continue

            meta = self.metadata[idx]
            results.append(RetrievedChunk(
                text=meta["text"],
                score=score,
                document=meta["document"],
                page=meta["page"],
                chunk_id=meta["id"],
            ))

        logger.info("Retrieved %d chunks above threshold %.2f", len(results), threshold)
        return results
def retrieve_bge(
    self,
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve using FAISS and rerank with BGE-Reranker.
    """

    # First retrieve normally
    retrieved = self.retrieve(
        query,
        top_k=top_k,
        threshold=threshold,
    )

    if len(retrieved) <= 1:
        return retrieved

    pairs = [(query, chunk.text) for chunk in retrieved]

    scores = self.bge_reranker.predict(pairs)

    reranked = [
        chunk for _, chunk in sorted(
            zip(scores, retrieved),
            key=lambda x: x[0],
            reverse=True
        )
    ]

    return reranked
    