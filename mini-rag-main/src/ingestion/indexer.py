"""
Ingestion pipeline: load PDFs → chunk → embed → build FAISS index → save.

Usage:
    python -m src.ingestion.indexer
"""
import hashlib
import json
import time
from pathlib import Path

from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.text_splitter import create_chunks
from src.embeddings.embedding_model import generate_embeddings
from src.vectorstore.faiss_manager import build_hnsw_index, save_index, save_metadata
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_pdf_hash(pdf_dir: Path) -> str:
    """Hash PDF filenames and sizes to detect changes."""
    entries: list[str] = []
    for f in sorted(pdf_dir.glob("*.pdf")):
        entries.append(f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}")
    return hashlib.md5("|".join(entries).encode()).hexdigest()


def _load_existing_hash(index_path: Path) -> str | None:
    """Read the stored hash from the previous indexing run."""
    hash_file = index_path / "source_hash.txt"
    if hash_file.exists():
        return hash_file.read_text().strip()
    return None


def _save_hash(index_path: Path, hash_val: str) -> None:
    """Persist the current PDF hash."""
    hash_file = index_path / "source_hash.txt"
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(hash_val)


def run_indexing(force: bool = False) -> None:
    """Execute the full ingestion pipeline."""
    start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Starting ingestion pipeline")
    logger.info("=" * 60)

    # Check if re-indexing is needed
    current_hash = _compute_pdf_hash(settings.pdf_dir)
    if not force:
        existing_hash = _load_existing_hash(settings.index_path)
        if existing_hash == current_hash and (settings.index_path / "index.faiss").exists():
            logger.info("PDFs unchanged since last indexing. Use --force to re-index.")
            return

    # Step 1: Load PDFs
    logger.info("[1/4] Loading PDFs from %s", settings.pdf_dir)
    pages = load_pdfs_from_directory(settings.pdf_dir)

    total_chars = sum(len(p["text"]) for p in pages)
    unique_docs = len({p["document"] for p in pages})
    logger.info("PDF stats: %d documents, %d pages, %d characters", unique_docs, len(pages), total_chars)

    # Step 2: Chunk
    logger.info("[2/4] Chunking (size=%d, overlap=%d)", settings.chunk_size, settings.chunk_overlap)
    chunks = create_chunks(pages, settings.chunk_size, settings.chunk_overlap)

    # Step 3: Embed
    logger.info("[3/4] Generating embeddings with %s", settings.embedding_model)
    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts, settings.embedding_model)

    # Step 4: Build and save
    logger.info("[4/4] Building FAISS HNSW index")
    index = build_hnsw_index(embeddings, m=settings.hnsw_m, ef_construction=settings.hnsw_ef_construction)
    save_index(index, settings.index_path)

    metadata = [
        {"id": c["id"], "document": c["document"], "page": c["page"],
         "text": c["text"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]
    save_metadata(metadata, settings.metadata_path)
    _save_hash(settings.index_path, current_hash)

    elapsed = time.perf_counter() - start
    logger.info("=" * 60)
    logger.info("Indexing complete in %.2fs", elapsed)
    logger.info("  Chunks:  %d", len(chunks))
    logger.info("  Vectors: %d (dim=%d)", embeddings.shape[0], embeddings.shape[1])
    logger.info("  Index:   %s", settings.index_path / "index.faiss")
    logger.info("  Meta:    %s", settings.metadata_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    run_indexing(force=force)
