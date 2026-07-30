import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str) -> SentenceTransformer:
    """Load and cache the sentence transformer model."""
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info("Model loaded successfully")
    return _model_cache[model_name]


def generate_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Generate normalized embeddings for a list of texts."""
    model = get_model(model_name)
    logger.info("Generating embeddings for %d texts (batch_size=%d)", len(texts), batch_size)

    all_embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(batch_emb)

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


def get_embedding_dimension(model_name: str) -> int:
    """Return the embedding dimension for a given model."""
    model = get_model(model_name)
    return model.get_sentence_embedding_dimension()
