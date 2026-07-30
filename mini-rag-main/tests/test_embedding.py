import numpy as np
import pytest

from src.embeddings.embedding_model import generate_embeddings, get_embedding_dimension


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


@pytest.fixture(scope="module")
def sample_texts():
    return [
        "Fraud detection in financial services uses machine learning.",
        "Banks invest heavily in digital security and compliance.",
        "The weather today is sunny with a chance of rain.",
    ]


def test_embedding_shape(sample_texts):
    embeddings = generate_embeddings(sample_texts, MODEL_NAME, batch_size=2)
    assert embeddings.shape[0] == len(sample_texts)
    assert embeddings.shape[1] > 0


def test_embeddings_normalized(sample_texts):
    embeddings = generate_embeddings(sample_texts, MODEL_NAME, normalize=True)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_similar_texts_closer(sample_texts):
    embeddings = generate_embeddings(sample_texts, MODEL_NAME, normalize=True)
    sim_01 = float(embeddings[0] @ embeddings[1])
    sim_02 = float(embeddings[0] @ embeddings[2])
    assert sim_01 > sim_02, "Related texts should have higher similarity"


def test_get_dimension():
    dim = get_embedding_dimension(MODEL_NAME)
    assert isinstance(dim, int)
    assert dim > 0
