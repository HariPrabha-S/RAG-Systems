"""
Table 3 - Retrieval Strategy benchmark.

Compares four retrieval strategies over the project's real corpus and
eval_dataset.json queries:
  - Dense only        (cosine similarity over Qwen3 embeddings)
  - BM25 only          (keyword search)
  - Hybrid weighted    (alpha * dense + (1-alpha) * BM25, alpha=0.7)
  - Hybrid + Query Expansion (Ollama LLM expands the query, then hybrid)

Reports Recall@10, MRR, and end-to-end latency (ms) per strategy.

Usage:
    python evaluations/retrieval_benchmark.py
"""
import os
import sys
import json
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(CURRENT_DIR)

from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.text_splitter import create_chunks
from src.embeddings.embedding_model import generate_embeddings
from src.llm.ollama_llm import OllamaLLM
from src.utils.config import settings
from run_metrics import MetricsCalculator  # reuse is_match / normalize_text

EMB_CACHE = os.path.join(CURRENT_DIR, "_corpus_embeddings.npy")
QUERY_CACHE = os.path.join(CURRENT_DIR, "_query_embeddings.npy")
TOP_K = 10
ALPHA = 0.7
QE_MODEL = "llama3.2:3b"  # model actually pulled in local Ollama


class RetrievedStub:
    """Minimal stand-in matching the .text attribute MetricsCalculator expects."""
    def __init__(self, text):
        self.text = text


def load_corpus():
    pages = load_pdfs_from_directory(settings.pdf_dir)
    chunks = create_chunks(pages, settings.chunk_size, settings.chunk_overlap)
    texts = [c["text"] for c in chunks]

    if os.path.exists(EMB_CACHE):
        embeddings = np.load(EMB_CACHE)
    else:
        embeddings = generate_embeddings(texts, settings.embedding_model)
        np.save(EMB_CACHE, embeddings)

    assert embeddings.shape[0] == len(texts), "Cached embeddings do not match current chunking output"
    return texts, embeddings


def load_queries():
    with open(os.path.join(CURRENT_DIR, "eval_dataset.json"), encoding="utf-8") as f:
        dataset = json.load(f)

    if os.path.exists(QUERY_CACHE):
        query_embeddings = np.load(QUERY_CACHE)
        assert query_embeddings.shape[0] == len(dataset)
    else:
        questions = [item["question"] for item in dataset]
        query_embeddings = generate_embeddings(questions, settings.embedding_model)
        np.save(QUERY_CACHE, query_embeddings)

    return dataset, query_embeddings


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def evaluate_strategy(name, dataset, per_query_fn):
    """per_query_fn(item, query_idx) -> (ranked_texts: list[str], latency_s: float)"""
    recalls, mrrs, latencies = [], [], []
    for i, item in enumerate(dataset):
        ranked_texts, latency_s = per_query_fn(item, i)
        retrieved = [RetrievedStub(t) for t in ranked_texts[:TOP_K]]
        metrics = MetricsCalculator.evaluate_results(item["gold_chunk_text"], retrieved, k=TOP_K)
        recalls.append(metrics["recall"])
        mrrs.append(metrics["mrr"])
        latencies.append(latency_s * 1000)
    return {
        "strategy": name,
        "recall_at_10": float(np.mean(recalls)),
        "mrr": float(np.mean(mrrs)),
        "latency_ms": float(np.mean(latencies)),
    }


def main():
    print("Loading corpus and queries...")
    texts, corpus_emb = load_corpus()
    dataset, query_emb = load_queries()
    n = len(texts)
    print(f"Corpus: {n} chunks | Queries: {len(dataset)}\n")

    bm25 = BM25Okapi([tokenize(t) for t in texts])

    results = []

    # ---- 1. Dense only ----
    def dense_fn(item, i):
        t0 = time.perf_counter()
        q_emb = generate_embeddings([item["question"]], settings.embedding_model, batch_size=1, normalize=True)[0]
        scores = corpus_emb @ q_emb
        order = np.argsort(scores)[::-1][:TOP_K]
        latency = time.perf_counter() - t0
        return [texts[j] for j in order], latency

    print("=== Dense only ===")
    results.append(evaluate_strategy("Dense only", dataset, dense_fn))
    print(results[-1], "\n")

    # ---- 2. BM25 only ----
    def bm25_fn(item, i):
        t0 = time.perf_counter()
        tokens = tokenize(item["question"])
        scores = bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:TOP_K]
        latency = time.perf_counter() - t0
        return [texts[j] for j in order], latency

    print("=== BM25 only ===")
    results.append(evaluate_strategy("BM25 only", dataset, bm25_fn))
    print(results[-1], "\n")

    # ---- 3. Hybrid weighted (alpha=0.7) ----
    def hybrid_fn(item, i):
        t0 = time.perf_counter()
        q_emb = generate_embeddings([item["question"]], settings.embedding_model, batch_size=1, normalize=True)[0]
        dense_scores = corpus_emb @ q_emb
        tokens = tokenize(item["question"])
        bm25_scores = bm25.get_scores(tokens)
        combined = ALPHA * minmax(dense_scores) + (1 - ALPHA) * minmax(bm25_scores)
        order = np.argsort(combined)[::-1][:TOP_K]
        latency = time.perf_counter() - t0
        return [texts[j] for j in order], latency

    print(f"=== Hybrid weighted (alpha={ALPHA}) ===")
    results.append(evaluate_strategy(f"Hybrid weighted (a={ALPHA})", dataset, hybrid_fn))
    print(results[-1], "\n")

    # ---- 4. Hybrid + Query Expansion ----
    print(f"Connecting to Ollama ({QE_MODEL}) for query expansion...")
    llm = OllamaLLM(settings.ollama_base_url, QE_MODEL)
    qe_system = "You expand short search queries for retrieval. Respond with ONLY the expanded query: the original terms plus 3-5 relevant synonyms or related keywords, space-separated. No explanation, no punctuation commentary."

    def hybrid_qe_fn(item, i):
        t0 = time.perf_counter()
        expanded = llm.generate(qe_system, item["question"]).strip() or item["question"]
        q_emb = generate_embeddings([expanded], settings.embedding_model, batch_size=1, normalize=True)[0]
        dense_scores = corpus_emb @ q_emb
        tokens = tokenize(expanded)
        bm25_scores = bm25.get_scores(tokens)
        combined = ALPHA * minmax(dense_scores) + (1 - ALPHA) * minmax(bm25_scores)
        order = np.argsort(combined)[::-1][:TOP_K]
        latency = time.perf_counter() - t0
        return [texts[j] for j in order], latency

    print("=== Hybrid + Query Expansion ===")
    results.append(evaluate_strategy("Hybrid + Query Expansion", dataset, hybrid_qe_fn))
    print(results[-1], "\n")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print(f"{'Strategy':32s} {'Recall@10':>10s} {'MRR':>8s} {'Latency(ms)':>13s}")
    print("-" * 80)
    for r in results:
        print(f"{r['strategy']:32s} {r['recall_at_10']:>10.4f} {r['mrr']:>8.4f} {r['latency_ms']:>13.2f}")

    winner = max(results, key=lambda r: (r["recall_at_10"], r["mrr"]))
    print("-" * 80)
    print(f"Suggested winner (best recall/MRR): {winner['strategy']}")
    print("=" * 80)

    with open(os.path.join(CURRENT_DIR, "retrieval_benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump({"n_corpus": n, "n_queries": len(dataset), "alpha": ALPHA, "results": results}, f, indent=2)
    print("\nSaved results to evaluations/retrieval_benchmark_results.json")


if __name__ == "__main__":
    main()
