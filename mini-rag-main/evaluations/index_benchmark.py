"""
Table 2 - Indexing benchmark.

Builds the corpus embeddings once (cached to disk), then builds several
FAISS index types over them and measures:
  - build time (s)
  - query latency (ms/query, averaged over the eval_dataset questions)
  - recall vs exact (top-10 overlap against a Flat/IndexFlatIP baseline)

Usage:
    python evaluations/index_benchmark.py
"""
import os
import sys
import json
import time

import numpy as np
import faiss

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.text_splitter import create_chunks
from src.embeddings.embedding_model import generate_embeddings
from src.utils.config import settings

EMB_CACHE = os.path.join(CURRENT_DIR, "_corpus_embeddings.npy")
QUERY_CACHE = os.path.join(CURRENT_DIR, "_query_embeddings.npy")
TOP_K = 10
N_QUERY_REPEATS = 5  # repeat each query search this many times for stable latency


def build_corpus_embeddings() -> np.ndarray:
    if os.path.exists(EMB_CACHE):
        emb = np.load(EMB_CACHE)
        print(f"Loaded cached corpus embeddings: {emb.shape}")
        return emb

    print("Building corpus embeddings from PDFs (first run, no cache found)...")
    pages = load_pdfs_from_directory(settings.pdf_dir)
    chunks = create_chunks(pages, settings.chunk_size, settings.chunk_overlap)
    texts = [c["text"] for c in chunks]
    emb = generate_embeddings(texts, settings.embedding_model)
    np.save(EMB_CACHE, emb)
    print(f"Corpus embeddings built: {emb.shape}")
    return emb


def build_query_embeddings() -> np.ndarray:
    if os.path.exists(QUERY_CACHE):
        emb = np.load(QUERY_CACHE)
        print(f"Loaded cached query embeddings: {emb.shape}")
        return emb

    with open(os.path.join(CURRENT_DIR, "eval_dataset.json"), encoding="utf-8") as f:
        dataset = json.load(f)
    questions = [item["question"] for item in dataset]
    emb = generate_embeddings(questions, settings.embedding_model)
    np.save(QUERY_CACHE, emb)
    print(f"Query embeddings built: {emb.shape}")
    return emb


def time_search(index, queries: np.ndarray, k: int):
    """Return (avg_latency_ms, all_indices[n_queries, k])."""
    # warm-up
    index.search(queries[:1], k)

    all_ids = []
    total_time = 0.0
    n_runs = 0
    for _ in range(N_QUERY_REPEATS):
        start = time.perf_counter()
        _, ids = index.search(queries, k)
        total_time += time.perf_counter() - start
        n_runs += 1
        all_ids = ids  # keep the last run's ids for recall calc

    avg_latency_ms = (total_time / n_runs / queries.shape[0]) * 1000
    return avg_latency_ms, all_ids


def recall_vs_exact(approx_ids: np.ndarray, exact_ids: np.ndarray) -> float:
    """Mean fraction of exact top-k neighbors also found in approx top-k, per query."""
    n_queries, k = exact_ids.shape
    scores = []
    for i in range(n_queries):
        exact_set = set(exact_ids[i].tolist())
        approx_set = set(approx_ids[i].tolist())
        scores.append(len(exact_set & approx_set) / k)
    return float(np.mean(scores))


def main():
    corpus = build_corpus_embeddings()
    queries = build_query_embeddings()
    n, dim = corpus.shape
    print(f"\nCorpus: {n} vectors, dim={dim} | Queries: {queries.shape[0]}\n")

    results = []

    # ---- 1. Flat (Exact) — the ground truth ----
    print("=== Flat (Exact) ===")
    t0 = time.perf_counter()
    flat = faiss.IndexFlatIP(dim)
    flat.add(corpus)
    build_time = time.perf_counter() - t0
    latency_ms, exact_ids = time_search(flat, queries, TOP_K)
    results.append({"index": "Flat (Exact)", "build_s": build_time, "latency_ms": latency_ms, "recall": 1.0})
    print(f"build={build_time:.4f}s  latency={latency_ms:.4f}ms  recall=1.0000 (baseline)\n")

    # nlist for IVF: rule of thumb ~ min(4*sqrt(n), n // 39) but must be >= 1 and <= n
    nlist = max(1, min(int(4 * np.sqrt(n)), n // 4, 100))
    nprobe = max(1, nlist // 8)

    # ---- 2. IVF-Flat ----
    print(f"=== IVF-Flat (nlist={nlist}, nprobe={nprobe}) ===")
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatIP(dim)
    ivf_flat = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf_flat.train(corpus)
    ivf_flat.add(corpus)
    ivf_flat.nprobe = nprobe
    build_time = time.perf_counter() - t0
    latency_ms, ids = time_search(ivf_flat, queries, TOP_K)
    recall = recall_vs_exact(ids, exact_ids)
    results.append({"index": f"IVF-Flat (nlist={nlist}, nprobe={nprobe})", "build_s": build_time, "latency_ms": latency_ms, "recall": recall})
    print(f"build={build_time:.4f}s  latency={latency_ms:.4f}ms  recall={recall:.4f}\n")

    # ---- 3. IVF-PQ ----
    # PQ needs dim divisible by m; pick a divisor of dim close to dim/4 (each subvector >=1 dim)
    m_candidates = [m for m in [64, 48, 32, 16, 8, 4, 2, 1] if dim % m == 0]
    m = m_candidates[0] if m_candidates else 1
    # PQ trains 2**nbits centroids per sub-quantizer; corpus must have >= 2**nbits points.
    nbits = max(1, min(8, int(np.floor(np.log2(max(2, n // 2))))))
    print(f"=== IVF-PQ (nlist={nlist}, nprobe={nprobe}, m={m}, nbits={nbits}) ===")
    t0 = time.perf_counter()
    quantizer2 = faiss.IndexFlatIP(dim)
    ivf_pq = faiss.IndexIVFPQ(quantizer2, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    ivf_pq.train(corpus)
    ivf_pq.add(corpus)
    ivf_pq.nprobe = nprobe
    build_time = time.perf_counter() - t0
    latency_ms, ids = time_search(ivf_pq, queries, TOP_K)
    recall = recall_vs_exact(ids, exact_ids)
    results.append({"index": f"IVF-PQ (m={m}, nbits={nbits})", "build_s": build_time, "latency_ms": latency_ms, "recall": recall})
    print(f"build={build_time:.4f}s  latency={latency_ms:.4f}ms  recall={recall:.4f}\n")

    # ---- 4. HNSW (M=32, ef_construction=200, ef_search=100) ----
    for m_hnsw, ef_c, ef_s in [(32, 200, 100), (64, 400, 200)]:
        print(f"=== HNSW (M={m_hnsw}, ef_c={ef_c}, ef_s={ef_s}) ===")
        t0 = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(dim, m_hnsw, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = ef_c
        hnsw.add(corpus)
        hnsw.hnsw.efSearch = ef_s
        build_time = time.perf_counter() - t0
        latency_ms, ids = time_search(hnsw, queries, TOP_K)
        recall = recall_vs_exact(ids, exact_ids)
        results.append({"index": f"HNSW (M={m_hnsw}, ef_c={ef_c}, ef_s={ef_s})", "build_s": build_time, "latency_ms": latency_ms, "recall": recall})
        print(f"build={build_time:.4f}s  latency={latency_ms:.4f}ms  recall={recall:.4f}\n")

    # ---- 5. HNSW + PQ (sample values: M=32, ef_c=200, ef_s=100, PQ m/nbits from above) ----
    print(f"=== HNSW + PQ (M=32, ef_c=200, ef_s=100, m={m}, nbits={nbits}) ===")
    t0 = time.perf_counter()
    hnsw_pq = faiss.IndexHNSWPQ(dim, m, 32, nbits)
    hnsw_pq.hnsw.efConstruction = 200
    hnsw_pq.train(corpus)
    hnsw_pq.add(corpus)
    hnsw_pq.hnsw.efSearch = 100
    build_time = time.perf_counter() - t0
    latency_ms, ids = time_search(hnsw_pq, queries, TOP_K)
    recall = recall_vs_exact(ids, exact_ids)
    results.append({"index": f"HNSW+PQ (M=32, m={m}, nbits={nbits})", "build_s": build_time, "latency_ms": latency_ms, "recall": recall})
    print(f"build={build_time:.4f}s  latency={latency_ms:.4f}ms  recall={recall:.4f}\n")

    # ---- Summary table ----
    print("\n" + "=" * 90)
    print(f"{'Index':45s} {'Build(s)':>10s} {'Latency(ms)':>13s} {'Recall':>8s}")
    print("-" * 90)
    for r in results:
        print(f"{r['index']:45s} {r['build_s']:>10.4f} {r['latency_ms']:>13.4f} {r['recall']:>8.4f}")

    winner = max(results[1:], key=lambda r: (r["recall"], -r["latency_ms"]))
    print("-" * 90)
    print(f"Suggested winner (fast + high recall): {winner['index']}")
    print("=" * 90)

    with open(os.path.join(CURRENT_DIR, "index_benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump({"n_vectors": n, "dim": dim, "n_queries": queries.shape[0], "results": results}, f, indent=2)
    print(f"\nSaved results to evaluations/index_benchmark_results.json")


if __name__ == "__main__":
    main()
