import json
import time
from pathlib import Path
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.chains.rag_chain import RAGChain
from src.utils.config import settings
from run_metrics import MetricsCalculator

# Fixed Top-k (winner from previous experiment)
TOP_K = 5

dataset_path = Path(__file__).parent / "eval_dataset.json"

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

chain = RAGChain(settings)

print("\n")
print("=" * 70)
print("Baseline Retrieval (No Reranker)")
print("=" * 70)

total_relevant = 0
total_precision = 0
total_recall = 0
total_latency = 0

for item in dataset:

    start = time.perf_counter()

    retrieved = chain.retrieve_only(
        item["question"],
        top_k=TOP_K
    )

    latency = (time.perf_counter() - start) * 1000

    metrics = MetricsCalculator.evaluate_results(
        item["gold_chunk_text"],
        retrieved,
        k=TOP_K
    )

    relevant = sum(
        MetricsCalculator.is_match(
            item["gold_chunk_text"],
            c.text
        )
        for c in retrieved
    )

    total_relevant += relevant
    total_precision += metrics["context_precision"]
    total_recall += metrics["recall"]
    total_latency += latency

n = len(dataset)

avg_relevant = total_relevant / n
avg_precision = total_precision / n
avg_recall = total_recall / n
avg_latency = total_latency / n

print(f"Top-k               : {TOP_K}")
print(f"Relevant Chunks     : {avg_relevant:.2f}")
print(f"Context Precision   : {avg_precision:.3f}")
print(f"Recall@10           : {avg_recall:.3f}")
print(f"Latency             : {avg_latency:.2f} ms")

print("=" * 70)