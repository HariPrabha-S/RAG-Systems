# Mini-RAG Retrieval Evaluation
### Person A — Evaluation Dataset & Retrieval Metrics

---

## Overview

This module evaluates the retrieval quality of the **Mini-RAG** system using a manually created evaluation dataset based on the **TCS White Paper – Fighting Fraud**.

The objective is to measure how accurately the **FAISS retriever** returns the correct document chunks before answer generation.

This work is assigned to **Person A** in the project.

---

## Person A — Responsibilities

- [x] Create evaluation dataset
- [x] Prepare 30–50 evaluation questions
- [x] Define gold chunk (ground truth)
- [x] Integrate evaluation with existing Mini-RAG retrieval pipeline
- [x] Calculate retrieval metrics
- [x] Measure retrieval latency
- [x] Generate evaluation reports
- [x] Produce CSV for spreadsheet
- [x] Produce summary report
- [x] Generate plots for visualization
- [x] Deliver outputs for Person B

---

## Existing Project Used

This evaluation is integrated with the existing Mini-RAG project. **No modifications were made to the retrieval pipeline** — only an evaluation layer was added on top.

**Existing modules reused:**

```
src/
├── chains/
│   └── rag_chain.py
├── retrieval/
│   └── retriever.py
├── vectorstore/
│   └── faiss_manager.py
├── embeddings/
│   └── embedding_model.py
└── utils/
    └── config.py
```

---

## Folder Structure

```
mini-rag-main/
└── evaluations/
    ├── eval_dataset.json
    ├── run_metrics.py
    ├── results.csv
    ├── summary.txt
    └── plots/
        ├── recall_plot.png
        ├── mrr_plot.png
        ├── ndcg_plot.png
        ├── latency_plot.png
        └── context_precision_plot.png
```

---

## Evaluation Dataset

The dataset consists of manually prepared evaluation questions extracted from the **TCS Fighting Fraud White Paper**.

Each record contains:

```json
{
  "query_id": "q1",
  "question": "...",
  "gold_chunk_text": "...",
  "source_doc": "fraud.pdf"
}
```

**Dataset size:** 50 questions

---

## Evaluation Pipeline

The evaluation pipeline performs the following steps for every query:

1. Load question
2. Generate query embedding
3. Retrieve top-K chunks
4. Compare retrieved chunk with gold chunk
5. Calculate metrics
6. Store result
7. Generate reports

---

## Retrieval Metrics

### Recall@10

Measures whether the correct chunk appears within the top 10 retrieved chunks.

**Formula:**

```
Recall@10 = 1   if Gold Chunk is Retrieved
Recall@10 = 0   otherwise
```

---

### Mean Reciprocal Rank (MRR)

Measures how early the correct chunk appears.

**Formula:**

```
MRR = 1 / Rank
```

**Example:**

| Correct chunk position | MRR |
|---|---|
| Rank 1 | 1.0 |
| Rank 2 | 0.5 |

---

### nDCG@10

Measures ranking quality. A higher score indicates the relevant chunk appears nearer the top.

**Range:** `0` (worst) → `1` (best)

---

### Context Precision

Measures the percentage of retrieved chunks that are actually relevant.

**Formula:**

```
Context Precision = Relevant Retrieved Chunks / Total Retrieved Chunks
```

---

### Latency

Measures retrieval time, reported in milliseconds (ms).

---

## Execution Flow

1. Load dataset
2. Initialize RAGChain
3. For every question →
4. Retrieve top-K chunks
5. Compute metrics
6. Save CSV
7. Save summary
8. Generate plots
9. Finish

---

## Generated Outputs

Running:

```bash
python evaluations/run_metrics.py
```

automatically produces:

- `results.csv`
- `summary.txt`
- `plots/`

No manual calculations are required.

---

## `results.csv`

Contains per-query evaluation results. Columns include:

| Column | Description |
|---|---|
| Query ID | Unique identifier for the question |
| Question | The evaluation question text |
| Recall | Recall@10 score |
| MRR | Mean Reciprocal Rank |
| nDCG | nDCG@10 score |
| Context Precision | Relevance ratio of retrieved chunks |
| Latency | Retrieval time (ms) |
| Retrieved Document | Source document name |
| Retrieved Page | Page number of retrieved chunk |
| Similarity Score | Vector similarity score |
| Retrieved Chunk IDs | IDs of retrieved chunks |
| Retrieved Text | Text content of retrieved chunks |

---

## `summary.txt`

Contains overall retrieval performance, including:

- Number of queries
- Average Recall
- Average MRR
- Average nDCG
- Average Context Precision
- Average Latency
- Best query
- Worst query
- Total runtime

---

## Generated Plots

The evaluation script automatically creates:

- Recall plot
- MRR plot
- Latency plot
- nDCG plot
- Context Precision plot

These plots are used by **Person B** while preparing the project report.

---

## Tools Used

**Programming Language**

- Python 3.11+

**Libraries**

- `json`
- `csv`
- `matplotlib`
- `numpy`
- `logging`
- `time`
- `math`
- `re`
- `pathlib`

**Mini-RAG Components**

- FAISS
- HNSW Index
- Sentence Embeddings
- Ollama
- Llama 3.2

---

## Evaluation Flow Diagram

```
┌────────────────────────┐
│   Evaluation Dataset    │
└────────────┬────────────┘
             │
             ▼
┌────────────────────────┐
│   Question Embedding    │
└────────────┬────────────┘
             │
             ▼
┌────────────────────────┐
│     FAISS Retriever     │
└────────────┬────────────┘
             │
     Top-K Retrieved Chunks
             │
             ▼
┌────────────────────────┐
│     Gold Chunk Match     │
└────────────┬────────────┘
             │
     ┌───────┴────────┐
     ▼                 ▼
┌──────────┐     ┌──────────────┐
│ Calculate │     │   Measure    │
│  Metrics  │     │   Latency    │
└─────┬─────┘     └──────┬───────┘
      │                  │
      └────────┬─────────┘
               ▼
   ┌───────────────────────┐
   │  Save Evaluation Results │
   └────────────┬────────────┘
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
results.csv  summary.txt  plots/
```

---

## Deliverables for Person B

Person A provides:

- `eval_dataset.json`
- `run_metrics.py`
- `results.csv`
- `summary.txt`
- `plots/`

Person B uses these outputs for:

- Google Spreadsheet
- Metric analysis
- Graph interpretation
- Project report
- Final presentation

---

## Deliverables for Person C

Person C uses:

- `results.csv`
- `summary.txt`
- `plots/`

for:

- Final documentation
- Performance discussion
- Future improvements
- Final demonstration

---

## How to Execute

**1. Activate virtual environment**

```bash
venv\Scripts\activate
```

**2. Navigate to project**

```bash
cd mini-rag-main
```

**3. Run evaluation**

```bash
python evaluations/run_metrics.py
```

Outputs are automatically generated inside `evaluations/`.

---

## Final Outputs

```
evaluations/
├── eval_dataset.json
├── run_metrics.py
├── results.csv
├── summary.txt
└── plots/
```

---

## Author

**Person A**
*Mini-RAG Evaluation*

- Retrieval performance analysis
- Dataset construction
- Metrics calculation
- Performance reporting