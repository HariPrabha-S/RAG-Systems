## MINI-RAG SYSTEM

A local RAG (Retrieval-Augmented Generation) pipeline that lets you ask questions about your PDF documents using FAISS and Ollama. Everything runs on your machine -- no API keys, no cloud dependencies.

I built this to understand how RAG systems actually work under the hood, without hiding behind LangChain abstractions. The whole thing is ~1200 lines of Python.

## What it does

You drop PDFs into a folder, run the indexer, and then ask questions through a CLI. The system finds the most relevant chunks from your documents using vector similarity search (FAISS HNSW), feeds them as context to a local LLM (via Ollama), and gives you an answer with source citations.

## How the pipeline works

1. **Ingestion** -- PDFs are parsed with PyMuPDF, split into sentence-aware chunks, and embedded using `all-MiniLM-L6-v2`
2. **Indexing** -- Embeddings are stored in a FAISS HNSW index. Metadata (document name, page number, text) is saved separately as JSON
3. **Retrieval** -- Query gets embedded, FAISS returns the top-k nearest chunks. There's also an MMR option if you want more diverse results
4. **Generation** -- Retrieved chunks are formatted into a grounded prompt and sent to Ollama. The LLM is instructed to only use the provided context

## Chunking approach

I went with sentence-aware chunking instead of naive fixed-length splitting. The difference matters -- fixed-length splits will happily cut a sentence in half, which kills retrieval quality because the embedding for a half-sentence is basically garbage.

Here's how it works:

1. The text from each PDF page is first split on paragraph breaks (`\n\n`), then each paragraph is split into sentences using punctuation boundaries (`.`, `!`, `?`)
2. Sentences are accumulated into a chunk until adding the next sentence would exceed `CHUNK_SIZE` (default 512 chars)
3. When a chunk is full, the last few sentences (up to `CHUNK_OVERLAP` chars worth) carry over into the next chunk. This overlap keeps context from getting lost at chunk boundaries

So instead of cutting at character 512 mid-word, every chunk starts and ends at a sentence boundary. Each chunk also keeps track of which document and page it came from, which is how we do source attribution later.

The chunk size and overlap are configurable in `.env`. I found 512/64 works well for most documents -- small enough to be specific, big enough to carry a complete thought.

## Project layout

```
src/
  ingestion/
    pdf_loader.py       -- extracts text from PDFs page by page
    text_splitter.py    -- sentence-aware chunking with overlap
    indexer.py          -- runs the full ingest pipeline
  embeddings/
    embedding_model.py  -- sentence-transformers wrapper, handles batching
  vectorstore/
    faiss_manager.py    -- build, save, load HNSW indices
  retrieval/
    retriever.py        -- top-k and MMR retrieval
  llm/
    ollama_llm.py       -- talks to Ollama's API
  chains/
    rag_chain.py        -- wires retrieval + generation together
  prompts/
    rag_prompt.py       -- prompt template
  utils/
    config.py           -- reads .env, exposes a Settings object
    logger.py           -- logging setup
  main.py               -- interactive CLI

data/raw/pdfs/          -- put your PDFs here
vectorstore/faiss_index/ -- index + metadata get saved here
tests/                   -- embedding and retrieval tests
```

## Getting started

You'll need Python 3.11+ and [Ollama](https://ollama.com/) installed.

```bash
git clone https://github.com/Bii05/mini_rag.git
cd mini_rag
pip install -r requirements.txt
```

Set up your `.env` file in the project root:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=5
CHUNK_SIZE=512
CHUNK_OVERLAP=64
SIMILARITY_THRESHOLD=0.3
```

Make sure Ollama is running and you've pulled a model:

```bash
ollama serve
ollama pull llama3.2:1b
```

## Usage

**Step 1:** Drop your PDF files into `data/raw/pdfs/`

**Step 2:** Build the index

```bash
python -m src.ingestion.indexer
```

This loads the PDFs, chunks them, generates embeddings, and builds the FAISS index. It also tracks file hashes so it won't re-index unless something changed (pass `--force` to override).

**Step 3:** Start asking questions

```bash
python src/main.py
```

You'll get an interactive prompt. Just type your question and hit enter.

```
Question: How does fraud detection work in financial services?

--------------------------------------------------
Answer:

According to the provided context, fighting fraud in financial
services involves monitoring, detecting, and reporting suspicious
or fraudulent activity within financial institutions...

--------------------------------------------------
Sources:
  - fighting-fraud-in-financial-services.pdf (page 1) [score: 0.639]
  - fighting-fraud-in-financial-services.pdf (page 2) [score: 0.567]
  - fighting-fraud-in-financial-services.pdf (page 3) [score: 0.551]

Latency: 35.70s
--------------------------------------------------
```

There are a couple of extra commands in the CLI:

- `/eval <question>` -- retrieves chunks without calling the LLM (useful for debugging retrieval quality)
- `/mmr <question>` -- uses Maximal Marginal Relevance to get more diverse results
- `/quit` -- exit

## Running tests

```bash
pytest tests/ -v
```

Tests cover embedding generation (shape, normalization, semantic similarity) and the retrieval pipeline (ranking, MMR, threshold filtering).

## Configuration

Everything is controlled through `.env`. The main knobs:

| Variable | What it controls | Default |
|---|---|---|
| `OLLAMA_MODEL` | Which Ollama model to use | `llama3.2:1b` |
| `EMBEDDING_MODEL` | Sentence-transformers model for embeddings | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Max characters per chunk | `512` |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks | `64` |
| `TOP_K` | Number of chunks to retrieve | `5` |
| `SIMILARITY_THRESHOLD` | Minimum similarity score to keep a result | `0.3` |
| `HNSW_M` | HNSW graph connectivity | `32` |

## Things I'd like to add

- Hybrid search combining BM25 with vector similarity
- Cross-encoder reranking for better precision
- Conversation memory so follow-up questions work
- A simple web UI (probably Streamlit or FastAPI)
- Support for other vector stores like ChromaDB

## Tech stack

Python, FAISS, sentence-transformers, PyMuPDF, Ollama, numpy

## Output
<img width="1652" height="952" alt="Screenshot 2026-06-21 204101" src="https://github.com/user-attachments/assets/511515b1-52eb-4679-b1d6-209866c116ba" />

