"""
Interactive RAG CLI application.

Usage:
    python src/main.py
    python -m src.main
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chains.rag_chain import RAGChain
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

BANNER = """
========================================================
            RAG Document Q&A System

  Commands:
    Type a question to get an answer
    /eval   - Retrieve only (no LLM generation)
    /mmr    - Use MMR retrieval for diversity
    /help   - Show this help message
    /quit   - Exit the application
========================================================
"""


def _display_response(answer: str, sources: list, elapsed: float) -> None:
    """Format and display the RAG response."""
    print(f"\n{'-' * 50}")
    print("Answer:\n")
    print(answer)
    print(f"\n{'-' * 50}")

    if sources:
        print("Sources:")
        seen = set()
        for s in sources:
            key = (s.document, s.page)
            if key not in seen:
                seen.add(key)
                print(f"  - {s.document} (page {s.page}) [score: {s.score:.3f}]")
        print(f"\nChunk IDs: {', '.join(s.chunk_id for s in sources)}")
    else:
        print("Sources: None")

    print(f"Latency: {elapsed:.2f}s")
    print(f"{'-' * 50}\n")


def _display_eval_results(chunks: list) -> None:
    """Display retrieved chunks without LLM generation."""
    print(f"\n{'-' * 50}")
    print(f"Retrieved {len(chunks)} chunks:\n")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] {c.document} (page {c.page}) | score: {c.score:.3f}")
        print(f"    {c.text[:150]}...")
        print()
    print(f"{'-' * 50}\n")


def main() -> None:
    """Run the interactive RAG CLI."""
    print(BANNER)

    try:
        logger.info("Initializing RAG chain...")
        chain = RAGChain(settings)
        logger.info("System ready.")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Run 'python -m src.ingestion.indexer' first to build the index.")
        sys.exit(1)
    except ConnectionError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print(BANNER)
            continue

        use_mmr = False
        eval_mode = False

        if user_input.lower().startswith("/mmr "):
            use_mmr = True
            user_input = user_input[5:].strip()
        elif user_input.lower().startswith("/eval "):
            eval_mode = True
            user_input = user_input[6:].strip()

        if not user_input:
            print("Please enter a question after the command.")
            continue

        start = time.perf_counter()

        try:
            if eval_mode:
                chunks = chain.retrieve_only(user_input)
                _display_eval_results(chunks)
            else:
                response = chain.query(user_input, use_mmr=use_mmr)
                elapsed = time.perf_counter() - start
                _display_response(response.answer, response.sources, elapsed)
        except Exception as e:
            logger.error("Error processing question: %s", e)
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
