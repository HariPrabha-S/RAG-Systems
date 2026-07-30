from dataclasses import dataclass

from src.retrieval.retriever import Retriever, RetrievedChunk
from src.llm.ollama_llm import OllamaLLM
from src.prompts.rag_prompt import format_prompt
from src.utils.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]


class RAGChain:
    """End-to-end RAG pipeline: retrieve context then generate an answer."""

    def __init__(self, settings: Settings):
        self.retriever = Retriever(settings)
        self.llm = OllamaLLM(settings.ollama_base_url, settings.ollama_model)

    def query(self, question: str, use_mmr: bool = False) -> RAGResponse:
        """Run the full RAG pipeline for a question."""
        logger.info("Processing question: %s", question[:80])

        if use_mmr:
            chunks = self.retriever.retrieve_mmr(question)
        else:
            chunks = self.retriever.retrieve(question)

        if not chunks:
            return RAGResponse(
                answer="No relevant documents found for your question.",
                sources=[],
            )

        context_dicts = [
            {"text": c.text, "document": c.document, "page": c.page}
            for c in chunks
        ]

        system_prompt, user_prompt = format_prompt(question, context_dicts)
        answer = self.llm.generate(system_prompt, user_prompt)

        return RAGResponse(answer=answer, sources=chunks)

    def retrieve_only(
        self,
        question: str,
        top_k: int | None = None,
        use_mmr: bool = False,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks without LLM generation."""
        if use_mmr:
            return self.retriever.retrieve_mmr(question, top_k=top_k)

        return self.retriever.retrieve(question, top_k=top_k)