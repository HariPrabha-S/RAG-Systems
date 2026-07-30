RAG_SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions based strictly on the provided context.

Rules:
1. Only use information from the provided context to answer.
2. If the context does not contain enough information, say "I don't have enough information in the provided documents to answer this question."
3. Never fabricate or assume information not present in the context.
4. When referencing information, cite the source document and page number.
5. Keep answers concise and well-structured."""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}

Provide a clear, accurate answer based only on the context above. Cite source documents and page numbers."""


def format_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    """Format the RAG prompt with retrieved context."""
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[Source {i}: {chunk['document']}, Page {chunk['page']}]"
        context_parts.append(f"{source}\n{chunk['text']}")

    context = "\n\n".join(context_parts)
    user_message = RAG_USER_TEMPLATE.format(context=context, question=question)
    return RAG_SYSTEM_PROMPT, user_message
