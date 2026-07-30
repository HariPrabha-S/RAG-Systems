import re
from typing import TypedDict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkData(TypedDict):
    id: str
    text: str
    document: str
    page: int
    chunk_index: int


# Sentence splitter
SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')

# Paragraph splitter
PARAGRAPH_BREAK = re.compile(r'\n\s*\n')


def split_paragraphs(text: str) -> list[str]:
    """Split page into paragraphs."""
    paragraphs = [p.strip() for p in PARAGRAPH_BREAK.split(text) if p.strip()]

    # If no paragraph breaks exist, treat the whole page as one paragraph.
    if not paragraphs:
        paragraphs = [text.strip()]

    return paragraphs


def split_sentences(text: str) -> list[str]:
    """Split paragraph into sentences."""
    sentences = [s.strip() for s in SENTENCE_ENDINGS.split(text) if s.strip()]

    # If sentence detection fails, keep the paragraph intact.
    if not sentences:
        sentences = [text.strip()]

    return sentences


def create_chunks(
    pages: list[dict],
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
) -> list[ChunkData]:

    all_chunks: list[ChunkData] = []
    global_index = 0

    for page_data in pages:

        text = page_data["text"]
        document = page_data["document"]
        page = page_data["page"]

        paragraphs = split_paragraphs(text)

        current_chunk = ""
        current_length = 0

        for paragraph in paragraphs:

            # ---------------------------------------
            # Paragraph fits
            # ---------------------------------------
            if len(paragraph) <= chunk_size:

                if current_length + len(paragraph) <= chunk_size:

                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph

                    current_length = len(current_chunk)

                else:

                    all_chunks.append({
                        "id": f"{document}::p{page}::c{global_index}",
                        "text": current_chunk.strip(),
                        "document": document,
                        "page": page,
                        "chunk_index": global_index,
                    })

                    global_index += 1

                    overlap = (
                        current_chunk[-chunk_overlap:]
                        if chunk_overlap > 0
                        else ""
                    )

                    current_chunk = overlap + "\n\n" + paragraph
                    current_length = len(current_chunk)

            # ---------------------------------------
            # Large paragraph → split into sentences
            # ---------------------------------------
            else:

                sentences = split_sentences(paragraph)

                sentence_chunk = ""
                sentence_length = 0

                for sentence in sentences:

                    if sentence_length + len(sentence) <= chunk_size:

                        if sentence_chunk:
                            sentence_chunk += " " + sentence
                        else:
                            sentence_chunk = sentence

                        sentence_length = len(sentence_chunk)

                    else:

                        all_chunks.append({
                            "id": f"{document}::p{page}::c{global_index}",
                            "text": sentence_chunk.strip(),
                            "document": document,
                            "page": page,
                            "chunk_index": global_index,
                        })

                        global_index += 1

                        overlap = (
                            sentence_chunk[-chunk_overlap:]
                            if chunk_overlap > 0
                            else ""
                        )

                        sentence_chunk = overlap + " " + sentence
                        sentence_length = len(sentence_chunk)

                if sentence_chunk:

                    if current_length + len(sentence_chunk) <= chunk_size:

                        if current_chunk:
                            current_chunk += "\n\n" + sentence_chunk
                        else:
                            current_chunk = sentence_chunk

                        current_length = len(current_chunk)

                    else:

                        if current_chunk.strip():

                            all_chunks.append({
                                "id": f"{document}::p{page}::c{global_index}",
                                "text": current_chunk.strip(),
                                "document": document,
                                "page": page,
                                "chunk_index": global_index,
                            })

                            global_index += 1

                        current_chunk = sentence_chunk
                        current_length = len(current_chunk)

        # ---------------------------------------
        # Save final chunk
        # ---------------------------------------
        if current_chunk.strip():

            all_chunks.append({
                "id": f"{document}::p{page}::c{global_index}",
                "text": current_chunk.strip(),
                "document": document,
                "page": page,
                "chunk_index": global_index,
            })

            global_index += 1

    logger.info(
        "Created %d hierarchical chunks from %d pages",
        len(all_chunks),
        len(pages),
    )

    return all_chunks