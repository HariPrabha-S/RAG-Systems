from pathlib import Path
from typing import TypedDict

import fitz

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PageData(TypedDict):
    text: str
    page: int
    document: str


def load_pdf(pdf_path: Path) -> list[PageData]:
    """Extract text from a single PDF, returning per-page data."""
    pages: list[PageData] = []
    doc_name = pdf_path.name

    try:
        doc = fitz.open(str(pdf_path))
    except fitz.FileDataError:
        logger.warning("Skipping encrypted or corrupted PDF: %s", doc_name)
        return pages

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "text": text,
                "page": page_num + 1,
                "document": doc_name,
            })

    doc.close()

    if not pages:
        logger.warning("No text extracted from: %s", doc_name)
    else:
        logger.info("Loaded %d pages from %s", len(pages), doc_name)

    return pages


def load_pdfs_from_directory(pdf_dir: Path) -> list[PageData]:
    """Load all PDFs from a directory."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    logger.info("Found %d PDF(s) in %s", len(pdf_files), pdf_dir)

    all_pages: list[PageData] = []
    for pdf_path in pdf_files:
        all_pages.extend(load_pdf(pdf_path))

    logger.info("Total pages extracted: %d", len(all_pages))
    return all_pages
