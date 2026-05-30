"""
utils/file_parser.py
--------------------
Extract plain text from uploaded files for compliance graph input.
Supports: PDF (pymupdf), DOCX (python-docx), TXT, MD.
"""

from pathlib import Path

MAX_CHARS = 50_000
_TRUNCATION_NOTE = "\n\n[NOTE: File truncated at 50,000 characters for compliance processing.]"


def supported_extensions() -> list[str]:
    return [".pdf", ".docx", ".txt", ".md"]


def extract_text(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Extract plain text from an uploaded file.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to detect format by extension).

    Returns:
        (extracted_text, file_type) where file_type is one of:
        "pdf", "docx", "txt", "md"

    Raises:
        ValueError: if the file extension is not supported.
        RuntimeError: if extraction fails (e.g. encrypted PDF, corrupt file).
    """
    suffix = Path(filename).suffix.lower()

    if suffix not in supported_extensions():
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported: .pdf, .docx, .txt, .md"
        )

    try:
        if suffix == ".pdf":
            text = _extract_pdf(file_bytes)
            file_type = "pdf"
        elif suffix == ".docx":
            text = _extract_docx(file_bytes)
            file_type = "docx"
        else:
            text = file_bytes.decode("utf-8", errors="replace")
            file_type = suffix.lstrip(".")
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}") from e

    if not text.strip():
        raise RuntimeError("File appears to be empty or contains no extractable text.")

    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + _TRUNCATION_NOTE

    return text, file_type


def _extract_pdf(file_bytes: bytes) -> str:
    import io
    import fitz  # pymupdf

    try:
        doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}") from e

    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()

    text = "\n\n".join(pages_text)
    if not text.strip():
        raise RuntimeError(
            "PDF appears to be a scanned image. Text extraction requires a text-layer PDF."
        )
    return text


def _extract_docx(file_bytes: bytes) -> str:
    import io
    from docx import Document

    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as e:
        raise RuntimeError(f"Could not open DOCX: {e}") from e

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
