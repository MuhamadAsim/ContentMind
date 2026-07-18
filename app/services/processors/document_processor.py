"""
DocumentProcessor — extracts text from document files and returns a
ProcessorResult. Currently supports PDF via PyMuPDF (fitz).

Designed to be extensible: the internal `_EXTENSION_HANDLERS` dict maps
file extensions to private extraction methods, so adding DOCX, TXT, or
Markdown support later means adding a method and one registry entry —
no changes to the public interface.

Supported formats (current):
    application/pdf  (.pdf)

Planned extensions (not yet implemented):
    .docx  — python-docx
    .txt / .md — direct read
"""
from pathlib import Path

from app.services.processors.base_processor import BaseProcessor, ProcessorResult


class DocumentProcessingError(Exception):
    """Raised when text extraction from a document fails or yields no content."""


class DocumentProcessor(BaseProcessor):
    """
    Extracts text from document files and returns a ProcessorResult.

    No constructor dependencies — document extraction libraries are
    pure Python and need no expensive model loading.
    """

    # Internal registry: lowercase extension → extraction method name.
    # Add new entries here when adding support for more doc types.
    _EXTENSION_HANDLERS: dict[str, str] = {
        ".pdf": "_extract_pdf",
        # Future:
        # ".docx": "_extract_docx",
        # ".txt":  "_extract_txt",
        # ".md":   "_extract_txt",
    }

    def process(self, file_path: str) -> ProcessorResult:
        """
        Dispatch to the correct extractor based on file extension, then
        return a ProcessorResult with extracted text and page count.

        Raises:
            DocumentProcessingError: if the format is unsupported,
                extraction fails, or the document yields no text.
        """
        ext = Path(file_path).suffix.lower()
        handler_name = self._EXTENSION_HANDLERS.get(ext)

        if not handler_name:
            raise DocumentProcessingError(
                f"DocumentProcessor does not support '{ext}' files. "
                f"Supported extensions: {list(self._EXTENSION_HANDLERS)}"
            )

        handler = getattr(self, handler_name)
        return handler(file_path)

    # ------------------------------------------------------------------
    # Private extraction methods — one per supported format
    # ------------------------------------------------------------------

    def _extract_pdf(self, file_path: str) -> ProcessorResult:
        """Extract text from a PDF using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise DocumentProcessingError(
                "PyMuPDF is not installed. Run: pip install pymupdf"
            ) from e

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to open PDF '{file_path}': {e}") from e

        page_count = len(doc)
        pages_text: list[str] = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages_text.append(text)

        doc.close()

        if not pages_text:
            raise DocumentProcessingError(
                f"No extractable text found in '{file_path}'. "
                "The PDF may be image-only (scanned) and requires OCR."
            )

        full_text = "\n\n".join(pages_text)

        return ProcessorResult(
            text=full_text,
            duration=None,
            page_count=page_count,
            metadata={"segments": None},  # No timestamps for documents
        )


if __name__ == "__main__":
    # Standalone test:
    # python -m app.services.processors.document_processor path/to/doc.pdf
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m app.services.processors.document_processor <path_to_file>")
        sys.exit(1)

    processor = DocumentProcessor()

    try:
        result = processor.process(sys.argv[1])
        print(f"\n✓ DocumentProcessor result:")
        print(f"  page_count : {result.page_count}")
        print(f"  text len   : {len(result.text)} chars")
        print(f"  text[:300] : {result.text[:300]}")
    except DocumentProcessingError as e:
        print(f"✗ Document processing failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
