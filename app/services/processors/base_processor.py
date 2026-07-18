"""
Base interface for all file processors.

Every processor converts a file on disk into a ProcessorResult —
a common structure containing the extracted text and any type-specific
metadata. The rest of the pipeline (chunker → embedder → Pinecone)
only ever sees a ProcessorResult; it has no knowledge of the source
file type.

To add a new processor:
1. Subclass BaseProcessor and implement `process()`.
2. Return a ProcessorResult — all fields except `text` are optional.
3. Register it in processor_factory.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProcessorResult:
    """
    Uniform output from every processor. The upload orchestrator reads
    this and routes downstream: segments → chunk_transcript() for
    audio/video (preserves timestamps), or text → chunk_text() for
    documents.
    """
    # Always present — the full extracted text from the file.
    text: str

    # Seconds of audio/video content. None for documents.
    duration: float | None = None

    # Number of pages. None for audio/video.
    page_count: int | None = None

    # Processor-specific extras. Audio/video processors put
    # `segments: list[TranscriptSegment]` here so the orchestrator
    # can call chunk_transcript() and preserve per-chunk timestamps.
    metadata: dict = field(default_factory=dict)


class BaseProcessor(ABC):
    """
    Abstract base for all file processors. Each subclass handles one
    family of file types and returns a ProcessorResult.

    Processors are pure converters — they must NOT perform chunking,
    embedding, or Pinecone operations. They also must NOT raise
    HTTPExceptions; raise domain-specific errors that the upload route
    translates into appropriate HTTP responses.
    """

    @abstractmethod
    def process(self, file_path: str) -> ProcessorResult:
        """
        Convert a file at `file_path` into a ProcessorResult.

        Args:
            file_path: Absolute or relative path to the file on disk.

        Returns:
            ProcessorResult with at minimum a non-empty `text` field.

        Raises:
            Any domain-specific exception (e.g. AudioExtractionError,
            DocumentProcessingError) — never HTTPException.
        """
        ...


if __name__ == "__main__":
    # Sanity-check the dataclass defaults.
    result = ProcessorResult(text="Hello world.")
    assert result.text == "Hello world."
    assert result.duration is None
    assert result.page_count is None
    assert result.metadata == {}
    print("ok - ProcessorResult defaults are correct.")
