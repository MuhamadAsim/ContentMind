"""
ProcessorFactory — registry-based dispatch to the correct processor.

Detection strategy (in order of preference):
  1. MIME type lookup (most reliable when the client sends a correct type)
  2. MIME type prefix match (e.g. "video/*" → VideoProcessor)
  3. File extension fallback (when MIME type is generic or absent)

Registering a new processor:
  Call ProcessorFactory.register() with the MIME types and/or extensions
  it handles, then pass the processor class. No changes to this module's
  core dispatch logic are required.

Example — adding a DOCX processor in the future:
  ProcessorFactory.register(
      mime_types=["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
      extensions=[".docx"],
      file_type_label="document",
  )(DocxProcessor)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.services.processors.base_processor import BaseProcessor

if TYPE_CHECKING:
    from app.services.transcription import Transcriber


class UnsupportedFileTypeError(Exception):
    """Raised when no registered processor handles the given file type."""


class _ProcessorEntry:
    """Internal record stored in the registry."""
    __slots__ = ("processor_class", "file_type_label", "needs_transcriber")

    def __init__(self, processor_class: type, file_type_label: str, needs_transcriber: bool):
        self.processor_class = processor_class
        self.file_type_label = file_type_label
        self.needs_transcriber = needs_transcriber


class ProcessorFactory:
    """
    Central registry that maps MIME types and file extensions to processor
    classes. All lookup logic lives here so no other part of the app needs
    to know about file-type detection.
    """

    # Exact MIME type → entry  (e.g. "audio/mpeg")
    _mime_registry: dict[str, _ProcessorEntry] = {}

    # MIME prefix → entry  (e.g. "video" for any "video/*")
    _mime_prefix_registry: dict[str, _ProcessorEntry] = {}

    # Lowercase file extension → entry  (e.g. ".pdf")
    _extension_registry: dict[str, _ProcessorEntry] = {}

    @classmethod
    def register(
        cls,
        mime_types: list[str] | None = None,
        mime_prefixes: list[str] | None = None,
        extensions: list[str] | None = None,
        file_type_label: str = "unknown",
        needs_transcriber: bool = False,
    ):
        """
        Decorator that registers a processor class for given MIME types,
        MIME prefixes, and/or file extensions.

        Args:
            mime_types:       Exact MIME strings, e.g. ["application/pdf"]
            mime_prefixes:    MIME category prefixes, e.g. ["video", "audio"]
            extensions:       Lowercase dotted extensions, e.g. [".mp3", ".wav"]
            file_type_label:  Short label stored in the database ("video", "audio", "document")
            needs_transcriber: True if the processor requires a Transcriber instance.
        """
        def decorator(processor_class: type) -> type:
            entry = _ProcessorEntry(
                processor_class=processor_class,
                file_type_label=file_type_label,
                needs_transcriber=needs_transcriber,
            )
            for mime_type in (mime_types or []):
                cls._mime_registry[mime_type.lower()] = entry
            for prefix in (mime_prefixes or []):
                cls._mime_prefix_registry[prefix.lower()] = entry
            for ext in (extensions or []):
                cls._extension_registry[ext.lower()] = entry
            return processor_class

        return decorator

    @classmethod
    def get_entry(cls, mime_type: str, filename: str) -> _ProcessorEntry:
        """
        Find the registry entry for a file, using the detection strategy
        (MIME exact → MIME prefix → extension).

        Raises:
            UnsupportedFileTypeError: if no match is found.
        """
        mime_lower = (mime_type or "").lower().strip()

        # 1. Exact MIME type match
        if mime_lower in cls._mime_registry:
            return cls._mime_registry[mime_lower]

        # 2. MIME prefix match  (e.g. "video/x-matroska" → prefix "video")
        if "/" in mime_lower:
            prefix = mime_lower.split("/")[0]
            if prefix in cls._mime_prefix_registry:
                return cls._mime_prefix_registry[prefix]

        # 3. File extension fallback (handles "application/octet-stream" etc.)
        ext = Path(filename).suffix.lower()
        if ext in cls._extension_registry:
            return cls._extension_registry[ext]

        raise UnsupportedFileTypeError(
            f"Unsupported file type: mime='{mime_type}', filename='{filename}'. "
            f"Supported MIME types: {sorted(cls._mime_registry)} | "
            f"Supported extensions: {sorted(cls._extension_registry)}"
        )

    @classmethod
    def get_processor(
        cls,
        mime_type: str,
        filename: str,
        transcriber: "Transcriber | None" = None,
    ) -> BaseProcessor:
        """
        Resolve and instantiate the correct processor.

        Args:
            mime_type:   MIME type string from the upload (may be unreliable).
            filename:    Original filename, used as fallback for extension lookup.
            transcriber: Required for audio/video processors; ignored for documents.

        Returns:
            An instantiated BaseProcessor subclass.

        Raises:
            UnsupportedFileTypeError: if no processor matches.
            ValueError: if a transcriber-requiring processor is requested but
                        no transcriber was provided.
        """
        entry = cls.get_entry(mime_type, filename)

        if entry.needs_transcriber:
            if transcriber is None:
                raise ValueError(
                    f"{entry.processor_class.__name__} requires a Transcriber instance, "
                    "but none was provided."
                )
            return entry.processor_class(transcriber=transcriber)

        return entry.processor_class()

    @classmethod
    def get_file_type_label(cls, mime_type: str, filename: str) -> str:
        """
        Return the short label for the database `type` column.
        Example: "video", "audio", "document".

        Raises:
            UnsupportedFileTypeError: if the type is unrecognised.
        """
        return cls.get_entry(mime_type, filename).file_type_label


# ---------------------------------------------------------------------------
# Built-in registrations — loaded when this module is imported.
# ---------------------------------------------------------------------------
# Import processors here (after class definition) to avoid circular imports.
from app.services.processors.video_processor import VideoProcessor       # noqa: E402
from app.services.processors.audio_processor import AudioProcessor       # noqa: E402
from app.services.processors.document_processor import DocumentProcessor  # noqa: E402

# Video
ProcessorFactory.register(
    mime_types=[
        "video/mp4",
        "video/mpeg",
        "video/quicktime",
        "video/x-msvideo",        # .avi
        "video/x-matroska",       # .mkv
        "video/webm",
        "video/x-flv",
        "video/3gpp",
        "video/3gpp2",
        "video/ogg",
    ],
    mime_prefixes=["video"],
    extensions=[".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".3gp", ".ogv"],
    file_type_label="video",
    needs_transcriber=True,
)(VideoProcessor)

# Audio
ProcessorFactory.register(
    mime_types=[
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mp4",
        "audio/x-m4a",
        "audio/m4a",
        "audio/flac",
        "audio/x-flac",
        "audio/ogg",
        "audio/aac",
        "audio/x-aac",
        "audio/webm",
    ],
    mime_prefixes=["audio"],
    extensions=[".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"],
    file_type_label="audio",
    needs_transcriber=True,
)(AudioProcessor)

# Document (PDF — extensible via DocumentProcessor's own internal registry)
ProcessorFactory.register(
    mime_types=["application/pdf"],
    extensions=[".pdf"],
    file_type_label="document",
    needs_transcriber=False,
)(DocumentProcessor)


if __name__ == "__main__":
    # Smoke-test the factory dispatch logic without running any real processors.
    tests = [
        ("video/mp4",         "lecture.mp4",   "video"),
        ("video/quicktime",   "clip.mov",      "video"),
        ("video/x-matroska",  "movie.mkv",     "video"),
        ("audio/mpeg",        "podcast.mp3",   "audio"),
        ("audio/wav",         "recording.wav", "audio"),
        ("audio/x-m4a",       "voice.m4a",     "audio"),
        ("audio/flac",        "track.flac",    "audio"),
        ("application/pdf",   "report.pdf",    "document"),
        # Extension fallback cases (generic MIME)
        ("application/octet-stream", "file.mp4",  "video"),
        ("application/octet-stream", "audio.mp3", "audio"),
        ("application/octet-stream", "doc.pdf",   "document"),
    ]

    print("Testing ProcessorFactory dispatch...\n")
    all_passed = True
    for mime, fname, expected_label in tests:
        try:
            label = ProcessorFactory.get_file_type_label(mime, fname)
            status = "ok" if label == expected_label else f"FAIL (got '{label}')"
            if label != expected_label:
                all_passed = False
        except UnsupportedFileTypeError as e:
            status = f"FAIL — UnsupportedFileTypeError: {e}"
            all_passed = False
        print(f"  [{status}] mime='{mime}' file='{fname}' -> label='{expected_label}'")

    # Test unsupported type raises correctly
    try:
        ProcessorFactory.get_file_type_label("text/plain", "notes.txt")
        print("\n  [FAIL] Expected UnsupportedFileTypeError for text/plain but got none")
        all_passed = False
    except UnsupportedFileTypeError:
        print("\n  [ok] UnsupportedFileTypeError raised correctly for text/plain")

    print(f"\n{'All tests passed.' if all_passed else 'SOME TESTS FAILED.'}")
