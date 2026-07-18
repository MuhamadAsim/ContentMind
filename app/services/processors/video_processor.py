"""
VideoProcessor — extracts audio from a video file, transcribes it with
faster-whisper, and returns a ProcessorResult with the full transcript
text and per-segment metadata (preserved for timestamp-aware chunking).

The Transcriber is injected via __init__ so the expensive Whisper model
is loaded exactly once at application startup, not per-request.
"""
import tempfile
from pathlib import Path

from app.services.audio_extractor import extract_audio, AudioExtractionError
from app.services.transcription import Transcriber
from app.services.processors.base_processor import BaseProcessor, ProcessorResult


class VideoProcessor(BaseProcessor):
    """
    Processes video files: video → WAV → transcript → ProcessorResult.

    The raw TranscriptSegment list is included in metadata["segments"]
    so the upload orchestrator can call chunker.chunk_transcript() and
    preserve per-chunk start/end timestamps in Pinecone metadata.
    """

    def __init__(self, transcriber: Transcriber):
        self._transcriber = transcriber

    def process(self, file_path: str) -> ProcessorResult:
        """
        Extract audio, transcribe, return full text + segment metadata.

        Raises:
            AudioExtractionError: if ffmpeg fails on this file.
            ValueError: if no speech is detected.
        """
        # Use a sibling temp file so we don't pollute the storage dir.
        # The upload route's finally block cleans the whole temp_dir,
        # but we clean up ourselves here too in case this is called standalone.
        temp_wav = Path(file_path).parent / (Path(file_path).stem + "_audio_tmp.wav")

        try:
            # 1. Extract audio track → 16kHz mono WAV (Whisper's expected format)
            extract_audio(file_path, str(temp_wav))

            # 2. Transcribe
            segments = self._transcriber.transcribe(str(temp_wav))
            if not segments:
                raise ValueError("No speech detected in the video.")

            full_text = self._transcriber.get_full_text(segments)
            duration = segments[-1].end if segments else None

            return ProcessorResult(
                text=full_text,
                duration=duration,
                page_count=None,
                metadata={"segments": segments},
            )

        finally:
            # Clean up the temporary WAV regardless of success or failure.
            if temp_wav.exists():
                temp_wav.unlink(missing_ok=True)


if __name__ == "__main__":
    # Standalone test:
    # python -m app.services.processors.video_processor path/to/video.mp4
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m app.services.processors.video_processor <path_to_video>")
        sys.exit(1)

    from app.config import settings

    transcriber = Transcriber(
        model_size=settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )
    processor = VideoProcessor(transcriber=transcriber)

    try:
        result = processor.process(sys.argv[1])
        print(f"\n✓ VideoProcessor result:")
        print(f"  duration   : {result.duration:.1f}s")
        print(f"  segments   : {len(result.metadata['segments'])}")
        print(f"  text[:200] : {result.text[:200]}")
    except AudioExtractionError as e:
        print(f"✗ Audio extraction failed: {e}")
    except ValueError as e:
        print(f"✗ Transcription error: {e}")
