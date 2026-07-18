"""
AudioProcessor — transcribes audio files directly with faster-whisper,
skipping the audio-extraction step needed for video.

faster-whisper calls ffmpeg internally and handles .mp3, .wav, .m4a,
.flac, .ogg, and .aac without needing an explicit conversion step.

Supported MIME types:
    audio/mpeg          (.mp3)
    audio/wav           (.wav)
    audio/x-wav         (.wav)
    audio/mp4           (.m4a / .mp4 audio-only)
    audio/x-m4a         (.m4a)
    audio/flac          (.flac)
    audio/ogg           (.ogg)
    audio/aac           (.aac)
    audio/x-aac         (.aac)
"""
from app.services.transcription import Transcriber
from app.services.processors.base_processor import BaseProcessor, ProcessorResult


class AudioProcessor(BaseProcessor):
    """
    Processes audio files: audio → transcript → ProcessorResult.

    Identical to VideoProcessor except there is no audio-extraction step.
    The Transcriber is injected via __init__ (shared singleton with VideoProcessor).
    """

    def __init__(self, transcriber: Transcriber):
        self._transcriber = transcriber

    def process(self, file_path: str) -> ProcessorResult:
        """
        Transcribe an audio file and return full text + segment metadata.

        Raises:
            ValueError: if no speech is detected.
            RuntimeError: if faster-whisper / ffmpeg fails on the file format.
        """
        segments = self._transcriber.transcribe(file_path)
        if not segments:
            raise ValueError("No speech detected in the audio file.")

        full_text = self._transcriber.get_full_text(segments)
        duration = segments[-1].end if segments else None

        return ProcessorResult(
            text=full_text,
            duration=duration,
            page_count=None,
            metadata={"segments": segments},
        )


if __name__ == "__main__":
    # Standalone test:
    # python -m app.services.processors.audio_processor path/to/audio.mp3
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m app.services.processors.audio_processor <path_to_audio>")
        sys.exit(1)

    from app.config import settings

    transcriber = Transcriber(
        model_size=settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )
    processor = AudioProcessor(transcriber=transcriber)

    try:
        result = processor.process(sys.argv[1])
        print(f"\n✓ AudioProcessor result:")
        print(f"  duration   : {result.duration:.1f}s")
        print(f"  segments   : {len(result.metadata['segments'])}")
        print(f"  text[:200] : {result.text[:200]}")
    except ValueError as e:
        print(f"✗ Transcription error: {e}")
    except Exception as e:
        print(f"✗ Failed: {e}")
