"""
Transcribes audio using faster-whisper (runs locally on CPU).
Built as a class so it can be swapped for a streaming STT service
in Phase 2 without touching any calling code.
"""
from dataclasses import dataclass
from faster_whisper import WhisperModel
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@dataclass
class TranscriptSegment:
    """One segment of transcribed speech with timestamps."""
    start: float       # seconds
    end: float         # seconds
    text: str


class Transcriber:
    """
    Wraps faster-whisper. Loads the model once (expensive) and reuses it
    across requests — never re-instantiate this per-request.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        print(f"Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Whisper model loaded.")

    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        """
        Transcribe an audio file and return timestamped segments.

        Args:
            audio_path: path to a .wav file

        Returns:
            List of TranscriptSegment, in chronological order.
        """
        segments, info = self.model.transcribe(audio_path, beam_size=5)

        print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

        result = []
        for seg in segments:
            result.append(TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
            ))
        return result

    def get_full_text(self, segments: list[TranscriptSegment]) -> str:
        """Join all segments into one plain-text transcript."""
        return " ".join(s.text for s in segments)


if __name__ == "__main__":
    # Standalone test — run after audio_extractor.py works.
    # Usage: python app/services/transcription.py test_output/audio.wav
    import sys

    if len(sys.argv) != 2:
        print("Usage: python transcription.py <path_to_wav>")
        sys.exit(1)

    audio_file = sys.argv[1]

    transcriber = Transcriber(model_size="small", device="cpu", compute_type="int8")
    segments = transcriber.transcribe(audio_file)

    print(f"\n✓ Transcribed {len(segments)} segments:\n")
    for s in segments:
        print(f"[{s.start:.1f}s -> {s.end:.1f}s]  {s.text}")

    print("\n--- Full transcript ---")
    print(transcriber.get_full_text(segments))