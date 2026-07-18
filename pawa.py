"""
Standalone transcription script — no app/API wiring needed.

Usage:
    python transcribe_to_file.py <path_to_audio.wav> [output.txt]

Example:
    python transcribe_to_file.py test_output/audio.wav
    python transcribe_to_file.py test_output/audio.wav my_transcript.txt

Requires:
    pip install faster-whisper
"""
import sys
import os
from dataclasses import dataclass
from faster_whisper import WhisperModel

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[TranscriptSegment]:
    """
    Load Whisper model and transcribe the given audio file.

    Raises:
        TranscriptionError: if the file is missing or ffmpeg/whisper fails.
    """
    if not os.path.exists(audio_path):
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    print(f"Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("Model loaded. Transcribing...")

    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
    except Exception as e:
        raise TranscriptionError(f"Transcription failed for {audio_path}: {e}") from e

    print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

    result = [
        TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
        for seg in segments
    ]

    if not result:
        print("Warning: no speech detected in this audio.")

    return result


def save_transcript(segments: list[TranscriptSegment], output_path: str) -> None:
    """
    Write transcript to a .txt file: timestamps + segments, then full plain text.
    """
    full_text = " ".join(s.text for s in segments)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== TIMESTAMPED SEGMENTS ===\n\n")
        for s in segments:
            f.write(f"[{s.start:.1f}s -> {s.end:.1f}s]  {s.text}\n")

        f.write("\n=== FULL TRANSCRIPT ===\n\n")
        f.write(full_text)

    print(f"\n✓ Transcript saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_to_file.py <path_to_audio> [output.txt]")
        sys.exit(1)

    audio_file = sys.argv[1]
    # If no output path given, reuse the audio filename with .txt extension
    output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(audio_file)[0] + ".txt"

    try:
        segments = transcribe_audio(audio_file)
        save_transcript(segments, output_file)
    except TranscriptionError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)