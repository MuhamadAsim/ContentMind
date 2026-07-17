"""
Extracts audio from a video file using ffmpeg.
Output: 16kHz mono WAV — Whisper's preferred input format.
"""
import subprocess
from pathlib import Path


class AudioExtractionError(Exception):
    """Raised when ffmpeg fails to extract audio."""
    pass


def extract_audio(video_path: str, output_path: str) -> str:
    """
    Extract audio from a video file and save as 16kHz mono WAV.

    Args:
        video_path: path to the input video file
        output_path: path where the .wav file should be saved

    Returns:
        The output_path, on success.

    Raises:
        AudioExtractionError: if ffmpeg fails or the video has no audio track.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-i", video_path,        # input file
        "-vn",                   # strip video stream, audio only
        "-acodec", "pcm_s16le",  # standard WAV codec
        "-ar", "16000",          # 16kHz sample rate (Whisper's expected rate)
        "-ac", "1",              # mono (1 channel)
        "-y",                    # overwrite output if it exists
        output_path,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg failed to extract audio from {video_path}:\n{result.stderr}"
        )

    if not Path(output_path).exists():
        raise AudioExtractionError(f"ffmpeg ran but no output file was created at {output_path}")

    return output_path


if __name__ == "__main__":
    # Standalone test — run this file directly to test extraction
    # before wiring it into the API.
    #
    # Usage: python app/services/audio_extractor.py path/to/video.mp4
    import sys

    if len(sys.argv) != 2:
        print("Usage: python audio_extractor.py <path_to_video>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_wav = "test_output/audio.wav"

    try:
        result_path = extract_audio(input_video, output_wav)
        print(f"✓ Audio extracted successfully: {result_path}")
    except AudioExtractionError as e:
        print(f"✗ Extraction failed: {e}")