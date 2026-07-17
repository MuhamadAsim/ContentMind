"""
Splits a transcript into overlapping, token-sized chunks for embedding.

Chunking is sentence/segment-aware (uses Whisper's natural segment
boundaries) instead of blind character slicing — this avoids cutting
a sentence in half mid-meaning, which hurts retrieval quality later.
"""
from dataclasses import dataclass
import tiktoken

from app.services.transcription import TranscriptSegment


@dataclass
class Chunk:
    """A chunk of transcript text ready for embedding, with timestamp range."""
    text: str
    start_time: float   # seconds — start of first segment in this chunk
    end_time: float     # seconds — end of last segment in this chunk


class Chunker:
    """
    Groups transcript segments into chunks of roughly `max_tokens`,
    with `overlap_tokens` of overlap between consecutive chunks so
    context isn't lost at chunk boundaries.
    """

    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # cl100k_base is the encoding used by GPT-3.5/4 family — good general-purpose choice
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_transcript(self, segments: list[TranscriptSegment]) -> list[Chunk]:
        """
        Combine consecutive segments into chunks up to max_tokens,
        carrying the last few segments forward as overlap into the next chunk.

        Args:
            segments: ordered list of TranscriptSegment from Transcriber

        Returns:
            List of Chunk objects, in chronological order.
        """
        if not segments:
            return []

        chunks: list[Chunk] = []
        current_segments: list[TranscriptSegment] = []
        current_tokens = 0

        for seg in segments:
            seg_tokens = self._token_count(seg.text)

            # If adding this segment would exceed max_tokens, close out the current chunk
            if current_tokens + seg_tokens > self.max_tokens and current_segments:
                chunks.append(self._build_chunk(current_segments))

                # Carry forward overlap: keep trailing segments whose combined
                # tokens are <= overlap_tokens, to preserve context continuity
                overlap_segments = []
                overlap_token_sum = 0
                for s in reversed(current_segments):
                    t = self._token_count(s.text)
                    if overlap_token_sum + t > self.overlap_tokens:
                        break
                    overlap_segments.insert(0, s)
                    overlap_token_sum += t

                current_segments = overlap_segments
                current_tokens = overlap_token_sum

            current_segments.append(seg)
            current_tokens += seg_tokens

        # Don't forget the last chunk
        if current_segments:
            chunks.append(self._build_chunk(current_segments))

        return chunks

    def _build_chunk(self, segments: list[TranscriptSegment]) -> Chunk:
        text = " ".join(s.text for s in segments)
        return Chunk(
            text=text,
            start_time=segments[0].start,
            end_time=segments[-1].end,
        )


if __name__ == "__main__":
    # Standalone test — chains audio_extractor -> transcription -> chunker
    # Usage: python app/services/chunker.py test_output/audio.wav
    import sys
    from app.services.transcription import Transcriber

    if len(sys.argv) != 2:
        print("Usage: python chunker.py <path_to_wav>")
        sys.exit(1)

    audio_file = sys.argv[1]

    transcriber = Transcriber(model_size="small", device="cpu", compute_type="int8")
    segments = transcriber.transcribe(audio_file)

    chunker = Chunker(max_tokens=500, overlap_tokens=50)
    chunks = chunker.chunk_transcript(segments)

    print(f"\n✓ Created {len(chunks)} chunks from {len(segments)} segments:\n")
    for i, c in enumerate(chunks):
        token_count = chunker._token_count(c.text)
        print(f"--- Chunk {i+1} [{c.start_time:.1f}s -> {c.end_time:.1f}s] ({token_count} tokens) ---")
        print(c.text)
        print()