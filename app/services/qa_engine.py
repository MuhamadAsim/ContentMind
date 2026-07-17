"""
Ties retrieval + LLM generation together: takes a question, retrieves
relevant transcript chunks from Pinecone, and asks the LLM (via OpenRouter)
to answer using only that retrieved context.
"""
from openai import OpenAI


class QAEngine:
    """
    Retrieval-augmented Q&A engine.
    Depends on an EmbeddingService (to embed the question) and a
    VectorStore (to retrieve matching chunks) — both injected in,
    so this class stays decoupled from *how* embedding/retrieval work.
    """

    def __init__(self, openrouter_api_key: str, model: str, embedding_service, vector_store):
        # OpenRouter is OpenAI-API-compatible — just point the base_url at it
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        self.model = model
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ask(self, session_id: str, question: str, top_k: int = 5) -> dict:
        """
        Answer a question using retrieved context from a specific session's
        transcript chunks.

        Returns:
            { "answer": str, "sources": [{"start_time": .., "end_time": ..}, ...] }
        """
        # 1. Embed the question using the same model as the chunks
        question_vector = self.embedding_service.embed([question])[0]

        # 2. Retrieve top matching chunks from this session's namespace
        matches = self.vector_store.query(session_id, question_vector, top_k=top_k)

        if not matches:
            return {
                "answer": "I couldn't find anything relevant in this video to answer that question.",
                "sources": [],
            }

        # 3. Build context block from retrieved chunks
        context_block = "\n\n".join(
            f"[{m['start_time']:.1f}s - {m['end_time']:.1f}s]: {m['text']}"
            for m in matches
        )

        # 4. Build the prompt — instruct the model to answer ONLY from context
        system_prompt = (
            "You are a helpful assistant answering questions about a video's transcript. "
            "Use ONLY the provided transcript excerpts to answer. "
            "If the answer isn't in the excerpts, say so clearly — do not make things up. "
            "When relevant, mention the approximate timestamp where the information was said."
        )
        user_prompt = f"Transcript excerpts:\n\n{context_block}\n\nQuestion: {question}"

        # 5. Call the LLM via OpenRouter
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # low temperature — factual retrieval-based answers, not creative
        )

        answer_text = response.choices[0].message.content

        sources = [
            {"start_time": m["start_time"], "end_time": m["end_time"]}
            for m in matches
        ]

        return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    # Standalone test — full pipeline: real audio -> transcript -> chunks ->
    # embed -> upsert -> ask a question -> get an answer.
    # Usage: python -m app.services.qa_engine test_output/audio.wav "your question here"
    import sys
    import os
    from dotenv import load_dotenv

    from app.services.transcription import Transcriber
    from app.services.chunker import Chunker
    from app.services.embedder import EmbeddingService
    from app.services.vector_store import VectorStore

    load_dotenv()

    if len(sys.argv) != 3:
        print('Usage: python -m app.services.qa_engine <path_to_wav> "<question>"')
        sys.exit(1)

    audio_file = sys.argv[1]
    question = sys.argv[2]
    test_session_id = "test-qa-session"

    print("1. Transcribing...")
    transcriber = Transcriber(model_size="small", device="cpu", compute_type="int8")
    segments = transcriber.transcribe(audio_file)

    print("2. Chunking...")
    chunker = Chunker(max_tokens=500, overlap_tokens=50)
    chunks = chunker.chunk_transcript(segments)
    print(f"   -> {len(chunks)} chunk(s)")

    print("3. Embedding chunks...")
    embedding_service = EmbeddingService(base_url=os.environ.get("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001"))
    chunk_texts = [c.text for c in chunks]
    chunk_vectors = embedding_service.embed(chunk_texts)
    chunk_metadata = [{"start_time": c.start_time, "end_time": c.end_time} for c in chunks]

    print("4. Upserting to Pinecone...")
    vector_store = VectorStore(
        api_key=os.environ["PINECONE_API_KEY"],
        index_name=os.environ.get("PINECONE_INDEX_NAME", "video-qa"),
    )
    vector_store.upsert_chunks(test_session_id, chunk_texts, chunk_vectors, chunk_metadata)

    print("5. Asking question...")
    qa_engine = QAEngine(
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ.get("DEFAULT_AI_MODEL", "openai/gpt-4o-mini"),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = qa_engine.ask(test_session_id, question)

    print(f"\n✓ Answer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")

    print("\n6. Cleaning up test session...")
    vector_store.delete_session(test_session_id)
    print("✓ Done")