"""
Wraps Pinecone for storing and querying transcript chunk embeddings.
Uses one namespace per session (session_id) so each video's chunks
stay isolated and can be deleted independently — matches the
"no persistence across sessions" requirement.
"""
from pinecone import Pinecone


class VectorStore:
    """
    Thin wrapper around a single Pinecone index.
    Instantiate once and reuse across the app.
    """

    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def upsert_chunks(
        self,
        session_id: str,
        chunk_texts: list[str],
        chunk_vectors: list[list[float]],
        chunk_metadata: list[dict],
    ) -> int:
        """
        Store chunk vectors under a namespace = session_id.

        Args:
            session_id: unique id for this upload session (used as namespace)
            chunk_texts: original text of each chunk (stored as metadata for display)
            chunk_vectors: embedding vectors, same order/length as chunk_texts
            chunk_metadata: extra metadata per chunk, e.g. {"start_time": .., "end_time": ..}

        Returns:
            Number of vectors upserted.
        """
        if not (len(chunk_texts) == len(chunk_vectors) == len(chunk_metadata)):
            raise ValueError("chunk_texts, chunk_vectors, and chunk_metadata must be the same length")

        vectors_to_upsert = []
        for i, (text, vector, meta) in enumerate(zip(chunk_texts, chunk_vectors, chunk_metadata)):
            vectors_to_upsert.append({
                "id": f"{session_id}-chunk-{i}",
                "values": vector,
                "metadata": {**meta, "text": text},
            })

        self.index.upsert(vectors=vectors_to_upsert, namespace=session_id)
        return len(vectors_to_upsert)

    def query(self, session_id: str, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """
        Find the top_k most similar chunks within a session's namespace.

        Returns:
            List of dicts with 'text', 'start_time', 'end_time', 'score'.
        """
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=session_id,
            include_metadata=True,
        )

        matches = []
        for match in results["matches"]:
            matches.append({
                "text": match["metadata"]["text"],
                "start_time": match["metadata"].get("start_time"),
                "end_time": match["metadata"].get("end_time"),
                "score": match["score"],
            })
        return matches

    def delete_session(self, session_id: str) -> None:
        """
        Delete all vectors for a session — call this when a session ends
        or a new video is uploaded, to keep storage clean (no persistence).
        """
        self.index.delete(delete_all=True, namespace=session_id)


if __name__ == "__main__":
    # Standalone test — upserts a few fake chunks, queries them, then cleans up.
    # Usage: python -m app.services.vector_store
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.environ.get("PINECONE_INDEX_NAME", "video-qa")

    store = VectorStore(api_key=api_key, index_name=index_name)

    test_session_id = "test-session-001"

    # Fake data — normally these vectors would come from embedder.py
    fake_texts = [
        "The cat sat on the mat.",
        "Stock prices rose sharply today.",
        "Python is a popular programming language.",
    ]
    fake_vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]  # dummy vectors, just for pipe-testing
    fake_metadata = [
        {"start_time": 0.0, "end_time": 5.0},
        {"start_time": 5.0, "end_time": 10.0},
        {"start_time": 10.0, "end_time": 15.0},
    ]

    print("Upserting test chunks...")
    count = store.upsert_chunks(test_session_id, fake_texts, fake_vectors, fake_metadata)
    print(f"ok - Upserted {count} vectors under namespace '{test_session_id}'")

    print("\nQuerying with a similar dummy vector...")
    results = store.query(test_session_id, query_vector=[0.1] * 384, top_k=2)
    for r in results:
        print(f"  score={r['score']:.4f}  [{r['start_time']}s-{r['end_time']}s]  {r['text']}")

    print("\nCleaning up test session...")
    store.delete_session(test_session_id)
    print("ok - Test session deleted from Pinecone")