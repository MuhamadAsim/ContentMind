"""
HTTP client for the existing DevMind Embedding Service.
This does NOT load any model in this process — it calls the already-running
microservice (started manually via `uvicorn server:app --port 8001`).
"""
import httpx


class EmbeddingServiceError(Exception):
    """Raised when the embedding service is unreachable or returns an error."""
    pass


class EmbeddingService:
    """
    Client for the DevMind Embedding Service.
    Reused across the whole app — instantiate once, call many times.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def health_check(self) -> dict:
        """
        Check the embedding service is up and get its reported model + dimensions.
        Raises EmbeddingServiceError if unreachable.
        """
        try:
            response = self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise EmbeddingServiceError(
                f"Cannot reach embedding service at {self.base_url}. "
                f"Start it with: uvicorn server:app --host 127.0.0.1 --port <port>"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts. Returns one vector per input text, same order.
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        try:
            response = self.client.post(
                f"{self.base_url}/embed",
                json={"texts": texts},
            )
            response.raise_for_status()
        except httpx.ConnectError:
            raise EmbeddingServiceError(
                f"Cannot reach embedding service at {self.base_url}. "
                f"Is it running? uvicorn server:app --host 127.0.0.1 --port <port>"
            )

        data = response.json()
        return data["embeddings"]


if __name__ == "__main__":
    # Standalone test — confirms the embedding service is reachable and working.
    # Make sure your DevMind service is running first (uvicorn server:app --port 8001)
    # Usage: python -m app.services.embedder

    service = EmbeddingService(base_url="http://127.0.0.1:8001")

    print("Checking embedding service health...")
    health = service.health_check()
    print(f"✓ Service is up. Model: {health['model']}, Dimensions: {health['dimensions']}")

    print("\nTesting embed() with sample texts...")
    test_texts = ["Hello world", "This is a test transcript chunk"]
    vectors = service.embed(test_texts)

    print(f"✓ Got {len(vectors)} vectors, each with {len(vectors[0])} dimensions")
    print(f"First 5 values of vector 1: {vectors[0][:5]}")