"""
FastAPI entrypoint. Verifies the embedding service is reachable and
dimension-compatible with Pinecone before accepting any requests.
"""
import sys
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base
from app.models.knowledge import KnowledgeFile
from app.routers import upload, query
from pathlib import Path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database and ensure upload directories exist
    Base.metadata.create_all(bind=engine)
    Path(settings.UPLOAD_STORAGE_DIR).mkdir(parents=True, exist_ok=True)

    # Startup check — fail fast with a clear message instead of a
    # confusing error later mid-upload.
    try:
        response = httpx.get(f"{settings.EMBEDDING_SERVICE_URL}/health", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Embedding service connected: {data['model']} ({data['dimensions']}-dim)")

        if data["dimensions"] != settings.PINECONE_DIMENSION:
            print(
                f"✗ FATAL: Embedding service outputs {data['dimensions']}-dim vectors, "
                f"but PINECONE_DIMENSION is set to {settings.PINECONE_DIMENSION}. "
                f"Fix your .env or Pinecone index config."
            )
            sys.exit(1)

    except httpx.ConnectError:
        print(
            f"✗ FATAL: Cannot reach embedding service at {settings.EMBEDDING_SERVICE_URL}. "
            f"Start it first: uvicorn server:app --host 127.0.0.1 --port <port>"
        )
        sys.exit(1)

    yield  # app runs here


app = FastAPI(title="Video Q&A Service", lifespan=lifespan)

# Allow the frontend (running on a different port, e.g. localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(query.router, prefix="/api", tags=["query"])


@app.get("/")
def root():
    return {"status": "running", "service": "Video Q&A Service"}