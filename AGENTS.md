# AGENTS.md — Video Q&A Backend

Context file for AI coding agents (Claude Code, Cursor, etc.) working on this repo.
Read this before making changes — it encodes decisions already made and why,
so don't re-litigate them without asking.

## What this is

Phase 1 of a two-phase project:
- **Phase 1 (this repo, in progress):** User uploads a short video (~5 min),
  asks questions about its content. Persistent library storage, no auth,
  no conversation chat history, no saved queries.
- **Phase 2 (future, not built yet):** Same backend evolves into a live
  meeting assistant — real-time audio in, streaming transcription, continuous
  Q&A during a live call. Phase 1 is deliberately architected so Phase 2 is
  additive, not a rewrite. Don't break that separation when adding code.

## Environment

- **OS: Windows.** All shell commands and paths in docs/comments should
  assume PowerShell, not bash — this project is not developed on Linux/Mac.
- **CPU only, no GPU.** Never suggest or default to CUDA/GPU-accelerated
  options for Whisper or any other model. Everything must run acceptably
  on CPU.
- **Virtual env:** standard `venv`, activated via
  `.\venv\Scripts\Activate.ps1`. Always assume the venv is active.

## Tech stack (locked in — do not swap without explicit confirmation)

| Layer | Choice | Notes |
|---|---|---|
| Backend framework | FastAPI | async, single service |
| Audio extraction | ffmpeg (via subprocess, not python bindings doing the actual decode) | 16kHz mono WAV output |
| Transcription | `faster-whisper`, model size `small`, device `cpu`, compute_type `int8` | NOT the original `openai-whisper` package — faster-whisper is required for CPU performance |
| Persistence / Library | SQLite + SQLAlchemy | Database is `knowledge.db`. Stores metadata of processed files. Video files are stored permanently under `storage/uploads/video/` |
| Chunking | `tiktoken` (cl100k_base) + sentence/segment-aware splitter | token-based, not char-based |
| Embeddings | **External DevMind Embedding Service** — HTTP client only, called via `services/embedder.py`. Do NOT load `sentence-transformers` in this repo's process. | Model: `BAAI/bge-small-en-v1.5`, 384 dimensions, started manually by the dev on a variable port |
| Vector DB | Pinecone (cloud, serverless). Package is `pinecone` (NOT `pinecone-client` — that's the deprecated name). | Index: `video-qa`, dimension=384, metric=cosine — must match embedding model exactly |
| Q&A LLM | OpenRouter API (OpenAI-compatible SDK, different `base_url`) | Default model: `openai/gpt-4o-mini`, configurable via `DEFAULT_AI_MODEL` env var |
| Session state | Replaced by SQLite persistent metadata | No conversation history is stored; only files are persisted |
| Frontend | React (Vite), plain CSS, no state library | Two screens only: upload, chat. No routing library needed at this scope |

## Project structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI entrypoint, startup DB schema check
│   ├── config.py               # pydantic-settings, reads .env — single source of config
│   ├── database.py             # SQLite connection engine & get_db session dependency
│   ├── models/
│   │   ├── knowledge.py        # SQLAlchemy model (KnowledgeFile)
│   │   └── schemas.py          # Pydantic request/response models
│   ├── services/
│   │   ├── audio_extractor.py  # ffmpeg wrapper
│   │   ├── transcription.py    # faster-whisper wrapper (Transcriber class)
│   │   ├── chunker.py          # token-based chunking (Chunker class)
│   │   ├── embedder.py         # HTTP client to DevMind embedding service
│   │   ├── vector_store.py     # Pinecone wrapper (VectorStore class)
│   │   ├── qa_engine.py        # retrieval + OpenRouter LLM call (QAEngine class)
│   │   └── knowledge_manager.py # database CRUD operations for KnowledgeFile
│   └── routers/
│       ├── upload.py           # POST /api/upload, GET /api/videos, DELETE /api/videos/{id}
│       └── query.py            # POST /api/ask
├── requirements.txt
└── .env
```

## Conventions to follow

1. **Services are classes with dependency injection, not free functions
   calling globals.** `QAEngine`, `VectorStore`, `EmbeddingService`,
   `Transcriber`, `Chunker` all take their dependencies via constructor args.
   This is deliberate — it's what lets Phase 2 swap implementations without
   touching callers. Preserve this pattern for any new service.

2. **Expensive objects (Whisper model, embedding client, Pinecone client)
   are instantiated ONCE at module load in `routers/upload.py`, never
   per-request.** Never re-instantiate `Transcriber()` inside a route handler.

3. **Every service file has a `if __name__ == "__main__":` standalone test
   block.** When adding a new service, add one too — this project is built
   and verified bottom-up (test each piece standalone before wiring into
   the API), not top-down. Run standalone tests with
   `python -m app.services.<name>` (module syntax, not direct file path —
   internal imports depend on it).

4. **Library persistence, by design.** Uploaded files are copied permanently to
   `storage/uploads/<type>/` (e.g. `storage/uploads/video/`), and their metadata is stored in SQLite. Temp video/audio files during processing are deleted in a `finally` block after each upload. Do not cache or persist conversations or Q&A history.

5. **Config always goes through `app/config.py` (`settings` object).**
   Never read `os.environ` directly elsewhere in the app.

6. **Pinecone namespace = kb_<uuid>.** Every upsert/query/delete must be
   scoped to a unique namespace per file. Never query or write across the
   whole index without a namespace — that would leak one file's chunks into another's
   answers.

7. **Fail fast, don't fail silently.** The app has a startup check in
   `main.py` that verifies the embedding service is reachable and its
   dimension matches `PINECONE_DIMENSION` before accepting requests. Keep
   this pattern for any new external dependency — clear error at startup,
   not a cryptic failure mid-request.

## Environment variables (`.env`)

```dotenv
# Vector DB
PINECONE_API_KEY=
PINECONE_INDEX_NAME=video-qa
PINECONE_DIMENSION=384

# Embedding microservice (external, manually started — port varies)
EMBEDDING_SERVICE_URL=http://127.0.0.1:8001

# LLM via OpenRouter
OPENROUTER_API_KEY=
DEFAULT_AI_MODEL=openai/gpt-4o-mini

# Database and Storage (Defaults are set in config.py)
# UPLOAD_STORAGE_DIR=storage/uploads
# DATABASE_URL=sqlite:///./knowledge.db
```

`EMBEDDING_SERVICE_URL`'s port is **not fixed** — the developer starts the
DevMind embedding service manually (`uvicorn server:app --host 127.0.0.1
--port <port>`) and picks the port each time. Never hardcode a port number
in code; always read it from `.env`.

## Running the project (development)

Three processes must run simultaneously, each in its own terminal:

```powershell
# Terminal 1 — embedding service (separate repo, started manually by dev)
uvicorn server:app --host 127.0.0.1 --port 8001

# Terminal 2 — this backend
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload

# Terminal 3 — frontend
cd frontend
npm run dev
```

## Known constraints — don't "fix" these, they're intentional

- **CPU-only Whisper is slow.** Acceptable at ~5 min videos (a few minutes
  of processing). Don't suggest GPU acceleration. If videos grow much
  longer, the fix is async/background processing with polling — not a
  faster model or GPU, unless the user explicitly changes the hardware
  constraint.
- **Synchronous `/api/upload`.** No background job queue, no polling
  endpoint. This is a deliberate YAGNI choice for Phase 1's ~5 min video
  scope. Don't add Celery/RQ/background tasks unless asked.
- **No auth, no rate limiting, no multi-user isolation beyond file_id.**
  This is a local single-user tool, not a deployed multi-tenant service.

## Phase 2 (future) — what NOT to break

When Phase 2 (live meeting assistant) work begins, these pieces are
expected to be **replaced or extended**, not the ones below them:

**Will change:**
- `transcription.py` → gets a streaming counterpart (e.g. Deepgram/
  AssemblyAI client, or `faster-whisper` on a rolling buffer)
- `routers/upload.py` → REST upload gets a WebSocket sibling for live audio
- Chunking/embedding becomes incremental instead of one-shot batch

**Must NOT need to change:**
- `qa_engine.py` — retrieval + answer generation logic
- `vector_store.py` — Pinecone wrapper
- `embedder.py` — HTTP client to the embedding service
- Prompt construction in `qa_engine.py`

If a Phase 2 change requires modifying any of the "must not change" files,
stop and flag it — that means the Phase 1 interface boundary was drawn in
the wrong place and needs a conscious redesign discussion, not a quiet
workaround.