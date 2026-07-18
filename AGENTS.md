# AGENTS.md — Knowledge Library Backend

Context file for AI coding agents (Claude Code, Cursor, etc.) working on this repo.
Read this before making changes — it encodes decisions already made and why,
so don't re-litigate them without asking.

## What this is

Phase 1 of a two-phase project:
- **Phase 1 (this repo, in progress):** User uploads a supported file (video,
  audio, or PDF), asks questions about its content. Persistent library storage,
  no auth, no conversation chat history, no saved queries.
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
| Audio extraction | ffmpeg (via subprocess) | 16kHz mono WAV output; used only for video→audio step |
| Transcription | `faster-whisper`, model size `small`, device `cpu`, compute_type `int8` | NOT the original `openai-whisper` package |
| Persistence / Library | SQLite + SQLAlchemy | Database is `knowledge.db`. Stores metadata of all file types. Files stored under `storage/uploads/{type}/` |
| PDF extraction | `pymupdf` (`import fitz`) | Pure CPU, no dependencies; extensible to other doc types via `DocumentProcessor._EXTENSION_HANDLERS` |
| Chunking | `tiktoken` (cl100k_base) + segment-aware or sentence-split | `chunk_transcript()` for audio/video (timestamps preserved); `chunk_text()` for documents |
| Embeddings | **External DevMind Embedding Service** — HTTP client only, called via `services/embedder.py`. Do NOT load `sentence-transformers` in this repo's process. | Model: `BAAI/bge-small-en-v1.5`, 384 dimensions, started manually by the dev on a variable port |
| Vector DB | Pinecone (cloud, serverless). Package is `pinecone` (NOT `pinecone-client`). | Index: `video-qa`, dimension=384, metric=cosine |
| Q&A LLM | OpenRouter API (OpenAI-compatible SDK, different `base_url`) | Default model: `openai/gpt-4o-mini`, configurable via `DEFAULT_AI_MODEL` env var |
| Text-to-Speech (TTS) | Piper TTS (via `piper-tts` package) | Uses local ONNX model `en_US-lessac-medium` stored at `app/resources/voices/` |
| Frontend | React (Vite), plain CSS, no state library | Upload box accepts video, audio, PDF |

## Supported file types

| Category | Formats | Processor | DB `type` |
|---|---|---|---|
| Video | MP4, MOV, AVI, MKV, WebM, FLV, WMV, 3GP, OGV | `VideoProcessor` | `"video"` |
| Audio | MP3, WAV, M4A, FLAC, OGG, AAC, WMA, Opus | `AudioProcessor` | `"audio"` |
| Document | PDF | `DocumentProcessor` | `"document"` |

To add a new format, register it in `processor_factory.py` (for a new processor class)
or add an entry to `DocumentProcessor._EXTENSION_HANDLERS` (for new document types
handled by the same extraction pattern).

## Upload pipeline

```
POST /api/upload
      │
      ▼
ProcessorFactory.get_file_type_label(mime, filename)   ← fails fast with 400 if unsupported
      │
      ▼
Save file to storage/uploads/{type}/{uuid}_{filename}
Create DB record (status="processing")
      │
      ▼
ProcessorFactory.get_processor(mime, filename, transcriber)
      │
      ▼
processor.process(file_path)  →  ProcessorResult(text, duration, page_count, metadata)
      │
      ├── metadata["segments"] present?
      │       YES → chunker.chunk_transcript(segments)   # audio/video: timestamps preserved
      │       NO  → chunker.chunk_text(text)             # documents: start/end = 0.0
      │
      ▼
embedding_service.embed(chunk_texts)
      │
      ▼
vector_store.upsert_chunks(namespace=kb_{uuid}, ...)
      │
      ▼
Update DB record (status="ready", duration/page_count set)
```

On any failure: DB → `"failed"`, Pinecone vectors cleaned up, file preserved on disk.

## Project structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI entrypoint; startup DB schema check + migration
│   ├── config.py               # pydantic-settings, reads .env — single source of config
│   ├── database.py             # SQLite connection engine & get_db session dependency
│   ├── models/
│   │   ├── knowledge.py        # SQLAlchemy model (KnowledgeFile) — includes page_count
│   │   └── schemas.py          # Pydantic request/response models
│   ├── services/
│   │   ├── audio_extractor.py  # ffmpeg wrapper (video→WAV only)
│   │   ├── transcription.py    # faster-whisper wrapper (Transcriber class)
│   │   ├── chunker.py          # token-based chunking; chunk_transcript() + chunk_text()
│   │   ├── embedder.py         # HTTP client to DevMind embedding service
│   │   ├── vector_store.py     # Pinecone wrapper (VectorStore class)
│   │   ├── qa_engine.py        # retrieval + OpenRouter LLM call (QAEngine class)
│   │   ├── knowledge_manager.py # database CRUD operations for KnowledgeFile
│   │   └── processors/
│   │       ├── __init__.py
│   │       ├── base_processor.py      # ProcessorResult dataclass + BaseProcessor ABC
│   │       ├── video_processor.py     # audio extract → transcribe → ProcessorResult
│   │       ├── audio_processor.py     # direct transcribe → ProcessorResult
│   │       ├── document_processor.py  # PDF text extraction → ProcessorResult (extensible)
│   │       └── processor_factory.py   # registry-based MIME/extension dispatch
│   └── routers/
│       ├── upload.py           # POST /api/upload, GET /api/files, DELETE /api/files/{id}
│       └── query.py            # POST /api/ask (works for all file types)
├── requirements.txt
└── .env
```

## Conventions to follow

1. **Services are classes with dependency injection, not free functions
   calling globals.** `QAEngine`, `VectorStore`, `EmbeddingService`,
   `Transcriber`, `Chunker`, all processors — take their dependencies via
   constructor args. Preserve this pattern for any new service.

2. **Expensive objects (Whisper model, embedding client, Pinecone client)
   are instantiated ONCE at module load in `routers/upload.py`, never
   per-request.** Never re-instantiate `Transcriber()` inside a route handler.
   The `Transcriber` singleton is shared between `VideoProcessor` and
   `AudioProcessor` via the factory.

3. **Every service file has a `if __name__ == "__main__":` standalone test
   block.** Run with `python -m app.services.<name>` (module syntax, not
   direct file path — internal imports depend on it). Processor standalone
   tests: `python -m app.services.processors.<name>`.

4. **Library persistence, by design.** Uploaded files are copied permanently
   to `storage/uploads/<type>/`. Temp audio files (WAV extracted from video)
   are cleaned up by the `VideoProcessor.process()` method. Do not cache or
   persist conversations or Q&A history.

5. **Config always goes through `app/config.py` (`settings` object).**
   Never read `os.environ` directly elsewhere in the app.

6. **Pinecone namespace = `kb_<uuid>`.** Every upsert/query/delete must be
   scoped to a unique namespace per file. Never query or write across the
   whole index without a namespace.

7. **Fail fast, don't fail silently.** Startup check in `main.py` verifies
   the embedding service is reachable and dimension-compatible. Keep this
   pattern for any new external dependency.

8. **Schema migrations are explicit, not implicit.** `Base.metadata.create_all()`
   only creates missing tables — it does NOT add columns to existing tables.
   New columns must be added via `ALTER TABLE` in `_run_migrations()` in
   `main.py`. Never assume `create_all()` performs schema migration.

9. **Processor registration is centralized in `processor_factory.py`.**
   Adding a new file type means: create a `BaseProcessor` subclass, then
   call `ProcessorFactory.register(mime_types=[...], extensions=[...], ...)`.
   Detection order: MIME exact match → MIME prefix (e.g. `video/*`) →
   file extension fallback. Do not scatter file-type detection elsewhere.

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
DevMind embedding service manually and picks the port each time. Never
hardcode a port number in code; always read it from `.env`.

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

## API routes

| Method | Route | Description |
|---|---|---|
| POST | `/api/upload` | Upload any supported file; returns `KnowledgeDetailResponse` |
| GET | `/api/files` | List all knowledge files (all types) |
| GET | `/api/files/{id}` | Get detail for any knowledge file |
| DELETE | `/api/files/{id}` | Delete file, Pinecone vectors, and DB record |
| POST | `/api/ask` | Ask a question against any `ready` file (accepts text or voice recording via multipart form) |

## Known constraints — don't "fix" these, they're intentional

- **CPU-only Whisper is slow.** Acceptable at ~5 min videos/audio. Don't
  suggest GPU acceleration. For longer content, the fix is async background
  processing with polling — not a faster model or GPU.
- **Synchronous `/api/upload`.** No background job queue. Deliberate YAGNI
  choice for Phase 1's short-content scope. Don't add Celery/RQ unless asked.
- **No auth, no rate limiting, no multi-user isolation beyond file_id.**
  Local single-user tool, not a multi-tenant service.

## Phase 2 (future) — what NOT to break

**Will change:**
- `transcription.py` → gets a streaming counterpart
- `routers/upload.py` → REST upload gets a WebSocket sibling for live audio
- Chunking/embedding becomes incremental instead of one-shot batch

**Must NOT need to change:**
- `qa_engine.py` — retrieval + answer generation logic
- `vector_store.py` — Pinecone wrapper
- `embedder.py` — HTTP client to the embedding service
- All processors in `services/processors/` — they are Phase 1 only
- Prompt construction in `qa_engine.py`

If a Phase 2 change requires modifying any "must not change" files, stop and
flag it — that means the Phase 1 interface boundary was drawn in the wrong
place and needs a conscious redesign discussion.