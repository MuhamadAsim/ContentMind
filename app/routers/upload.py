"""
Upload router — thin orchestrator for the multi-file knowledge pipeline.

Responsibilities:
  1. Validate the upload and determine file type via ProcessorFactory
  2. Save the file permanently to storage/uploads/{type}/
  3. Create an initial 'processing' DB record
  4. Dispatch to the correct processor (video / audio / document)
  5. Route to timestamp-aware chunking (audio/video) or plain-text
     chunking (documents) based on what the processor returns
  6. Embed chunks and upsert to Pinecone
  7. Update DB record to 'ready'
  8. Clean up temp files on success or failure

All routes are now generic (/files, /files/{id}) so the library is not
video-specific. The downstream pipeline (Chunker, EmbeddingService,
VectorStore, QAEngine) never touches file-type logic.
"""
from datetime import datetime
import shutil
import uuid
import traceback
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.services.knowledge_manager import KnowledgeManager
from app.services.transcription import Transcriber
from app.services.chunker import Chunker
from app.services.embedder import EmbeddingService, EmbeddingServiceError
from app.services.vector_store import VectorStore
from app.services.processors.processor_factory import ProcessorFactory, UnsupportedFileTypeError
from app.models.schemas import KnowledgeListItem, KnowledgeDetailResponse

router = APIRouter()

# ---------------------------------------------------------------------------
# Singleton expensive objects — instantiated ONCE at module load.
# Never re-instantiate these inside a route handler.
# ---------------------------------------------------------------------------
transcriber = Transcriber(
    model_size=settings.WHISPER_MODEL_SIZE,
    device=settings.WHISPER_DEVICE,
    compute_type=settings.WHISPER_COMPUTE_TYPE,
)
chunker = Chunker(
    max_tokens=settings.CHUNK_MAX_TOKENS,
    overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
)
embedding_service = EmbeddingService(base_url=settings.EMBEDDING_SERVICE_URL)
vector_store = VectorStore(
    api_key=settings.PINECONE_API_KEY,
    index_name=settings.PINECONE_INDEX_NAME,
)


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=KnowledgeDetailResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Accept any supported file (video, audio, PDF), process it through the
    appropriate processor, chunk and embed the extracted text, store vectors
    in Pinecone, and return the completed knowledge file record.
    """
    # ------------------------------------------------------------------
    # 1. Determine file type — fail fast with a 400 if unsupported.
    #    Use both MIME type and filename for reliable detection.
    # ------------------------------------------------------------------
    mime_type = file.content_type or "application/octet-stream"
    filename = file.filename or "upload"

    try:
        file_type = ProcessorFactory.get_file_type_label(mime_type, filename)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ------------------------------------------------------------------
    # 2. Prepare storage path
    # ------------------------------------------------------------------
    file_id = str(uuid.uuid4())
    stored_filename = f"{file_id}_{filename}"

    storage_dir = Path(settings.UPLOAD_STORAGE_DIR) / file_type
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = str(storage_dir / stored_filename)

    # 3. Get file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    knowledge_manager = KnowledgeManager(db)

    # ------------------------------------------------------------------
    # 4. Create initial DB record (status = 'processing')
    # ------------------------------------------------------------------
    kb_file = knowledge_manager.create_knowledge_record(
        file_id=file_id,
        title=Path(filename).stem,
        file_type=file_type,
        original_filename=filename,
        stored_filename=stored_filename,
        storage_path=storage_path,
        file_size=file_size,
        mime_type=mime_type,
        pinecone_namespace=f"kb_{file_id}",
    )

    # ------------------------------------------------------------------
    # 5. Save file permanently
    # ------------------------------------------------------------------
    try:
        with open(storage_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        knowledge_manager.update_status(file_id, status="failed")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # ------------------------------------------------------------------
    # 6. Process → Chunk → Embed → Upsert (with error handling + cleanup)
    # ------------------------------------------------------------------
    try:
        # Resolve and run the processor
        processor = ProcessorFactory.get_processor(mime_type, filename, transcriber=transcriber)
        result = processor.process(storage_path)

        if not result.text.strip():
            raise ValueError("No extractable text found in the uploaded file.")

        # Route to timestamp-aware or plain-text chunking based on
        # whether the processor returned transcript segments.
        segments = result.metadata.get("segments")
        if segments:
            # Audio/video: preserve per-chunk timestamps for source references
            chunks = chunker.chunk_transcript(segments)
            chunk_metadata = [
                {"start_time": c.start_time, "end_time": c.end_time}
                for c in chunks
            ]
        else:
            # Documents: no timestamps; start_time=0.0, end_time=0.0
            chunks = chunker.chunk_text(result.text)
            chunk_metadata = [
                {"start_time": c.start_time, "end_time": c.end_time}
                for c in chunks
            ]

        if not chunks:
            raise ValueError("File produced no chunks after processing.")

        # Embed
        chunk_texts = [c.text for c in chunks]
        chunk_vectors = embedding_service.embed(chunk_texts)

        # Upsert to Pinecone under this file's isolated namespace
        vector_store.upsert_chunks(
            kb_file.pinecone_namespace, chunk_texts, chunk_vectors, chunk_metadata
        )

        # Mark ready; persist type-specific metadata
        kb_file = knowledge_manager.update_status(
            file_id=file_id,
            status="ready",
            duration=result.duration,
            page_count=result.page_count,
            processed_at=datetime.utcnow(),
        )

        return kb_file

    except UnsupportedFileTypeError as e:
        print(f"Unsupported file type: {e}")
        knowledge_manager.update_status(file_id, status="failed")
        _cleanup_pinecone(vector_store, kb_file.pinecone_namespace)
        raise HTTPException(status_code=400, detail=str(e))

    except EmbeddingServiceError as e:
        print(f"Embedding service failed: {e}")
        traceback.print_exc()
        knowledge_manager.update_status(file_id, status="failed")
        _cleanup_pinecone(vector_store, kb_file.pinecone_namespace)
        raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {e}")

    except Exception as e:
        print(f"File processing failed: {e}")
        traceback.print_exc()
        knowledge_manager.update_status(file_id, status="failed")
        _cleanup_pinecone(vector_store, kb_file.pinecone_namespace)
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")


def _cleanup_pinecone(vs: VectorStore, namespace: str) -> None:
    """Best-effort Pinecone cleanup — swallows errors so they don't mask the original."""
    try:
        vs.delete_session(namespace)
    except Exception as cleanup_err:
        print(f"Warning: failed to clean up Pinecone namespace '{namespace}': {cleanup_err}")


# ---------------------------------------------------------------------------
# GET /api/files — list all knowledge files (all types)
# ---------------------------------------------------------------------------

@router.get("/files", response_model=list[KnowledgeListItem])
async def list_files(db: Session = Depends(get_db)):
    """List all knowledge files in the library, regardless of type."""
    knowledge_manager = KnowledgeManager(db)
    files = knowledge_manager.list_knowledge_files()  # no type filter
    return [
        KnowledgeListItem(
            id=f.id,
            title=f.title,
            type=f.type,
            status=f.status,
            created_at=f.created_at,
        )
        for f in files
    ]


# ---------------------------------------------------------------------------
# GET /api/files/{file_id} — get detail for any knowledge file
# ---------------------------------------------------------------------------

@router.get("/files/{file_id}", response_model=KnowledgeDetailResponse)
async def get_file_detail(file_id: str, db: Session = Depends(get_db)):
    """Retrieve full metadata for any knowledge file by ID."""
    knowledge_manager = KnowledgeManager(db)
    f = knowledge_manager.get_knowledge_file(file_id)
    if not f:
        raise HTTPException(status_code=404, detail="File not found.")
    return f


# ---------------------------------------------------------------------------
# DELETE /api/files/{file_id} — delete any knowledge file
# ---------------------------------------------------------------------------

@router.delete("/files/{file_id}")
async def delete_file(file_id: str, db: Session = Depends(get_db)):
    """Delete a knowledge file's DB record, physical file, and Pinecone vectors."""
    knowledge_manager = KnowledgeManager(db)
    f = knowledge_manager.get_knowledge_file(file_id)
    if not f:
        raise HTTPException(status_code=404, detail="File not found.")

    # 1. Delete Pinecone vectors
    _cleanup_pinecone(vector_store, f.pinecone_namespace)

    # 2. Delete physical file from disk
    if f.storage_path:
        file_path = Path(f.storage_path)
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Warning: failed to delete file {f.storage_path}: {e}")

    # 3. Delete DB record
    knowledge_manager.delete_knowledge_record(file_id)

    return {"status": "success", "message": f"File {file_id} deleted successfully."}