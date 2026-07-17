"""
API endpoints for uploading, listing, fetching, and deleting video files.
Coordinates between the DB service (KnowledgeManager), local file storage,
and the AI processing pipeline (extraction -> transcription -> chunking -> embedding -> Pinecone).
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
from app.services.audio_extractor import extract_audio, AudioExtractionError
from app.services.transcription import Transcriber
from app.services.chunker import Chunker
from app.services.embedder import EmbeddingService, EmbeddingServiceError
from app.services.vector_store import VectorStore
from app.models.schemas import KnowledgeListItem, KnowledgeDetailResponse

router = APIRouter()

# Instantiate expensive objects ONCE at module load, not per-request.
# Loading Whisper on every request would be extremely slow.
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


@router.post("/upload", response_model=KnowledgeDetailResponse)
async def upload_video(
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Accepts a video file, creates a metadata record in database with status 'processing',
    copies the file permanently to storage/uploads/video/, then runs it through the
    pipeline (extract audio -> transcribe -> chunk -> embed -> store in Pinecone).
    Updates status to 'ready' and saves duration on success.
    On failure, marks status 'failed' and cleans up partial vectors from Pinecone.
    """
    file_id = str(uuid.uuid4())
    stored_filename = f"{file_id}_{video.filename}"
    
    video_storage_dir = Path(settings.UPLOAD_STORAGE_DIR) / "video"
    video_storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = str(video_storage_dir / stored_filename)

    # 1. Get file size
    video.file.seek(0, 2)
    file_size = video.file.tell()
    video.file.seek(0)
    
    mime_type = video.content_type or "video/mp4"

    knowledge_manager = KnowledgeManager(db)
    
    # 2. Create initial DB record with 'processing' status
    kb_file = knowledge_manager.create_knowledge_record(
        file_id=file_id,
        title=Path(video.filename).stem,
        file_type="video",
        original_filename=video.filename,
        stored_filename=stored_filename,
        storage_path=storage_path,
        file_size=file_size,
        mime_type=mime_type,
        pinecone_namespace=f"kb_{file_id}",
    )

    # 3. Save uploaded video permanently to storage path
    try:
        with open(storage_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        knowledge_manager.update_status(file_id, status="failed")
        raise HTTPException(status_code=500, detail=f"Failed to save video file: {e}")

    # Set up temp folder for temporary audio processing
    temp_dir = Path(settings.TEMP_DIR) / file_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    audio_path = temp_dir / "audio.wav"

    try:
        # 4. Extract audio
        extract_audio(storage_path, str(audio_path))

        # 5. Transcribe
        segments = transcriber.transcribe(str(audio_path))
        if not segments:
            raise ValueError("No speech detected in the video.")

        # 6. Chunk
        chunks = chunker.chunk_transcript(segments)

        # 7. Embed
        chunk_texts = [c.text for c in chunks]
        chunk_vectors = embedding_service.embed(chunk_texts)
        chunk_metadata = [{"start_time": c.start_time, "end_time": c.end_time} for c in chunks]

        # 8. Upsert to Pinecone under this file's namespace
        vector_store.upsert_chunks(kb_file.pinecone_namespace, chunk_texts, chunk_vectors, chunk_metadata)

        # 9. Mark DB record ready and save duration
        duration = segments[-1].end if segments else None
        kb_file = knowledge_manager.update_status(
            file_id=file_id,
            status="ready",
            duration=duration,
            processed_at=datetime.utcnow()
        )

        return kb_file

    except AudioExtractionError as e:
        print(f"Audio extraction failed: {e}")
        traceback.print_exc()
        knowledge_manager.update_status(file_id, status="failed")
        # Cleanup any partial vectors
        try:
            vector_store.delete_session(kb_file.pinecone_namespace)
        except Exception:
            pass
        raise HTTPException(status_code=422, detail=f"Audio extraction failed: {e}")

    except EmbeddingServiceError as e:
        print(f"Embedding service failed: {e}")
        traceback.print_exc()
        knowledge_manager.update_status(file_id, status="failed")
        try:
            vector_store.delete_session(kb_file.pinecone_namespace)
        except Exception:
            pass
        raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {e}")

    except Exception as e:
        print(f"File processing failed: {e}")
        traceback.print_exc()
        knowledge_manager.update_status(file_id, status="failed")
        try:
            vector_store.delete_session(kb_file.pinecone_namespace)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")

    finally:
        # 10. Clean up temp audio processing files
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/videos", response_model=list[KnowledgeListItem])
async def list_videos(db: Session = Depends(get_db)):
    """List all video files in the library."""
    knowledge_manager = KnowledgeManager(db)
    files = knowledge_manager.list_knowledge_files(file_type="video")
    return [
        KnowledgeListItem(
            id=f.id,
            title=f.title,
            status=f.status,
            created_at=f.created_at
        )
        for f in files
    ]


@router.get("/videos/{file_id}", response_model=KnowledgeDetailResponse)
async def get_video_detail(file_id: str, db: Session = Depends(get_db)):
    """Retrieve metadata detail for a specific video."""
    knowledge_manager = KnowledgeManager(db)
    f = knowledge_manager.get_knowledge_file(file_id)
    if not f or f.type != "video":
        raise HTTPException(status_code=404, detail="Video not found")
    return f


@router.delete("/videos/{file_id}")
async def delete_video(file_id: str, db: Session = Depends(get_db)):
    """Delete a video's database record, physical file, and Pinecone vectors."""
    knowledge_manager = KnowledgeManager(db)
    f = knowledge_manager.get_knowledge_file(file_id)
    if not f or f.type != "video":
        raise HTTPException(status_code=404, detail="Video not found")

    # 1. Delete from Pinecone
    try:
        vector_store.delete_session(f.pinecone_namespace)
    except Exception as e:
        print(f"Warning: Failed to delete Pinecone namespace {f.pinecone_namespace}: {e}")

    # 2. Delete stored file from disk
    if f.storage_path:
        file_path = Path(f.storage_path)
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete video file {f.storage_path}: {e}")

    # 3. Delete metadata record from DB
    knowledge_manager.delete_knowledge_record(file_id)

    return {"status": "success", "message": f"Video {file_id} deleted successfully."}