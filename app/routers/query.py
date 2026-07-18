"""
POST /api/ask — answers a question about any previously processed knowledge
file (video, audio, or document) by looking up its Pinecone namespace.
"""
import base64
import shutil
import uuid
import wave
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Form, File, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.services.knowledge_manager import KnowledgeManager
from app.models.schemas import AskResponse
from app.services.qa_engine import QAEngine
from app.services.audio_extractor import extract_audio, AudioExtractionError
from app.routers.upload import embedding_service, vector_store  # reuse same instances

router = APIRouter()

qa_engine = QAEngine(
    openrouter_api_key=settings.OPENROUTER_API_KEY,
    model=settings.DEFAULT_AI_MODEL,
    embedding_service=embedding_service,
    vector_store=vector_store,
)

_piper_voice = None

def get_piper_voice():
    global _piper_voice
    if _piper_voice is None:
        model_path = Path(__file__).parent.parent / "resources" / "voices" / "en_US-lessac-medium.onnx"
        config_path = Path(__file__).parent.parent / "resources" / "voices" / "en_US-lessac-medium.onnx.json"
        if not model_path.exists():
            raise RuntimeError(f"Piper voice model not found at {model_path}. Run download_voice script first.")
        print(f"Loading Piper voice model from {model_path}...")
        from piper.voice import PiperVoice
        _piper_voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        print("Piper voice model loaded successfully.")
    return _piper_voice


def synthesize_text_to_base64(text: str) -> str:
    # Ensure temp dir exists
    temp_dir = Path(settings.TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to temp wav
    temp_wav_path = temp_dir / f"tts_{uuid.uuid4()}.wav"
    
    try:
        voice = get_piper_voice()
        with wave.open(str(temp_wav_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
            
        # Read WAV file and base64-encode it
        with open(temp_wav_path, "rb") as f:
            audio_bytes = f.read()
            
        return base64.b64encode(audio_bytes).decode("utf-8")
    finally:
        if temp_wav_path.exists():
            temp_wav_path.unlink()


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    file_id: str = Form(...),
    question: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    voice_output: bool = Form(False),
    db: Session = Depends(get_db)
):
    """
    Answer a question against any ready knowledge file.
    Accepts text question or audio query file via multipart/form-data.
    """
    knowledge_manager = KnowledgeManager(db)
    f = knowledge_manager.get_knowledge_file(file_id)

    if not f:
        raise HTTPException(status_code=404, detail="File not found.")

    if f.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"File is not ready for querying. Current status: {f.status}",
        )

    actual_question = question

    if file:
        temp_dir = Path(settings.TEMP_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        temp_input_id = str(uuid.uuid4())
        suffix = Path(file.filename or "audio").suffix or ".webm"
        temp_input_path = temp_dir / f"query_{temp_input_id}{suffix}"
        temp_wav_path = temp_dir / f"query_{temp_input_id}.wav"
        
        try:
            with open(temp_input_path, "wb") as temp_f:
                shutil.copyfileobj(file.file, temp_f)
                
            # Convert to WAV
            extract_audio(str(temp_input_path), str(temp_wav_path))
            
            # Transcribe
            from app.routers.upload import transcriber
            segments = transcriber.transcribe(str(temp_wav_path))
            transcribed_text = transcriber.get_full_text(segments).strip()
            
            if not transcribed_text:
                raise HTTPException(
                    status_code=400,
                    detail="No speech detected in the audio query. Please try again."
                )
                
            actual_question = transcribed_text
            
        except AudioExtractionError as e:
            print(f"Error converting audio query: {e}")
            raise HTTPException(
                status_code=400,
                detail="Could not process the audio query. Please ensure it is a valid audio recording."
            )
        finally:
            # Clean up temp query files
            if temp_input_path.exists():
                temp_input_path.unlink()
            if temp_wav_path.exists():
                temp_wav_path.unlink()

    if not actual_question or not actual_question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = qa_engine.ask(
            session_id=f.pinecone_namespace,
            question=actual_question,
            top_k=settings.RETRIEVAL_TOP_K,
        )
        
        audio_base64 = None
        if voice_output:
            try:
                audio_base64 = synthesize_text_to_base64(result["answer"])
            except Exception as tts_err:
                print(f"Warning: Piper TTS synthesis failed: {tts_err}")
                # Don't fail the whole request if only TTS fails
                
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "audio_base64": audio_base64
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")