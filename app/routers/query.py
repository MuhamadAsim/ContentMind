"""
POST /api/ask — answers a question about a previously uploaded video,
using its file_id to look up its Pinecone namespace for scoped retrieval.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.services.knowledge_manager import KnowledgeManager
from app.models.schemas import AskRequest, AskResponse
from app.services.qa_engine import QAEngine
from app.routers.upload import embedding_service, vector_store  # reuse same instances

router = APIRouter()

qa_engine = QAEngine(
    openrouter_api_key=settings.OPENROUTER_API_KEY,
    model=settings.DEFAULT_AI_MODEL,
    embedding_service=embedding_service,
    vector_store=vector_store,
)


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, db: Session = Depends(get_db)):
    knowledge_manager = KnowledgeManager(db)
    f = knowledge_manager.get_knowledge_file(request.file_id)
    if not f or f.type != "video":
        raise HTTPException(
            status_code=404,
            detail="Video not found.",
        )
    if f.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Video is not ready for querying. Current status: {f.status}",
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Pass the pinecone_namespace as the session_id to QAEngine
        result = qa_engine.ask(
            session_id=f.pinecone_namespace,
            question=request.question,
            top_k=settings.RETRIEVAL_TOP_K,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")