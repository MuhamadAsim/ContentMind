"""
Pydantic models for API request/response validation.
"""
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class UploadResponse(BaseModel):
    session_id: str
    status: str  # "ready"
    chunks_created: int


class AskRequest(BaseModel):
    file_id: str
    question: str


class SourceReference(BaseModel):
    # start_time / end_time are set for audio/video chunks;
    # they will be 0.0 / 0.0 for document chunks (no timestamps).
    start_time: float
    end_time: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceReference]


class ErrorResponse(BaseModel):
    detail: str


class KnowledgeListItem(BaseModel):
    id: str
    title: str
    type: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class KnowledgeDetailResponse(BaseModel):
    id: str
    title: str
    type: str
    original_filename: str
    stored_filename: str
    storage_path: str
    file_size: int
    mime_type: str
    duration: float | None = None
    page_count: int | None = None
    status: str
    pinecone_namespace: str
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)