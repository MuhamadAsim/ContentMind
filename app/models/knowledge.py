"""
SQLAlchemy model for persistent knowledge files (initially videos).
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, Integer
from app.database import Base

class KnowledgeFile(Base):
    __tablename__ = "knowledge_files"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    type = Column(String, nullable=False, default="video")  # e.g., "video", "pdf", etc.
    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)  # in bytes
    mime_type = Column(String, nullable=False)
    duration = Column(Float, nullable=True)  # in seconds (optional, e.g., for audio/video)
    status = Column(String, nullable=False, default="processing")  # "processing", "ready", "failed"
    pinecone_namespace = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)  # Set when status is updated to "ready"
