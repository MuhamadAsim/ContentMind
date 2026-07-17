"""
Service for managing knowledge file database records.
Handles only metadata and database operations.
"""
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.knowledge import KnowledgeFile

class KnowledgeManager:
    """
    Handles all CRUD operations for the persistent KnowledgeFile model and
    interacts with the database session.
    """
    def __init__(self, db: Session):
        self.db = db

    def create_knowledge_record(
        self,
        file_id: str,
        title: str,
        file_type: str,
        original_filename: str,
        stored_filename: str,
        storage_path: str,
        file_size: int,
        mime_type: str,
        pinecone_namespace: str,
    ) -> KnowledgeFile:
        """Create a new knowledge file record in the database with 'processing' status."""
        knowledge_file = KnowledgeFile(
            id=file_id,
            title=title,
            type=file_type,
            original_filename=original_filename,
            stored_filename=stored_filename,
            storage_path=storage_path,
            file_size=file_size,
            mime_type=mime_type,
            pinecone_namespace=pinecone_namespace,
            status="processing",
        )
        self.db.add(knowledge_file)
        self.db.commit()
        self.db.refresh(knowledge_file)
        return knowledge_file

    def update_status(
        self,
        file_id: str,
        status: str,
        duration: float = None,
        processed_at: datetime = None,
    ) -> KnowledgeFile | None:
        """Update the processing status of a knowledge file record."""
        knowledge_file = self.db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
        if knowledge_file:
            knowledge_file.status = status
            if duration is not None:
                knowledge_file.duration = duration
            if processed_at is not None:
                knowledge_file.processed_at = processed_at
            self.db.commit()
            self.db.refresh(knowledge_file)
        return knowledge_file

    def get_knowledge_file(self, file_id: str) -> KnowledgeFile | None:
        """Retrieve a single knowledge file record by its ID."""
        return self.db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()

    def list_knowledge_files(self, file_type: str = None) -> list[KnowledgeFile]:
        """List all knowledge files, optionally filtered by type (e.g. 'video')."""
        query = self.db.query(KnowledgeFile)
        if file_type:
            query = query.filter(KnowledgeFile.type == file_type)
        return query.all()

    def delete_knowledge_record(self, file_id: str) -> KnowledgeFile | None:
        """Delete a knowledge file record from the database."""
        knowledge_file = self.db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
        if knowledge_file:
            self.db.delete(knowledge_file)
            self.db.commit()
        return knowledge_file


if __name__ == "__main__":
    # Standalone test block for KnowledgeManager
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base

    print("Running standalone tests for KnowledgeManager...")

    # Set up in-memory sqlite for test
    TEST_DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()

    manager = KnowledgeManager(db)

    # Test create
    test_id = "test-uuid-999"
    print("Testing create_knowledge_record...")
    v = manager.create_knowledge_record(
        file_id=test_id,
        title="Test Video File",
        file_type="video",
        original_filename="sample.mp4",
        stored_filename="test-uuid-999.mp4",
        storage_path="storage/uploads/video/test-uuid-999.mp4",
        file_size=1024 * 1024 * 5,  # 5MB
        mime_type="video/mp4",
        pinecone_namespace="kb_test-uuid-999",
    )
    assert v.id == test_id
    assert v.status == "processing"
    assert v.type == "video"
    assert v.file_size == 1024 * 1024 * 5
    assert v.mime_type == "video/mp4"
    print("ok - create_knowledge_record passed")

    # Test update to ready
    print("Testing update_status to ready...")
    now = datetime.utcnow()
    v = manager.update_status(test_id, status="ready", duration=45.2, processed_at=now)
    assert v.status == "ready"
    assert v.duration == 45.2
    assert v.processed_at == now
    print("ok - update_status passed")

    # Test get
    print("Testing get_knowledge_file...")
    v_get = manager.get_knowledge_file(test_id)
    assert v_get is not None
    assert v_get.title == "Test Video File"
    print("ok - get_knowledge_file passed")

    # Test list
    print("Testing list_knowledge_files...")
    v_list = manager.list_knowledge_files("video")
    assert len(v_list) == 1
    assert v_list[0].id == test_id
    print("ok - list_knowledge_files passed")

    # Test delete
    print("Testing delete_knowledge_record...")
    v_del = manager.delete_knowledge_record(test_id)
    assert v_del is not None
    assert manager.get_knowledge_file(test_id) is None
    print("ok - delete_knowledge_record passed")

    db.close()
    print("All KnowledgeManager standalone tests completed successfully!")
