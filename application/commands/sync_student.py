import logging
from pydantic import BaseModel, Field
from typing import List

from domain.models.student import Student
from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import IStudentVectorRepository
from application.services.formatters.student import format_student_document
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)

logger = logging.getLogger(__name__)


class SyncStudentCommand(BaseModel):
    """
    The incoming fat payload DTO from the C# core backend synchronization event.
    Enriches relational keys with descriptive text names at the boundary source.
    """
    student: Student
    major_name: str
    specialty_names: List[str] = Field(default_factory=list)
    skill_names: List[str] = Field(default_factory=list)


class SyncStudentCommandHandler:
    """
    Coordinates the ingestion pipeline for student read-models.
    Translates structural data to prose, generates vectors, and updates storage.
    """

    def __init__(
            self,
            embedding_service: IEmbeddingService,
            student_repository: IStudentVectorRepository
    ):
        self._embedding_service = embedding_service
        self._student_repository = student_repository

    def handle(self, command: SyncStudentCommand) -> None:
        """
        Executes the synchronization transaction workflow.

        Raises:
            EmbeddingServiceException: If vector generation fails.
            VectorRepositoryException: If database indexing fails.
        """
        student_id = command.student.id
        logger.info(f"Starting synchronization pipeline for student ID: {student_id}")

        # 1. Materialize raw structural attributes into clean text prose paragraphs
        prose_document = format_student_document(
            student=command.student,
            major_name=command.major_name,
            specialty_names=command.specialty_names,
            skill_names=command.skill_names
        )

        # 2. Generate vector embedding using the isolated application port
        try:
            logger.debug(f"Generating embedding vector for student: {student_id}")
            vector = self._embedding_service.embed_document(prose_document)
        except Exception as ex:
            error_msg = f"Failed to generate vector embedding for student {student_id}."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise EmbeddingServiceException(error_msg) from ex

        # 3. Persist the domain read-model alongside its embedding to LanceDB
        try:
            logger.debug(f"Upserting student record to vector database: {student_id}")
            self._student_repository.upsert(command.student, vector)
            logger.info(f"Successfully synchronized student read-model vector space for: {student_id}")
        except Exception as ex:
            error_msg = f"Failed to persist vectorized student record for {student_id} into storage."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex