import logging
from typing import List
from pydantic import BaseModel, Field

from domain.models.professor import Professor
from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import IProfessorVectorRepository
from application.services.formatters.professor import format_professor_document
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)

logger = logging.getLogger(__name__)


class SyncProfessorCommand(BaseModel):
    """
    The incoming fat payload DTO from the C# core backend synchronization event.
    Flattens the research interest aggregate strings directly at the boundary source
    to maintain high-performance ingestion.
    """
    professor: Professor
    interest_areas: List[str] = Field(
        default_factory=list,
        description="Text labels of the professor's active research interests."
    )


class SyncProfessorCommandHandler:
    """
    Coordinates the ingestion pipeline for Professor faculty profiles.
    Transforms raw instructional and research attributes into semantic prose,
    generates isolated vectors, and updates the LanceDB repository.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        professor_repository: IProfessorVectorRepository
    ):
        self._embedding_service = embedding_service
        self._professor_repository = professor_repository

    def handle(self, command: SyncProfessorCommand) -> None:
        """
        Executes the professor profile synchronization workflow.

        Raises:
            EmbeddingServiceException: If the embedding service adapter fails.
            VectorRepositoryException: If the LanceDB persistence layer fails.
        """
        professor_id = command.professor.id
        logger.info(f"Starting synchronization pipeline for professor ID: {professor_id}")

        # 1. Compile institutional variables and interest fields into a unified prose document
        prose_document = format_professor_document(
            professor=command.professor,
            interest_areas=command.interest_areas
        )

        # 2. Compute vector embeddings through the domain-isolated application port
        try:
            logger.debug(f"Generating semantic vector for professor profile: {professor_id}")
            vector = self._embedding_service.embed_document(prose_document)
        except Exception as ex:
            error_msg = f"Failed to generate vector embedding for professor {professor_id}."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise EmbeddingServiceException(error_msg) from ex

        # 3. Upsert the master read-model accompanied by its vector map into LanceDB
        try:
            logger.debug(f"Upserting professor record into vector storage: {professor_id}")
            self._professor_repository.upsert(command.professor, vector)
            logger.info(f"Successfully synchronized professor vector space for: {professor_id}")
        except Exception as ex:
            error_msg = f"Failed to persist vectorized professor profile for {professor_id} into storage."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex