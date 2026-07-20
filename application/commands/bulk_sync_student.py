import logging
from typing import List
from pydantic import BaseModel, Field

from application.commands.sync_student import SyncStudentCommand
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)
from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import IStudentVectorRepository
from application.services.formatters.student import format_student_document
from infrastructure.config.settings import settings
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.repositories.student_repository import (
    StudentVectorRepository,
)

logger = logging.getLogger(__name__)


class BulkSyncStudentCommand(BaseModel):
    """
    Bulk payload DTO containing a batch of student profile records for backfill/synchronization.
    """

    items: List[SyncStudentCommand] = Field(
        default_factory=list,
        description="Collection of student sync payloads to be processed in bulk.",
    )


class BulkSyncStudentCommandHandler:
    """
    Orchestrates VRAM-safe batch synchronization for Student profile records.
    Processes items in configurable chunks, generates batch embeddings,
    upserts to a staging table, performs a Blue/Green table swap, and invalidates cache.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        student_repository: IStudentVectorRepository | None = None,
        staging_table_name: str = "students_sync",
        live_table_name: str = "students",
    ):
        self._embedding_service = embedding_service
        self._live_repository = student_repository or StudentVectorRepository(
            table_name=live_table_name
        )
        self._staging_repository = StudentVectorRepository(
            table_name=staging_table_name
        )
        self._staging_table_name = staging_table_name
        self._live_table_name = live_table_name

    def handle(self, command: BulkSyncStudentCommand) -> None:
        """
        Executes the Blue/Green bulk synchronization workflow for student records.

        Raises:
            EmbeddingServiceException: If the batch embedding generation fails.
            VectorRepositoryException: If persistence or table swapping fails.
        """
        total_items = len(command.items)
        if total_items == 0:
            logger.info(
                "BulkSyncStudentCommand received an empty items list. Nothing to process."
            )
            return

        chunk_size = settings.BATCH_CHUNK_SIZE
        logger.info(
            f"Starting bulk synchronization for {total_items} students (Chunk size: {chunk_size})"
        )

        # 1. Process items in VRAM-safe chunks to protect GPU memory
        for i in range(0, total_items, chunk_size):
            chunk = command.items[i : i + chunk_size]
            chunk_students = [item.student for item in chunk]

            # Map the pure formatter over the chunk
            prose_documents = [
                format_student_document(
                    student=item.student,
                    major_name=item.major_name,
                    specialty_names=item.specialty_names,
                    skill_names=item.skill_names,
                )
                for item in chunk
            ]

            # Compute batch embeddings using PyTorch parallelization
            try:
                logger.debug(
                    f"Generating batch embeddings for chunk starting at index {i} ({len(chunk)} records)"
                )
                vectors = self._embedding_service.embed_documents_batch(prose_documents)
            except Exception as ex:
                error_msg = f"Failed to generate batch embeddings for student chunk starting at index {i}."
                logger.error(f"{error_msg} Details: {str(ex)}")
                raise EmbeddingServiceException(error_msg) from ex

            # Upsert batch into staging table
            try:
                logger.debug(
                    f"Upserting student chunk into staging table '{self._staging_table_name}'"
                )
                self._staging_repository.bulk_upsert(chunk_students, vectors)
            except Exception as ex:
                error_msg = (
                    "Failed to bulk upsert student chunk into staging storage."
                )
                logger.error(f"{error_msg} Details: {str(ex)}")
                raise VectorRepositoryException(error_msg) from ex

        # 2. Promote staging table to live table via Blue/Green swap
        try:
            logger.info(
                f"Promoting staging table '{self._staging_table_name}' to live production table '{self._live_table_name}'"
            )
            lancedb_client.swap_tables(self._staging_table_name, self._live_table_name)
        except Exception as ex:
            error_msg = "Failed during Blue/Green table swap operation for students."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex

        # 3. Reload in-memory table handles across repository instances
        try:
            logger.info(
                "Invalidating in-memory cache for live student repository handle."
            )
            self._live_repository.reload_table()
            self._staging_repository.reload_table()
        except Exception as ex:
            error_msg = "Failed to reload table cache for student repository after Blue/Green swap."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex

        logger.info(
            f"Successfully completed Blue/Green bulk sync for {total_items} students."
        )