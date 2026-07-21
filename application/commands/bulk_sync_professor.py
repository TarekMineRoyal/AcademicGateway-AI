import logging
from typing import List
from pydantic import BaseModel, Field

from application.commands.sync_professor import SyncProfessorCommand
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)
from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import IProfessorVectorRepository
from application.services.formatters.professor import format_professor_document
from infrastructure.config.settings import settings
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.repositories.professor_repository import (
    ProfessorVectorRepository,
)

logger = logging.getLogger(__name__)


class BulkSyncProfessorCommand(BaseModel):
    """
    Bulk payload DTO containing a batch of professor records for backfill/synchronization.
    """

    items: List[SyncProfessorCommand] = Field(
        default_factory=list,
        description="Collection of professor sync payloads to be processed in bulk.",
    )


class BulkSyncProfessorCommandHandler:
    """
    Orchestrates VRAM-safe batch synchronization for Professor records.
    Processes items in configurable chunks, generates batch embeddings,
    upserts to a staging table, performs a Blue/Green table swap, and invalidates cache.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        professor_repository: IProfessorVectorRepository | None = None,
        staging_table_name: str = "professors_sync",
        live_table_name: str = "professors",
    ):
        self._embedding_service = embedding_service
        self._live_repository = professor_repository or ProfessorVectorRepository(
            table_name=live_table_name
        )
        self._staging_repository = ProfessorVectorRepository(
            table_name=staging_table_name
        )
        self._staging_table_name = staging_table_name
        self._live_table_name = live_table_name

    def handle(self, command: BulkSyncProfessorCommand) -> None:
        """
        Executes the Blue/Green bulk synchronization workflow.
        Catches background exceptions internally to prevent corrupting ASGI HTTP streams.
        """
        total_items = len(command.items)
        if total_items == 0:
            logger.info(
                "BulkSyncProfessorCommand received an empty items list. Nothing to process."
            )
            return

        try:
            chunk_size = settings.BATCH_CHUNK_SIZE
            total_chunks = (total_items + chunk_size - 1) // chunk_size
            logger.info(
                f"Starting bulk synchronization for {total_items} professors across {total_chunks} chunk(s) (Chunk size: {chunk_size})"
            )

            # 1. Process items in VRAM-safe chunks to protect GPU memory
            for i in range(0, total_items, chunk_size):
                chunk = command.items[i : i + chunk_size]
                chunk_num = (i // chunk_size) + 1
                logger.info(
                    f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} records) for bulk professor sync."
                )

                chunk_professors = [item.professor for item in chunk]

                # Map the pure formatter over the chunk
                prose_documents = [
                    format_professor_document(
                        professor=item.professor, interest_areas=item.interest_areas
                    )
                    for item in chunk
                ]

                # Compute batch embeddings using PyTorch parallelization
                try:
                    logger.debug(
                        f"Generating batch embeddings for chunk {chunk_num}/{total_chunks} ({len(chunk)} records)"
                    )
                    vectors = self._embedding_service.embed_documents_batch(prose_documents)
                except Exception as ex:
                    error_msg = f"Failed to generate batch embeddings for professor chunk {chunk_num}/{total_chunks}."
                    logger.error(f"{error_msg} Details: {str(ex)}")
                    raise EmbeddingServiceException(error_msg) from ex

                # Upsert batch into staging table
                try:
                    logger.debug(
                        f"Upserting professor chunk {chunk_num}/{total_chunks} into staging table '{self._staging_table_name}'"
                    )
                    self._staging_repository.bulk_upsert(chunk_professors, vectors)
                except Exception as ex:
                    error_msg = (
                        f"Failed to bulk upsert professor chunk {chunk_num}/{total_chunks} into staging storage."
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
                error_msg = "Failed during Blue/Green table swap operation for professors."
                logger.error(f"{error_msg} Details: {str(ex)}")
                raise VectorRepositoryException(error_msg) from ex

            # 3. Reload in-memory table handles across repository instances
            try:
                logger.info(
                    "Invalidating in-memory cache for live professor repository handle."
                )
                self._live_repository.reload_table()
                self._staging_repository.reload_table()
            except Exception as ex:
                error_msg = "Failed to reload table cache for professor repository after Blue/Green swap."
                logger.error(f"{error_msg} Details: {str(ex)}")
                raise VectorRepositoryException(error_msg) from ex

            logger.info(
                f"Successfully completed Blue/Green bulk sync for {total_items} professors."
            )

        except Exception as ex:
            logger.critical(
                f"Bulk professor synchronization background process failed: {str(ex)}",
                exc_info=True,
            )