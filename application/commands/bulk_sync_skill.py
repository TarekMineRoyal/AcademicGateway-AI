import logging
from typing import List
from pydantic import BaseModel, Field

from application.commands.sync_skill import SyncSkillCommand
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)
from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import ISkillVectorRepository
from application.services.formatters.skill import format_skill_document
from infrastructure.config.settings import settings
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.repositories.skill_repository import (
    SkillVectorRepository,
)

logger = logging.getLogger(__name__)


class BulkSyncSkillCommand(BaseModel):
    """
    Bulk payload DTO containing a batch of skill records for backfill/synchronization.
    """

    items: List[SyncSkillCommand] = Field(
        default_factory=list,
        description="Collection of skill sync payloads to be processed in bulk.",
    )


class BulkSyncSkillCommandHandler:
    """
    Orchestrates VRAM-safe batch synchronization for Skill records.
    Processes items in configurable chunks, generates batch embeddings,
    upserts to a staging table, performs a Blue/Green table swap, and invalidates cache.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        skill_repository: ISkillVectorRepository | None = None,
        staging_table_name: str = "skills_sync",
        live_table_name: str = "skills",
    ):
        self._embedding_service = embedding_service
        self._live_repository = skill_repository or SkillVectorRepository(
            table_name=live_table_name
        )
        self._staging_repository = SkillVectorRepository(
            table_name=staging_table_name
        )
        self._staging_table_name = staging_table_name
        self._live_table_name = live_table_name

    def handle(self, command: BulkSyncSkillCommand) -> None:
        """
        Executes the Blue/Green bulk synchronization workflow for skills.

        Raises:
            EmbeddingServiceException: If the batch embedding generation fails.
            VectorRepositoryException: If persistence or table swapping fails.
        """
        total_items = len(command.items)
        if total_items == 0:
            logger.info(
                "BulkSyncSkillCommand received an empty items list. Nothing to process."
            )
            return

        chunk_size = settings.BATCH_CHUNK_SIZE
        total_chunks = (total_items + chunk_size - 1) // chunk_size
        logger.info(
            f"Starting bulk synchronization for {total_items} skills across {total_chunks} chunk(s) (Chunk size: {chunk_size})"
        )

        # 1. Process items in VRAM-safe chunks to protect GPU memory
        for i in range(0, total_items, chunk_size):
            chunk = command.items[i : i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            logger.info(
                f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} records) for bulk skill sync."
            )

            chunk_skills = [item.skill for item in chunk]

            # Map the pure formatter over the chunk
            prose_documents = [format_skill_document(item.skill) for item in chunk]

            # Compute batch embeddings using PyTorch parallelization
            try:
                logger.debug(
                    f"Generating batch embeddings for chunk {chunk_num}/{total_chunks} ({len(chunk)} records)"
                )
                vectors = self._embedding_service.embed_documents_batch(prose_documents)
            except Exception as ex:
                error_msg = f"Failed to generate batch embeddings for skill chunk {chunk_num}/{total_chunks}."
                logger.error(f"{error_msg} Details: {str(ex)}")
                raise EmbeddingServiceException(error_msg) from ex

            # Upsert batch into staging table
            try:
                logger.debug(
                    f"Upserting skill chunk {chunk_num}/{total_chunks} into staging table '{self._staging_table_name}'"
                )
                self._staging_repository.bulk_upsert(chunk_skills, vectors)
            except Exception as ex:
                error_msg = f"Failed to bulk upsert skill chunk {chunk_num}/{total_chunks} into staging storage."
                logger.error(f"{error_msg} Details: {str(ex)}")
                raise VectorRepositoryException(error_msg) from ex

        # 2. Promote staging table to live table via Blue/Green swap
        try:
            logger.info(
                f"Promoting staging table '{self._staging_table_name}' to live production table '{self._live_table_name}'"
            )
            lancedb_client.swap_tables(self._staging_table_name, self._live_table_name)
        except Exception as ex:
            error_msg = "Failed during Blue/Green table swap operation for skills."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex

        # 3. Reload in-memory table handles across repository instances
        try:
            logger.info("Invalidating in-memory cache for live skill repository handle.")
            self._live_repository.reload_table()
            self._staging_repository.reload_table()
        except Exception as ex:
            error_msg = "Failed to reload table cache for skill repository after Blue/Green swap."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex

        logger.info(
            f"Successfully completed Blue/Green bulk sync for {total_items} skills."
        )