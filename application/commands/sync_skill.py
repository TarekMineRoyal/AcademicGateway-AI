import logging
from pydantic import BaseModel

from domain.models.skill import Skill
from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import ISkillVectorRepository
from application.services.formatters.skill import format_skill_document
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)

logger = logging.getLogger(__name__)


class SyncSkillCommand(BaseModel):
    """
    Data Transfer Object representing the payload sent by the C# backend
    whenever a global competency skill is created or modified.
    Uses a 'fat' DTO pattern to directly accept the materialized domain read-model.
    """
    skill: Skill


class SyncSkillHandler:
    """
    Handles the synchronization transaction lifecycle for an individual skill asset.
    """

    def __init__(self, embedding_service: IEmbeddingService, skill_repository: ISkillVectorRepository):
        """
        Injects the abstract decoupled infrastructure layer boundaries.
        """
        self._embedder = embedding_service
        self._repository = skill_repository

    def handle(self, command: SyncSkillCommand) -> None:
        """
        Executes the business synchronization pipeline for the skill entity.

        Args:
            command (SyncSkillCommand): The inbound mutation tracking criteria.

        Raises:
            EmbeddingServiceException: If the embedding service adapter fails.
            VectorRepositoryException: If the LanceDB persistence layer fails.
        """
        skill_id = command.skill.id
        logger.info(f"Starting synchronization pipeline for skill ID: {skill_id} ({command.skill.name})")

        # 1. Compile entity attributes into a clean prose paragraph text block
        narrative_document = format_skill_document(command.skill)

        # 2. Generate the mathematical vector (Infrastructure injects Nomic prefixes)
        try:
            logger.debug(f"Generating semantic vector for skill: {skill_id}")
            vector = self._embedder.embed_document(narrative_document)
        except Exception as ex:
            error_msg = f"Failed to generate vector embedding for skill {skill_id}."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise EmbeddingServiceException(error_msg) from ex

        # 3. Commit the structural record data and float array onto the storage index
        try:
            logger.debug(f"Upserting skill record into vector storage: {skill_id}")
            self._repository.upsert(command.skill, vector)
            logger.info(f"Successfully synchronized skill vector space for: {skill_id}")
        except Exception as ex:
            error_msg = f"Failed to persist vectorized skill record for {skill_id} into storage."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex