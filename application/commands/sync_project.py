import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from domain.models.project_template import ProjectTemplate
from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import IProjectTemplateVectorRepository
from application.services.formatters.project_template import format_project_document
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)

logger = logging.getLogger(__name__)


class SyncProjectCommand(BaseModel):
    """
    The incoming fat payload DTO from the C# core backend synchronization event.
    Enriches relational keys with descriptive text labels at the source boundary
    to keep the vector engine decoupled from transactional databases.
    """
    template: ProjectTemplate
    major_name: Optional[str] = Field(default=None, description="The descriptive name of the aligned major.")
    specialty_name: Optional[str] = Field(default=None, description="The descriptive name of the concentration track.")
    skill_names: List[str] = Field(default_factory=list, description="Text labels of required competencies.")


class SyncProjectCommandHandler:
    """
    Coordinates the ingestion pipeline for Project Template blueprints.
    Flattens structural project parameters to prose, generates vector profiles,
    and updates the LanceDB read-model collection.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        project_repository: IProjectTemplateVectorRepository
    ):
        self._embedding_service = embedding_service
        self._project_repository = project_repository

    def handle(self, command: SyncProjectCommand) -> None:
        """
        Executes the project blueprint synchronization workflow.

        Raises:
            EmbeddingServiceException: If the embedding service adapter fails.
            VectorRepositoryException: If the LanceDB persistence layer fails.
        """
        template_id = command.template.id
        logger.info(
            f"Starting synchronization pipeline for project template ID: {template_id} ({command.template.title})"
        )

        # 1. Transform the structural project data into a clean narrative block
        prose_document = format_project_document(
            template=command.template,
            major_name=command.major_name,
            specialty_name=command.specialty_name,
            skill_names=command.skill_names
        )

        # 2. Extract vector embeddings via the isolated application port
        try:
            logger.debug(f"Generating semantic vector for project template: {template_id}")
            vector = self._embedding_service.embed_document(prose_document)
        except Exception as ex:
            error_msg = f"Failed to generate vector embedding for project template {template_id}."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise EmbeddingServiceException(error_msg) from ex

        # 3. Store the domain read-model alongside its vector map in LanceDB
        try:
            logger.debug(f"Upserting project template record into vector storage: {template_id}")
            self._project_repository.upsert(command.template, vector)
            logger.info(f"Successfully synchronized project template vector space for: {template_id}")
        except Exception as ex:
            error_msg = f"Failed to persist vectorized project blueprint for {template_id} into storage."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex