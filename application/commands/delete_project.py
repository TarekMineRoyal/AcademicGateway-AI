import logging
import uuid
from pydantic import BaseModel, Field

from application.exceptions.application_exceptions import VectorRepositoryException
from application.interfaces.vector_repositories import IProjectTemplateVectorRepository

logger = logging.getLogger(__name__)


class DeleteProjectCommand(BaseModel):
    """Command payload for removing a project template vector blueprint."""

    id: uuid.UUID = Field(
        ..., description="Unique primary identifier (UUID) of the project template entity to remove."
    )


class DeleteProjectCommandHandler:
    """Handles execution of project template vector node deletion."""

    def __init__(self, project_repository: IProjectTemplateVectorRepository) -> None:
        self._project_repository = project_repository

    def handle(self, command: DeleteProjectCommand) -> None:
        project_id = command.id
        logger.info(f"Executing deletion for project template vector node ID: {project_id}")

        try:
            self._project_repository.delete(project_id)
            logger.info(f"Successfully processed deletion for project template vector node ID: {project_id}")
        except Exception as ex:
            error_msg = f"Failed to delete project template vector record for ID {project_id}."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex