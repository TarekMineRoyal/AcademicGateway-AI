import logging
import uuid
from pydantic import BaseModel, Field

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
        logger.info(f"Executing deletion for project template vector node: {command.id}")
        self._project_repository.delete(command.id)
        logger.info(f"Successfully processed deletion for project template vector node: {command.id}")