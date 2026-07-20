import logging
import uuid
from pydantic import BaseModel, Field

from application.exceptions.application_exceptions import VectorRepositoryException
from application.interfaces.vector_repositories import ISkillVectorRepository

logger = logging.getLogger(__name__)


class DeleteSkillCommand(BaseModel):
    """Command payload for removing a skill capability vector node."""

    id: uuid.UUID = Field(
        ..., description="Unique primary identifier (UUID) of the skill capability entity to remove."
    )


class DeleteSkillHandler:
    """Handles execution of skill capability vector node deletion."""

    def __init__(self, skill_repository: ISkillVectorRepository) -> None:
        self._skill_repository = skill_repository

    def handle(self, command: DeleteSkillCommand) -> None:
        skill_id = command.id
        logger.info(f"Executing deletion for skill vector node ID: {skill_id}")

        try:
            self._skill_repository.delete(skill_id)
            logger.info(f"Successfully processed deletion for skill vector node ID: {skill_id}")
        except Exception as ex:
            error_msg = f"Failed to delete skill vector record for ID {skill_id}."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex