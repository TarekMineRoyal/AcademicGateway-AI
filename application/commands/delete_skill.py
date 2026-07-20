import logging
import uuid
from pydantic import BaseModel, Field

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
        logger.info(f"Executing deletion for skill vector node: {command.id}")
        self._skill_repository.delete(command.id)
        logger.info(f"Successfully processed deletion for skill vector node: {command.id}")