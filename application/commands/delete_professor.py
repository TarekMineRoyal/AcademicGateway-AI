import logging
import uuid
from pydantic import BaseModel, Field

from application.exceptions.application_exceptions import VectorRepositoryException
from application.interfaces.vector_repositories import IProfessorVectorRepository

logger = logging.getLogger(__name__)


class DeleteProfessorCommand(BaseModel):
    """Command payload for removing a professor vector profile."""

    id: uuid.UUID = Field(
        ..., description="Unique primary identifier (UUID) of the professor entity to remove."
    )


class DeleteProfessorCommandHandler:
    """Handles execution of professor vector node deletion."""

    def __init__(self, professor_repository: IProfessorVectorRepository) -> None:
        self._professor_repository = professor_repository

    def handle(self, command: DeleteProfessorCommand) -> None:
        professor_id = command.id
        logger.info(f"Executing deletion for professor vector node ID: {professor_id}")

        try:
            self._professor_repository.delete(professor_id)
            logger.info(f"Successfully processed deletion for professor vector node ID: {professor_id}")
        except Exception as ex:
            error_msg = f"Failed to delete professor vector record for ID {professor_id}."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex