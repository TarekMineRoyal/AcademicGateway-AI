import logging
import uuid
from pydantic import BaseModel, Field

from application.exceptions.application_exceptions import VectorRepositoryException
from application.interfaces.vector_repositories import IStudentVectorRepository

logger = logging.getLogger(__name__)


class DeleteStudentCommand(BaseModel):
    """Command payload for removing a student vector profile."""

    id: uuid.UUID = Field(
        ..., description="Unique primary identifier (UUID) of the student entity to remove."
    )


class DeleteStudentCommandHandler:
    """Handles execution of student vector node deletion."""

    def __init__(self, student_repository: IStudentVectorRepository) -> None:
        self._student_repository = student_repository

    def handle(self, command: DeleteStudentCommand) -> None:
        student_id = command.id
        logger.info(f"Executing deletion for student vector node ID: {student_id}")

        try:
            self._student_repository.delete(student_id)
            logger.info(f"Successfully processed deletion for student vector node ID: {student_id}")
        except Exception as ex:
            error_msg = f"Failed to delete student vector record for ID {student_id}."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex