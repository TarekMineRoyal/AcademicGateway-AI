import logging
from pydantic import BaseModel

from domain.models.major import Major
from domain.models.specialty import Specialty
from application.interfaces.vector_repositories import ICurriculumRepository
from application.exceptions.application_exceptions import VectorRepositoryException

logger = logging.getLogger(__name__)


class SyncMajorCommand(BaseModel):
    """
    DTO representing a primary academic major track update from C#.
    Uses a 'fat' DTO pattern to directly accept the materialized domain read-model.
    """
    major: Major


class SyncSpecialtyCommand(BaseModel):
    """
    DTO representing a narrow sub-track specialization update from C#.
    Uses a 'fat' DTO pattern to directly accept the materialized domain read-model.
    """
    specialty: Specialty


class SyncMajorHandler:
    """Coordinates writing incoming Major lookup text mutations into storage."""

    def __init__(self, repository: ICurriculumRepository):
        self._repository = repository

    def handle(self, command: SyncMajorCommand) -> None:
        """
        Executes the synchronization pipeline for a Major entity.

        Raises:
            VectorRepositoryException: If the persistence layer fails.
        """
        major_id = command.major.id
        logger.info(f"Starting synchronization pipeline for major ID: {major_id}")

        try:
            logger.debug(f"Upserting major record into storage: {major_id}")
            self._repository.upsert_major(command.major)
            logger.info(f"Successfully synchronized major space for: {major_id}")
        except Exception as ex:
            error_msg = f"Failed to persist major record for {major_id} into storage."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex


class SyncSpecialtyHandler:
    """Coordinates writing incoming Specialty lookup text mutations into storage."""

    def __init__(self, repository: ICurriculumRepository):
        self._repository = repository

    def handle(self, command: SyncSpecialtyCommand) -> None:
        """
        Executes the synchronization pipeline for a Specialty entity.

        Raises:
            VectorRepositoryException: If the persistence layer fails.
        """
        specialty_id = command.specialty.id
        logger.info(f"Starting synchronization pipeline for specialty ID: {specialty_id}")

        try:
            logger.debug(f"Upserting specialty record into storage: {specialty_id}")
            self._repository.upsert_specialty(command.specialty)
            logger.info(f"Successfully synchronized specialty space for: {specialty_id}")
        except Exception as ex:
            error_msg = f"Failed to persist specialty record for {specialty_id} into storage."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex