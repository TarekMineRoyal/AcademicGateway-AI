import uuid
from pydantic import BaseModel

from domain.models.major import Major
from domain.models.specialty import Specialty
from abc import ABC, abstractmethod


class ICurriculumRepository(ABC):
    """
    Abstract interface contract for storing and looking up relational
    academic structural definitions (Majors and Specialties).
    """
    @abstractmethod
    def upsert_major(self, major: Major) -> None:
        pass

    @abstractmethod
    def upsert_specialty(self, specialty: Specialty) -> None:
        pass

    # --- ADDED TO FIX LOOKUP ERRORS ---
    @abstractmethod
    def get_major_name(self, major_id: uuid.UUID) -> Optional[str]:
        """Resolves a Major ID into its clean text string name for text formatting."""
        pass

    @abstractmethod
    def get_specialty_names(self, specialty_ids: List[uuid.UUID]) -> List[str]:
        """Resolves a collection of Specialty IDs into their text string names."""
        pass


class SyncMajorCommand(BaseModel):
    """DTO representing a primary academic major track update from C#."""
    id: uuid.UUID
    name: str


class SyncSpecialtyCommand(BaseModel):
    """DTO representing a narrow sub-track specialization update from C#."""
    id: uuid.UUID
    name: str
    major_id: uuid.UUID


class SyncMajorHandler:
    """Coordinates writing incoming Major lookup text mutations into storage."""
    def __init__(self, repository: ICurriculumRepository):
        self._repository = repository

    def handle(self, command: SyncMajorCommand) -> None:
        major = Major(id=command.id, name=command.name)
        self._repository.upsert_major(major)


class SyncSpecialtyHandler:
    """Coordinates writing incoming Specialty lookup text mutations into storage."""
    def __init__(self, repository: ICurriculumRepository):
        self._repository = repository

    def handle(self, command: SyncSpecialtyCommand) -> None:
        specialty = Specialty(
            id=command.id,
            name=command.name,
            major_id=command.major_id
        )
        self._repository.upsert_specialty(specialty)