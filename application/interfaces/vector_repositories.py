import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from domain.models.professor import Professor
from domain.models.project_template import ProjectTemplate
from domain.models.skill import Skill
from domain.models.student import Student


class ISkillVectorRepository(ABC):
    """Defines the abstract vector search contract for Skill records."""

    @abstractmethod
    def __init__(self, table_name: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def upsert(self, skill: Skill, vector: List[float]) -> None:
        pass

    @abstractmethod
    def bulk_upsert(self, skills: List[Skill], vectors: List[List[float]]) -> None:
        """Bulk upserts multiple skill records with their corresponding vectors."""
        pass

    @abstractmethod
    def delete(self, entity_id: uuid.UUID) -> None:
        """Removes a skill vector node from persistent storage by its unique ID."""
        pass

    @abstractmethod
    def find_nearest(self, vector: List[float], limit: int = 10) -> List[uuid.UUID]:
        pass

    @abstractmethod
    def reload_table(self) -> None:
        """Reloads the cached table handle to reflect swapped database state."""
        pass


class IStudentVectorRepository(ABC):
    """Defines the abstract vector search contract for Student profiles."""

    @abstractmethod
    def __init__(self, table_name: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def upsert(self, student: Student, vector: List[float]) -> None:
        """Saves or updates a student profile record with its contextual embedding."""
        pass

    @abstractmethod
    def bulk_upsert(self, students: List[Student], vectors: List[List[float]]) -> None:
        """Bulk upserts multiple student records with their corresponding vectors."""
        pass

    @abstractmethod
    def delete(self, entity_id: uuid.UUID) -> None:
        """Removes a student profile vector node from persistent storage by its unique ID."""
        pass

    @abstractmethod
    def get_by_id(self, student_id: uuid.UUID) -> Optional[Student]:
        """Fetches a flattened student profile to extract metadata tags for query setups."""
        pass

    @abstractmethod
    def reload_table(self) -> None:
        """Reloads the cached table handle to reflect swapped database state."""
        pass


class IProjectTemplateVectorRepository(ABC):
    """Defines the abstract vector search contract for Project Template blueprints."""

    @abstractmethod
    def __init__(self, table_name: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def upsert(self, template: ProjectTemplate, vector: List[float]) -> None:
        """Saves or updates a project template blueprint with its unified descriptive embedding."""
        pass

    @abstractmethod
    def bulk_upsert(
        self, templates: List[ProjectTemplate], vectors: List[List[float]]
    ) -> None:
        """Bulk upserts multiple project templates with their corresponding vectors."""
        pass

    @abstractmethod
    def delete(self, entity_id: uuid.UUID) -> None:
        """Removes a project blueprint vector node from persistent storage by its unique ID."""
        pass

    @abstractmethod
    def find_nearest(
        self,
        vector: List[float],
        filter_expression: Optional[str] = None,
        limit: int = 10,
    ) -> List[uuid.UUID]:
        """
        Executes a vector search against project templates combining vector similarity
        with an optional scalar query string filter (LanceDB SQL syntax style).
        """
        pass

    @abstractmethod
    def reload_table(self) -> None:
        """Reloads the cached table handle to reflect swapped database state."""
        pass


class IProfessorVectorRepository(ABC):
    """Defines the abstract vector search contract for Professor profiles."""

    @abstractmethod
    def __init__(self, table_name: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def upsert(self, professor: Professor, vector: List[float]) -> None:
        """Saves or updates a professor faculty profile with its interest embedding."""
        pass

    @abstractmethod
    def bulk_upsert(
        self, professors: List[Professor], vectors: List[List[float]]
    ) -> None:
        """Bulk upserts multiple professor records with their corresponding vectors."""
        pass

    @abstractmethod
    def delete(self, entity_id: uuid.UUID) -> None:
        """Removes a professor profile vector node from persistent storage by its unique ID."""
        pass

    @abstractmethod
    def find_nearest(
        self,
        vector: List[float],
        filter_expression: Optional[str] = None,
        limit: int = 10,
    ) -> List[uuid.UUID]:
        """
        Executes a vector search against professors combining similarity weights
        with scalar payload pre-filtering parameters.
        """
        pass

    @abstractmethod
    def reload_table(self) -> None:
        """Reloads the cached table handle to reflect swapped database state."""
        pass