import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from domain.models.skill import Skill
from domain.models.student import Student
from domain.models.project_template import ProjectTemplate
from domain.models.professor import Professor


class ISkillVectorRepository(ABC):
    """Defines the abstract vector search contract for Skill records."""

    @abstractmethod
    def upsert(self, skill: Skill, vector: List[float]) -> None:
        pass

    @abstractmethod
    def find_nearest(self, vector: List[float], limit: int = 10) -> List[uuid.UUID]:
        pass


class IStudentVectorRepository(ABC):
    """Defines the abstract vector search contract for Student profiles."""

    @abstractmethod
    def upsert(self, student: Student, vector: List[float]) -> None:
        """Saves or updates a student profile record with its contextual embedding."""
        pass

    @abstractmethod
    def get_by_id(self, student_id: uuid.UUID) -> Optional[Student]:
        """Fetches a flattened student profile to extract metadata tags for query setups."""
        pass


class IProjectTemplateVectorRepository(ABC):
    """Defines the abstract vector search contract for Project Template blueprints."""

    @abstractmethod
    def upsert(self, template: ProjectTemplate, vector: List[float]) -> None:
        """Saves or updates a project template blueprint with its unified descriptive embedding."""
        pass

    @abstractmethod
    def find_nearest(
        self,
        vector: List[float],
        filter_expression: Optional[str] = None,
        limit: int = 10
    ) -> List[uuid.UUID]:
        """
        Executes a vector search against project templates combining vector similarity
        with an optional scalar query string filter (LanceDB SQL syntax style).
        """
        pass


class IProfessorVectorRepository(ABC):
    """Defines the abstract vector search contract for Professor profiles."""

    @abstractmethod
    def upsert(self, professor: Professor, vector: List[float]) -> None:
        """Saves or updates a professor faculty profile with its interest embedding."""
        pass

    @abstractmethod
    def find_nearest(
        self,
        vector: List[float],
        filter_expression: Optional[str] = None,
        limit: int = 10
    ) -> List[uuid.UUID]:
        """
        Executes a vector search against professors combining similarity weights
        with scalar payload pre-filtering parameters.
        """
        pass