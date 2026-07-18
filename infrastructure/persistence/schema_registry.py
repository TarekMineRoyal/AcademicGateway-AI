import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import Field
from lancedb.pydantic import LanceModel, Vector
from infrastructure.config.settings import settings

# Global dimensional anchor for Nomic text embeddings
VECTOR_DIMENSION = settings.EMBEDDING_DIMENSION


class SkillTableSchema(LanceModel):
    """
    Physical LanceDB layout for the global technical skills index.
    """
    id: str = Field(description="Stringified UUID primary lookup key.")
    name: str = Field(description="Raw text label name of the capability.")
    vector: Vector(VECTOR_DIMENSION)


class StudentTableSchema(LanceModel):
    """
    Physical LanceDB layout optimizing complex student profiles, degree context,
    and multi-skill metrics for fast semantic matches.
    """
    id: str = Field(description="Stringified UUID profile identifier.")
    full_name: str
    major_id: str
    specialty_ids: List[str] = Field(default_factory=list)
    skill_ids: List[str] = Field(default_factory=list)
    about_me: Optional[str] = None
    vector: Vector(VECTOR_DIMENSION)


class ProfessorTableSchema(LanceModel):
    """
    Physical LanceDB layout tracking academic faculty members, current capacity status,
    and scientific research alignment metrics.
    """
    id: str = Field(description="Stringified UUID faculty identifier.")
    full_name: str
    department: str
    rank: str
    is_accepting_projects: bool  # Accelerated scalar predicate index destination
    research_interest_ids: List[str] = Field(default_factory=list)
    about_me: Optional[str] = None
    vector: Vector(VECTOR_DIMENSION)


class ProjectTemplateTableSchema(LanceModel):
    """
    Physical LanceDB layout mapping vectorized project blueprints alongside
    relational major, specialty, and competency filtering parameters.
    """
    id: str = Field(description="Stringified UUID blueprint identifier.")
    title: str
    description: str
    provider_id: str
    created_at: datetime
    skill_ids: List[str] = Field(default_factory=list)
    major_id: Optional[str] = None
    specialty_id: Optional[str] = None
    vector: Vector(VECTOR_DIMENSION)