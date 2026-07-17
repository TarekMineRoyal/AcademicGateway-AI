import uuid
from typing import List, Optional
from pydantic import BaseModel, Field


class Student(BaseModel):
    """
    Lean read-model representing a Student profile within the AI vector space.
    Flattens relational join tables into clean identity arrays for high-speed indexing.
    """
    id: uuid.UUID
    full_name: str
    major_id: uuid.UUID
    specialty_ids: List[uuid.UUID] = Field(default_factory=list)
    skill_ids: List[uuid.UUID] = Field(default_factory=list)

    # The goldmine field for our Nomic-powered semantic text chunks
    about_me: Optional[str] = Field(default=None, description="Aspirational summary or biography text.")