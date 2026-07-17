import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ProjectTemplate(BaseModel):
    """
    Lean read-model schema representing a Project Template blueprint.
    Strips away workflow state-machines, DAG validation loops, and factory logic
    to serve as a pure vectorized node for matchmaking.
    """
    id: uuid.UUID
    title: str
    description: str
    provider_id: uuid.UUID
    created_at: datetime

    # Flattened targets for fast vector space filtering
    skill_ids: List[uuid.UUID] = Field(default_factory=list)
    major_id: Optional[uuid.UUID] = Field(default=None, description="Structural filter target major alignment.")
    specialty_id: Optional[uuid.UUID] = Field(default=None, description="Structural filter target specialty alignment.")