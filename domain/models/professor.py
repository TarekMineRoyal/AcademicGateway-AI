import uuid
from typing import List, Optional
from pydantic import BaseModel, Field


class Professor(BaseModel):
    """
    Lean read-model schema representing a Professor profile within the AI vector space.
    Optimized for contextual matching against incoming student project applications.
    """
    id: uuid.UUID
    full_name: str
    department: str
    rank: str

    # Critical pre-filtering flag to instantly drop unavailable advisors from searches
    is_accepting_projects: bool

    # Flattened representation of the research interest aggregate relations
    research_interest_ids: List[uuid.UUID] = Field(default_factory=list)

    # The dedicated field to catch bio summaries and lab direction text
    about_me: Optional[str] = Field(default=None, description="Faculty bio, current lab focus, or summary text.")