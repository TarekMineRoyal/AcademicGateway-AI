import uuid
from pydantic import BaseModel

class ResearchInterest(BaseModel):
    """
    Lean read-model schema representing a global academic Research Interest topic.
    Serves as a standardized text area lookup index for profiling faculty expertise.
    """
    id: uuid.UUID
    area: str