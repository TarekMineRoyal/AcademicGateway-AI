import uuid
from pydantic import BaseModel

class Specialty(BaseModel):
    id: uuid.UUID
    name: str
    major_id: uuid.UUID