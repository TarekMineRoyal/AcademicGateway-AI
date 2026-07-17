import uuid
from pydantic import BaseModel

class Skill(BaseModel):
    id: uuid.UUID
    name: str