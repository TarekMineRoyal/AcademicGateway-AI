import uuid
from pydantic import BaseModel

class Major(BaseModel):
    id: uuid.UUID
    name: str