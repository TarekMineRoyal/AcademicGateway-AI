import uuid
from pydantic import BaseModel

from domain.models.skill import Skill
from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import ISkillVectorRepository
from application.services.formatters.skill import format_skill_document


class SyncSkillCommand(BaseModel):
    """
    Data Transfer Object representing the payload sent by the C# backend
    whenever a global competency skill is created or modified.
    """
    id: uuid.UUID
    name: str


class SyncSkillHandler:
    """
    Handles the synchronization transaction lifecycle for an individual skill asset.
    """

    def __init__(self, embedder: IEmbeddingService, repository: ISkillVectorRepository):
        """
        Injects the abstract decoupled infrastructure layer boundaries.
        """
        self._embedder = embedder
        self._repository = repository

    def handle(self, command: SyncSkillCommand) -> None:
        """
        Executes the business synchronization pipeline for the skill entity.

        Args:
            command (SyncSkillCommand): The inbound mutation tracking criteria.
        """
        # 1. Instantiate the read-model domain entity state
        skill = Skill(id=command.id, name=command.name)

        # 2. Compile entity attributes into a clean prose paragraph text block
        narrative_document = format_skill_document(skill)

        # 3. Generate the mathematical vector (Infrastructure injects Nomic prefixes)
        vector = self._embedder.embed_document(narrative_document)

        # 4. Commit the structural record data and float array onto the storage index
        self._repository.upsert(skill, vector)