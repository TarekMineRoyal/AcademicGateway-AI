import logging
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import ISkillVectorRepository
from application.services.formatters.student import format_skill_recommendation_query
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)

logger = logging.getLogger(__name__)


class GetSkillRecommendationsQuery(BaseModel):
    """
    Query parameters for discovering adjacent or relevant skills for a student.
    Uses the Stateless Searcher pattern to accept contextual properties directly.
    """
    major_name: str
    specialty_names: List[str] = Field(default_factory=list)
    skill_names: List[str] = Field(default_factory=list)
    about_me: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)


class GetSkillRecommendationsQueryHandler:
    """
    Handles the execution of semantic skill discovery workflows.
    Translates active skills and tracks into a vector space search parameter.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        skill_repository: ISkillVectorRepository
    ):
        self._embedding_service = embedding_service
        self._skill_repository = skill_repository

    def handle(self, query: GetSkillRecommendationsQuery) -> List[uuid.UUID]:
        """
        Executes a vector similarity scan to fetch missing or next-step skill recommendations.

        Raises:
            EmbeddingServiceException: If query vector generation fails.
            VectorRepositoryException: If the underlying vector DB execution fails.
        """
        logger.info(
            f"Executing skill recommendation discovery query for major='{query.major_name}' "
            f"(limit={query.limit}, skills_count={len(query.skill_names)})"
        )

        # 1. Map existing contextual coordinates into a target progression query prose string
        query_prose = format_skill_recommendation_query(
            major_name=query.major_name,
            specialty_names=query.specialty_names,
            skill_names=query.skill_names,
            about_me=query.about_me
        )

        # 2. Extract semantic query vectors through the un-prefixed application port
        try:
            logger.debug("Generating query embedding vector for skill recommendation matrix.")
            query_vector = self._embedding_service.embed_query(query_prose)
        except Exception as ex:
            error_msg = "Failed to compute specialized query embedding vector for skill discovery."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise EmbeddingServiceException(error_msg) from ex

        # 3. Match nearest concept tracking IDs matching the exact interface signature
        try:
            logger.debug(f"Executing find_nearest vector lookups for skill nodes up to limit: {query.limit}")
            matched_skill_ids = self._skill_repository.find_nearest(
                vector=query_vector,
                limit=query.limit
            )
            logger.info(f"Successfully discovered {len(matched_skill_ids)} unique skill recommendations.")
            return matched_skill_ids
        except Exception as ex:
            error_msg = "Failed to complete nearest-neighbor skill vector matching."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex