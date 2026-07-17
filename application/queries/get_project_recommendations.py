import logging
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import IProjectTemplateVectorRepository
from application.services.formatters.student import format_project_recommendation_query
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)

logger = logging.getLogger(__name__)


class GetProjectRecommendationsQuery(BaseModel):
    """
    Query parameters for fetching semantic project recommendations for a student.
    Follows the Stateless Searcher pattern by receiving the textual context directly.
    """
    major_name: str
    specialty_names: List[str] = Field(default_factory=list)
    skill_names: List[str] = Field(default_factory=list)
    about_me: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)

    # Optional filter to restrict recommendations to the student's primary major ID
    restrict_to_major_id: Optional[uuid.UUID] = None


class GetProjectRecommendationsQueryHandler:
    """
    Handles the execution of semantic project matchmaking recommendation workflows.
    Translates student context into search prose, handles embeddings, and executes lookups.
    """

    def __init__(
            self,
            embedding_service: IEmbeddingService,
            project_repository: IProjectTemplateVectorRepository
    ):
        self._embedding_service = embedding_service
        self._project_repository = project_repository

    def handle(self, query: GetProjectRecommendationsQuery) -> List[uuid.UUID]:
        """
        Executes a vector nearest-neighbor match against project blueprints.

        Raises:
            EmbeddingServiceException: If query vector generation fails.
            VectorRepositoryException: If the underlying vector DB execution fails.
        """
        logger.info("Executing project recommendation query via stateless read-path.")

        # 1. Translate the current context parameters into targeted query search prose
        query_prose = format_project_recommendation_query(
            major_name=query.major_name,
            specialty_names=query.specialty_names,
            skill_names=query.skill_names,
            about_me=query.about_me
        )

        # 2. Compute a query-optimized vector map utilizing the dedicated query port
        try:
            logger.debug("Generating semantic query search vector.")
            query_vector = self._embedding_service.embed_query(query_prose)
        except Exception as ex:
            error_msg = "Failed to generate specialized query embedding vector for matchmaking."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise EmbeddingServiceException(error_msg) from ex

        # 3. Construct LanceDB SQL-style scalar filters to strip invalid records
        # Enforce that only 'Approved' templates are considered for matching
        filter_conditions = ["status = 'Approved'"]

        if query.restrict_to_major_id:
            filter_conditions.append(f"major_id = '{str(query.restrict_to_major_id)}'")

        filter_expression = " AND ".join(filter_conditions)

        # 4. Query nearest neighbor matches matching the exact interface tracking signature
        try:
            logger.debug(f"Executing find_nearest vector scan with filter: {filter_expression}")
            matched_project_ids = self._project_repository.find_nearest(
                vector=query_vector,
                filter_expression=filter_expression,
                limit=query.limit
            )
            logger.info(f"Successfully calculated {len(matched_project_ids)} project suggestions.")
            return matched_project_ids
        except Exception as ex:
            error_msg = "Failed to complete nearest-neighbor project template vector matching."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex