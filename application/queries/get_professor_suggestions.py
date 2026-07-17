import logging
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import IProfessorVectorRepository
from application.services.formatters.project_template import format_professor_advisor_query
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)

logger = logging.getLogger(__name__)


class GetProfessorSuggestionsQuery(BaseModel):
    """
    Query parameters for finding matching faculty advisors for a project.
    Follows the Stateless Searcher pattern by receiving flat textual context directly.
    """
    title: str = Field(..., description="The title of the project template.")
    description: str = Field(..., description="The core scope and description of the project.")
    major_name: Optional[str] = Field(default=None, description="Descriptive name of the aligned major.")
    specialty_name: Optional[str] = Field(default=None, description="Descriptive name of the sub-track concentration.")
    skill_names: List[str] = Field(default_factory=list, description="Labels of required student skills.")
    limit: int = Field(default=10, ge=1, le=100)


class GetProfessorSuggestionsQueryHandler:
    """
    Handles the execution of semantic advisor matchmaking workflows.
    Translates project requirements into advisor-targeted queries and scans the vector space.
    """

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        professor_repository: IProfessorVectorRepository
    ):
        self._embedding_service = embedding_service
        self._professor_repository = professor_repository

    def handle(self, query: GetProfessorSuggestionsQuery) -> List[uuid.UUID]:
        """
        Executes a vector similarity match against faculty professor profiles.

        Raises:
            EmbeddingServiceException: If query vector generation fails.
            VectorRepositoryException: If the underlying vector DB lookup fails.
        """
        logger.info("Executing professor advisor matchmaking query via stateless read-path.")

        # 1. Format the flat details into a targeted advisor semantic search query string
        query_prose = format_professor_advisor_query(
            title=query.title,
            description=query.description,
            major_name=query.major_name,
            specialty_name=query.specialty_name,
            skill_names=query.skill_names
        )

        # 2. Extract semantic search weights using the isolated application port
        try:
            logger.debug("Generating query embedding vector for advisor matching.")
            query_vector = self._embedding_service.embed_query(query_prose)
        except Exception as ex:
            error_msg = "Failed to generate specialized query embedding vector for professor matching."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise EmbeddingServiceException(error_msg) from ex

        # 3. Construct mandatory scalar pre-filters to exclude unavailable advisors
        filter_expression = "is_accepting_projects = true"

        # 4. Perform similarity match using the exact repository interface contract
        try:
            logger.debug(f"Executing find_nearest vector search with filter: {filter_expression}")
            matched_professor_ids = self._professor_repository.find_nearest(
                vector=query_vector,
                filter_expression=filter_expression,
                limit=query.limit
            )
            logger.info(f"Successfully calculated {len(matched_professor_ids)} faculty advisor suggestions.")
            return matched_professor_ids
        except Exception as ex:
            error_msg = "Failed to complete nearest-neighbor professor vector matchmaking."
            logger.error(f"{error_msg} Details: {str(ex)}")
            raise VectorRepositoryException(error_msg) from ex