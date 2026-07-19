import pytest
import uuid
from unittest.mock import MagicMock

# Exception and Interface Imports
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)
from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import (
    IProfessorVectorRepository,
    IProjectTemplateVectorRepository,
    ISkillVectorRepository,
)

# Handler and Query Imports
from application.queries.get_professor_suggestions import (
    GetProfessorSuggestionsQuery,
    GetProfessorSuggestionsQueryHandler,
)
from application.queries.get_project_recommendations import (
    GetProjectRecommendationsQuery,
    GetProjectRecommendationsQueryHandler,
)
from application.queries.get_skill_recommendations import (
    GetSkillRecommendationsQuery,
    GetSkillRecommendationsQueryHandler,
)

pytestmark = pytest.mark.unit


# ============================================================================
# 1. PROFESSOR SUGGESTIONS HANDLER TESTS (Checkpoint 2)
# ============================================================================

def test_get_professor_suggestions_handler_success():
    """Assert that the professor advisor query executes successfully with fixed predicate filters."""
    # Arrange
    mock_embedding_service = MagicMock(spec=IEmbeddingService)
    mock_professor_repo = MagicMock(spec=IProfessorVectorRepository)

    fake_vector = [0.1, 0.2, 0.3]
    fake_ids = [uuid.uuid4(), uuid.uuid4()]

    mock_embedding_service.embed_query.return_value = fake_vector
    mock_professor_repo.find_nearest.return_value = fake_ids

    handler = GetProfessorSuggestionsQueryHandler(mock_embedding_service, mock_professor_repo)
    query = GetProfessorSuggestionsQuery(
        title="Distributed Systems Simulation",
        description="Testing consensus engines under network partitions.",
        major_name="Computer Science",
        limit=5
    )

    # Act
    result = handler.handle(query)

    # Assert
    assert result == fake_ids
    # Verify exact scalar logic: Must enforce availability filter string
    mock_professor_repo.find_nearest.assert_called_once_with(
        vector=fake_vector,
        filter_expression="is_accepting_projects = true",
        limit=5
    )


def test_get_professor_suggestions_handler_embedding_failure_mapping():
    """Verify that internal service drops map safely into custom application exception tokens."""
    mock_embedding_service = MagicMock(spec=IEmbeddingService)
    mock_professor_repo = MagicMock(spec=IProfessorVectorRepository)

    mock_embedding_service.embed_query.side_effect = Exception("CUDA out of memory")

    handler = GetProfessorSuggestionsQueryHandler(mock_embedding_service, mock_professor_repo)
    query = GetProfessorSuggestionsQuery(title="X", description="Y")

    with pytest.raises(EmbeddingServiceException) as exc_info:
        handler.handle(query)

    assert "Failed to generate specialized query embedding vector" in str(exc_info.value)


def test_get_professor_suggestions_handler_repository_failure_mapping():
    """Verify repository infrastructure errors are normalized into high-level boundary contracts."""
    mock_embedding_service = MagicMock(spec=IEmbeddingService)
    mock_professor_repo = MagicMock(spec=IProfessorVectorRepository)

    mock_embedding_service.embed_query.return_value = [0.1]
    mock_professor_repo.find_nearest.side_effect = Exception("LanceDB file lock contention")

    handler = GetProfessorSuggestionsQueryHandler(mock_embedding_service, mock_professor_repo)
    query = GetProfessorSuggestionsQuery(title="X", description="Y")

    with pytest.raises(VectorRepositoryException) as exc_info:
        handler.handle(query)

    assert "Failed to complete nearest-neighbor professor vector matchmaking" in str(exc_info.value)


# ============================================================================
# 2. PROJECT RECOMMENDATIONS HANDLER TESTS (Checkpoint 2)
# ============================================================================

def test_get_project_recommendations_handler_with_major_restriction():
    """Assert conditional SQL predicate expressions build cleanly when major constraints exist."""
    mock_embedding_service = MagicMock(spec=IEmbeddingService)
    mock_project_repo = MagicMock(spec=IProjectTemplateVectorRepository)

    fake_vector = [0.9, 0.8]
    target_major_id = uuid.uuid4()

    mock_embedding_service.embed_query.return_value = fake_vector
    mock_project_repo.find_nearest.return_value = []

    handler = GetProjectRecommendationsQueryHandler(mock_embedding_service, mock_project_repo)
    query = GetProjectRecommendationsQuery(
        major_name="Electrical Engineering",
        restrict_to_major_id=target_major_id,
        limit=10
    )

    # Act
    handler.handle(query)

    # Assert
    # Verify exact SQL syntax interpolation to prevent system crashes
    expected_filter = f"major_id = '{str(target_major_id)}'"
    mock_project_repo.find_nearest.assert_called_once_with(
        vector=fake_vector,
        filter_expression=expected_filter,
        limit=10
    )


def test_get_project_recommendations_handler_without_major_restriction():
    """Confirm the dynamic compiler evaluates filter expressions to None when parameters are missing."""
    mock_embedding_service = MagicMock(spec=IEmbeddingService)
    mock_project_repo = MagicMock(spec=IProjectTemplateVectorRepository)

    mock_embedding_service.embed_query.return_value = [0.9]
    mock_project_repo.find_nearest.return_value = []

    handler = GetProjectRecommendationsQueryHandler(mock_embedding_service, mock_project_repo)
    query = GetProjectRecommendationsQuery(
        major_name="Electrical Engineering",
        restrict_to_major_id=None  # Explicitly omitted constraint
    )

    # Act
    handler.handle(query)

    # Assert
    mock_project_repo.find_nearest.assert_called_once_with(
        vector=[0.9],
        filter_expression=None,
        limit=10
    )


# ============================================================================
# 3. SKILL RECOMMENDATIONS HANDLER TESTS
# ============================================================================

def test_get_skill_recommendations_handler_success():
    """Verify linear procedural workflow execution for matching adjacent structural skill nodes."""
    mock_embedding_service = MagicMock(spec=IEmbeddingService)
    mock_skill_repo = MagicMock(spec=ISkillVectorRepository)

    fake_vector = [0.5, 0.5]
    fake_skill_ids = [uuid.uuid4()]

    mock_embedding_service.embed_query.return_value = fake_vector
    mock_skill_repo.find_nearest.return_value = fake_skill_ids

    handler = GetSkillRecommendationsQueryHandler(mock_embedding_service, mock_skill_repo)
    query = GetSkillRecommendationsQuery(
        major_name="Chemistry",
        skill_names=["Lab Safety", "Spectroscopy"],
        limit=15
    )

    # Act
    result = handler.handle(query)

    # Assert
    assert result == fake_skill_ids
    mock_skill_repo.find_nearest.assert_called_once_with(
        vector=fake_vector,
        limit=15
    )