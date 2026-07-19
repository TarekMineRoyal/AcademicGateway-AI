import pytest
from unittest.mock import MagicMock, patch
from fastapi import status
from fastapi.testclient import TestClient

# Component Imports
from api.main import app
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)
from infrastructure.persistence.lancedb_client import lancedb_client

pytestmark = pytest.mark.api


# ============================================================================
# DUMMY TEST ROUTES FOR EXCEPTION TESTING
# ============================================================================
# We temporarily append diagnostic endpoints onto the active app instance 
# to isolate and validate global middleware handlers without testing pass-through shells.
@app.get("/_test/raise-embedding-exception")
async def route_raise_embedding():
    raise EmbeddingServiceException("CUDA out of VRAM pressure on RTX 3050.")


@app.get("/_test/raise-repository-exception")
async def route_raise_repository():
    raise VectorRepositoryException("LanceDB disk file lock contention detected.")


@pytest.fixture
def api_client():
    """Provides a thread-safe FastAPI TestClient wrapper instance."""
    return TestClient(app)


# ============================================================================
# 5. API ERROR BOUNDARIES & HEALTH MONITORING TESTS (Checkpoint 5)
# ============================================================================

def test_health_endpoint_returns_healthy_when_database_connected(api_client):
    """Hit /health and verify database table check logic reports true status configurations."""
    # Arrange
    mock_connection = MagicMock()
    mock_connection.table_names.return_value = ["professors", "students", "skills"]

    with patch.object(lancedb_client, "get_connection", return_value=mock_connection) as mock_get_conn:
        # Act
        response = api_client.get("/health")

        # Assert
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "status": "healthy",
            "database": "connected"
        }
        mock_get_conn.assert_called_once()
        mock_connection.table_names.assert_called_once()


def test_health_endpoint_handles_connection_drops_safely(api_client):
    """Hit /health and verify database drops convert gracefully into service status errors."""
    # Arrange
    with patch.object(lancedb_client, "get_connection", side_effect=Exception("Disk I/O Timeout Failure")):
        # Act
        response = api_client.get("/health")

        # Assert
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.json() == {
            "status": "unhealthy",
            "reason": "Disk I/O Timeout Failure"
        }


def test_global_exception_handler_maps_embedding_failure_to_503(api_client):
    """Force an embedding mockup failure and verify the system outputs HTTP 503 contracts."""
    # Act
    response = api_client.get("/_test/raise-embedding-exception")

    # Assert
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    # Deep payload verification to ensure exact contracts required by the C# application
    json_data = response.json()
    assert "detail" in json_data
    assert json_data["detail"] == "Embedding generation failed. Check local GPU/model status."


def test_global_exception_handler_maps_repository_failure_to_500(api_client):
    """Force a database execution failure and verify the system outputs HTTP 500 contracts."""
    # Act
    response = api_client.get("/_test/raise-repository-exception")

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    # Verify exact JSON formatting context match
    json_data = response.json()
    assert "detail" in json_data
    assert json_data["detail"] == "Vector database execution failed. Transaction aborted."