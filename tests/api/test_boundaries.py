from unittest.mock import MagicMock, patch
import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Component Imports
from api.dependencies import (
    get_bulk_sync_professor_handler,
    get_bulk_sync_project_handler,
    get_bulk_sync_skill_handler,
    get_bulk_sync_student_handler,
)
from api.main import app
from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)
from infrastructure.persistence.lancedb_client import lancedb_client

import uuid
from api.dependencies import (
    get_delete_professor_handler,
    get_delete_project_handler,
    get_delete_skill_handler,
    get_delete_student_handler,
)

pytestmark = pytest.mark.api


# ============================================================================
# DUMMY TEST ROUTES FOR EXCEPTION TESTING
# ============================================================================
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
# 5. API ERROR BOUNDARIES & HEALTH MONITORING TESTS
# ============================================================================


def test_health_endpoint_returns_healthy_when_database_connected(api_client):
    """Hit /health and verify database table check logic reports true status configurations."""
    mock_connection = MagicMock()
    mock_connection.table_names.return_value = ["professors", "students", "skills"]

    with patch.object(
        lancedb_client, "get_connection", return_value=mock_connection
    ) as mock_get_conn:
        response = api_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "healthy", "database": "connected"}
        mock_get_conn.assert_called_once()
        mock_connection.table_names.assert_called_once()


def test_health_endpoint_handles_connection_drops_safely(api_client):
    """Hit /health and verify database drops convert gracefully into service status errors."""
    with patch.object(
        lancedb_client,
        "get_connection",
        side_effect=Exception("Disk I/O Timeout Failure"),
    ):
        response = api_client.get("/health")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.json() == {
            "status": "unhealthy",
            "reason": "Disk I/O Timeout Failure",
        }


def test_global_exception_handler_maps_embedding_failure_to_503(api_client):
    """Force an embedding mockup failure and verify the system outputs HTTP 503 contracts."""
    response = api_client.get("/_test/raise-embedding-exception")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    json_data = response.json()
    assert "detail" in json_data
    assert (
        json_data["detail"]
        == "Embedding generation failed. Check local GPU/model status."
    )


def test_global_exception_handler_maps_repository_failure_to_500(api_client):
    """Force a database execution failure and verify the system outputs HTTP 500 contracts."""
    response = api_client.get("/_test/raise-repository-exception")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    json_data = response.json()
    assert "detail" in json_data
    assert json_data["detail"] == "Vector database execution failed. Transaction aborted."


# ============================================================================
# BULK SYNCHRONIZATION API BOUNDARY TESTS
# ============================================================================


@pytest.mark.parametrize(
    "endpoint, dependency_func",
    [
        ("/api/v1/sync/bulk/student", get_bulk_sync_student_handler),
        ("/api/v1/sync/bulk/project", get_bulk_sync_project_handler),
        ("/api/v1/sync/bulk/professor", get_bulk_sync_professor_handler),
        ("/api/v1/sync/bulk/skill", get_bulk_sync_skill_handler),
    ],
)
def test_bulk_sync_endpoints_return_202_accepted(
    api_client, endpoint, dependency_func
):
    """Verify that all /api/v1/sync/bulk/* endpoints accept payloads and immediately respond with 202 Accepted."""
    mock_handler = MagicMock()
    app.dependency_overrides[dependency_func] = lambda: mock_handler

    try:
        response = api_client.post(endpoint, json={"items": []})

        assert response.status_code == status.HTTP_202_ACCEPTED
        json_data = response.json()
        assert json_data["status"] == "accepted"
        assert "message" in json_data
    finally:
        app.dependency_overrides.clear()

# ============================================================================
# DELETION API BOUNDARY TESTS
# ============================================================================


@pytest.mark.parametrize(
    "endpoint_template, dependency_func",
    [
        ("/api/v1/sync/student/{id}", get_delete_student_handler),
        ("/api/v1/sync/project/{id}", get_delete_project_handler),
        ("/api/v1/sync/professor/{id}", get_delete_professor_handler),
        ("/api/v1/sync/skill/{id}", get_delete_skill_handler),
    ],
)
def test_delete_endpoints_return_200_ok(
    api_client, endpoint_template, dependency_func
):
    """Verify that all DELETE /api/v1/sync/*/{id} endpoints execute and return HTTP 200 OK."""
    mock_handler = MagicMock()
    app.dependency_overrides[dependency_func] = lambda: mock_handler

    target_id = uuid.uuid4()
    endpoint = endpoint_template.format(id=target_id)

    try:
        response = api_client.delete(endpoint)

        assert response.status_code == status.HTTP_200_OK
        json_data = response.json()
        assert "message" in json_data
        assert str(target_id) in json_data["message"]
        assert mock_handler.handle.call_count == 1
    finally:
        app.dependency_overrides.clear()