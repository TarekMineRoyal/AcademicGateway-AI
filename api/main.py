import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from application.exceptions.application_exceptions import (
    EmbeddingServiceException,
    VectorRepositoryException,
)
from infrastructure.persistence.lancedb_client import lancedb_client
from api.routers import sync_router, search_router

# Configure infrastructure logging layout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vector Matchmaking Engine",
    description="High-performance LanceDB semantic vector indexing microservice.",
    version="1.0.0",
)


# ==============================================================================
# GLOBAL EXCEPTION MAPPERS (Translates CQRS Exceptions to Clean HTTP Responses)
# ==============================================================================
@app.exception_handler(EmbeddingServiceException)
async def embedding_exception_handler(request: Request, exc: EmbeddingServiceException):
    logger.error(f"HTTP 503 - Embedding Pipeline Failure: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Embedding generation failed. Check local GPU/model status."},
    )


@app.exception_handler(VectorRepositoryException)
async def repository_exception_handler(request: Request, exc: VectorRepositoryException):
    logger.error(f"HTTP 500 - Vector Database Storage Failure: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Vector database execution failed. Transaction aborted."},
    )


# ==============================================================================
# ROUTE REGISTRATION & MOUNTING
# ==============================================================================
# Mounts Route Categories 1 & 3: Database Synchronization Operations
app.include_router(sync_router.router, prefix="/api/v1/sync", tags=["Synchronization"])

# Mounts Route Category 2: Semantic Recommendation Searches
app.include_router(search_router.router, prefix="/api/v1/search", tags=["Semantic Search"])


# ==============================================================================
# LIVE HEALTH MONITORING ENDPOINT
# ==============================================================================
@app.get("/health", status_code=status.HTTP_200_OK, tags=["Infrastructure"])
def health_check():
    """
    Verifies the operational status of the localized LanceDB engine connection.
    Used by C# orchestrators or container checkers for lifecycle management.
    """
    try:
        conn = lancedb_client.get_connection()
        # Ping the underlying storage layout safely
        conn.table_names()
        return {"status": "healthy", "database": "connected"}
    except Exception as ex:
        logger.critical(f"Health check failed! Storage engine offline: {str(ex)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "reason": str(ex)},
        )