import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from fastapi import Request, Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware to track HTTP request lifecycles, execution duration,
    status codes, and trace context via X-Request-ID.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()

        # Retrieve existing request ID from headers or generate a new UUID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        try:
            response = await call_next(request)
        except Exception as exc:
            process_time = (time.perf_counter() - start_time) * 1000
            logger.exception(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(process_time, 2),
                }
            )
            raise exc

        process_time = (time.perf_counter() - start_time) * 1000

        # Attach X-Request-ID to the outgoing response headers
        response.headers["X-Request-ID"] = request_id

        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} ({process_time:.2f}ms)",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(process_time, 2),
            }
        )

        return response