from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import logging

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add a unique request ID to each request"""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        logger.info(
            f"Request started: {request.method} {request.url} - Request ID: {request_id}"
        )
        response = await call_next(request)
        logger.info(
            f"Request completed: {request.method} {request.url} - Request ID: {request_id}"
        )
        return response
