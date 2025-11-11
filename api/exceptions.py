"""
Exception handlers for the API.

Centralized error handling for the FastAPI application.
"""
import logging
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def generic_exception_handler(request: Request, exc: Exception):
    """
    Generic exception handler for unhandled exceptions.

    Args:
        request: The incoming request
        exc: The exception that was raised

    Returns:
        JSONResponse with error details and 500 status code
    """
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )
