"""
Agent Governance Hub - Main Application

A lightweight, open-source governance middleware for LLM agents.
"""
import logging
import uvicorn

from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from config.settings import settings
from governance.policy_loader import PolicyLoader, PolicyLoadError
from governance.policy_engine import PolicyEngine
from governance.models import EvaluationResult

logger = logging.getLogger(__name__)


class EvaluationRequest(BaseModel):
    """Request model for policy evaluation."""
    agent_id: str
    action: str
    context: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the policy engine on application startup.
    Loads policies from the configured policy directory.
    """
    try:
        loader = PolicyLoader(settings.policy_dir)
        policies = loader.load_all_policies()
        app.state.policy_engine = PolicyEngine(policies)
        logger.info(
            "Loaded %d policies from %s", len(policies), settings.policy_dir
        )
    except PolicyLoadError as e:
        logger.error("Failed to load policies: %s", e)

    yield
    app.state.policy_engine = None
    logger.info("PolicyEngine shutdown complete.")


async def health():
    """
    Health check endpoint to verify the service is running.

    Returns:
        dict: Service health status
    """
    return {
        "status": "healthy",
        "service": "agent-governance-hub",
        "version": "0.1.0"
    }


async def generic_exception_handler(exc: Exception):
    """
    Generic exception handler for unhandled exceptions.

    Args:
        request: The incoming request
        exc: The exception that was raised

    Returns:
        JSONResponse with error details
    """
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )


def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="Agent Governance Hub",
        description="A lightweight governance middleware for LLM agents",
        version="0.1.0",
        lifespan=lifespan
    )

    # Register health check endpoint
    app.add_api_route("/health", health, methods=["GET"])

    # Define evaluate_policy within create_app to access app.state
    async def evaluate_policy_endpoint(
        request: EvaluationRequest
    ) -> EvaluationResult:
        """
        Evaluate an agent action against governance policies.

        Args:
            request: Evaluation request with agent_id, action, and context

        Returns:
            EvaluationResult with decision and reasoning

        Raises:
            HTTPException: If policy engine is not initialized
        """
        if app.state.policy_engine is None:
            raise HTTPException(
                status_code=503,
                detail="Policy engine not initialized"
            )

        result = app.state.policy_engine.evaluate(
            agent_id=request.agent_id,
            action=request.action,
            context=request.context
        )

        return result

    # Register governance evaluation endpoint
    app.add_api_route(
        "/governance/evaluate",
        evaluate_policy_endpoint,
        methods=["POST"],
        response_model=EvaluationResult
    )

    # Register global exception handler
    app.add_exception_handler(Exception, generic_exception_handler)

    # TODO: add routers from routes/ in future
    # app.include_router(
    #     agents_router, prefix="/agents", tags=["agents"]
    # )
    # app.include_router(
    #     policies_router, prefix="/policies", tags=["policies"]
    # )

    return app


# For backward compatibility and direct uvicorn usage
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "api.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
