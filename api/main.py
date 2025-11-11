"""
Agent Governance Hub - Main Application

A lightweight, open-source governance middleware for LLM agents.
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import settings
from governance.policy_loader import PolicyLoader, PolicyLoadError
from governance.policy_engine import PolicyEngine
from governance.models import EvaluationResult

app = FastAPI(
    title="Agent Governance Hub",
    description="A lightweight governance middleware for LLM agents",
    version="0.1.0",
)

# Global policy engine instance
policy_engine: Optional[PolicyEngine] = None


class EvaluationRequest(BaseModel):
    """Request model for policy evaluation."""
    agent_id: str
    action: str
    context: Dict[str, Any] = {}


@app.on_event("startup")
async def startup_event():
    """
    Initialize the policy engine on application startup.
    Loads policies from the configured policy directory.
    """
    global policy_engine
    
    try:
        loader = PolicyLoader(settings.policy_dir)
        policies = loader.load_all_policies()
        policy_engine = PolicyEngine(policies)
        print(f"✓ Loaded {len(policies)} policies from {settings.policy_dir}")
    except PolicyLoadError as e:
        print(f"✗ Failed to load policies: {e}")
        raise


@app.get("/health")
async def health_check():
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


@app.post("/governance/evaluate", response_model=EvaluationResult)
async def evaluate_policy(request: EvaluationRequest) -> EvaluationResult:
    """
    Evaluate an agent action against governance policies.
    
    Args:
        request: Evaluation request with agent_id, action, and context
        
    Returns:
        EvaluationResult with decision and reasoning
        
    Raises:
        HTTPException: If policy engine is not initialized
    """
    if policy_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Policy engine not initialized"
        )
    
    result = policy_engine.evaluate(
        agent_id=request.agent_id,
        action=request.action,
        context=request.context
    )
    
    return result
