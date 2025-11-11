"""
Agent Governance Hub - Main Application

A lightweight, open-source governance middleware for LLM agents.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Agent Governance Hub",
    description="A lightweight governance middleware for LLM agents",
    version="0.1.0",
)


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
