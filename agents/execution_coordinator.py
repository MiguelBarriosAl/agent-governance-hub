"""
Execution Coordinator

Coordinates agent execution with governance callbacks.
Separates execution logic from agent orchestration.
"""
from typing import List
import logging

from langchain.agents import AgentExecutor

from agents.callbacks import (
    PolicyEnforcementCallback,
    ObservabilityCallback
)

logger = logging.getLogger(__name__)


class ExecutionCoordinator:
    """
    Coordinates agent execution with governance callbacks.
    
    Responsibilities:
    - Create policy enforcement callback
    - Create observability callback
    - Execute agent with both callbacks
    """
    
    def __init__(self, agent, tools: List):
        """
        Initialize execution coordinator.
        
        Args:
            agent: The parent agent (BaseAgent) for policy evaluation
            tools: List of tools for policy enforcement callback
        """
        self.agent = agent
        self.tools = tools
    
    def execute(self, executor: AgentExecutor, query: str) -> str:
        """
        Execute agent with policy enforcement and observability.
        
        Args:
            executor: Configured AgentExecutor
            query: User query to process
            
        Returns:
            Agent's final answer
        """
        logger.debug("Executing agent with governance callbacks")
        
        # Create separated callbacks
        policy_callback = PolicyEnforcementCallback(self.agent, self.tools)
        observability_callback = ObservabilityCallback(self.agent)
        
        # Execute with both callbacks
        result = executor.invoke(
            {"input": query},
            config={"callbacks": [policy_callback, observability_callback]}
        )
        
        return result.get("output", "No response generated")
