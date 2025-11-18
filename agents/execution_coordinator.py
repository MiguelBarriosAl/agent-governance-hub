"""
Execution Coordinator

Coordinates agent execution with governance callbacks.
Separates execution logic from agent orchestration.
"""
from typing import List, Dict, Any
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
    - Track tool usage for observability
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
    
    def execute(
        self, executor: AgentExecutor, query: str
    ) -> Dict[str, Any]:
        """
        Execute agent with policy enforcement and observability.
        
        Args:
            executor: Configured AgentExecutor
            query: User query to process
            
        Returns:
            Dict with answer and execution metadata (tools used)
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
        
        answer = result.get("output", "No response generated")
        
        # Check intermediate steps to see if tools were used
        intermediate_steps = result.get("intermediate_steps", [])
        logger.debug("Intermediate steps count: %d", len(intermediate_steps))
        
        tools_used = []
        for step in intermediate_steps:
            if step and len(step) > 0:
                action = step[0]
                logger.debug("Step action type: %s", type(action))
                if hasattr(action, 'tool'):
                    tools_used.append(action.tool)
                    logger.debug("Tool used: %s", action.tool)
        
        used_rag = "vector_retrieval" in tools_used
        logger.debug("Tools used: %s, used_rag: %s", tools_used, used_rag)
        
        return {
            "answer": answer,
            "tools_used": tools_used,
            "used_rag": used_rag
        }
