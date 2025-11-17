"""
Agent Callbacks

Separated concerns: policy enforcement and observability.
"""
from typing import Dict, Any, List
import logging
import time

from langchain.callbacks.base import BaseCallbackHandler
from governance.models import DecisionType

logger = logging.getLogger(__name__)


class PolicyEnforcementCallback(BaseCallbackHandler):
    """
    Callback that enforces governance policies on tool executions.
    
    Intercepts tool calls and validates them against the policy engine
    before execution. Raises exception if action is blocked.
    """
    
    def __init__(self, agent_ref, tools_list):
        """
        Initialize policy enforcement callback.
        
        Args:
            agent_ref: Reference to the agent (for policy evaluation)
            tools_list: List of tools (to get policy_action metadata)
        """
        super().__init__()
        self.agent = agent_ref
        self.tools_map = {tool.name: tool for tool in tools_list}
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ):
        """
        Intercept tool execution and validate against policies.
        
        Gets the policy_action from tool metadata dynamically.
        If tool has no policy_action, execution proceeds without check.
        """
        tool_name = serialized.get("name", "")
        
        # Get tool from map
        tool = self.tools_map.get(tool_name)
        
        if tool is None:
            # Tool not found, allow execution
            return
        
        # Get policy action from tool metadata
        policy_action = getattr(tool, "policy_action", None)
        
        if policy_action is None:
            # Tool has no governance metadata, allow execution
            return
        
        # Evaluate action against policy
        evaluation = self.agent.evaluate_action(
            action=policy_action,
            context={"query": input_str}
        )
        
        # Block if policy denies
        if evaluation.decision == DecisionType.BLOCK:
            raise Exception(
                f"Action blocked by policy: {evaluation.reason} "
                f"(rule: {evaluation.rule_id})"
            )


class ObservabilityCallback(BaseCallbackHandler):
    """
    Callback for logging and observability of agent behavior.
    
    Records LLM reasoning, tool usage, timing, and agent decisions
    for traceability and debugging.
    """
    
    def __init__(self, agent_ref):
        """
        Initialize observability callback.
        
        Args:
            agent_ref: Reference to the agent (for context in logs)
        """
        super().__init__()
        self.agent = agent_ref
        self.tool_start_time = None
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs
    ):
        """Log when LLM reasoning starts."""
        logger.info(
            "LLM reasoning started",
            extra={
                "agent": self.agent.name,
                "model": self.agent.llm.model_name,
                "prompts_count": len(prompts)
            }
        )
    
    def on_llm_end(self, response, **kwargs):
        """Log when LLM reasoning completes."""
        logger.info(
            "LLM reasoning completed",
            extra={
                "agent": self.agent.name,
                "generations": len(response.generations)
            }
        )
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ):
        """Log tool execution start."""
        tool_name = serialized.get("name", "unknown")
        self.tool_start_time = time.time()
        
        logger.info(
            "Tool execution requested by LLM",
            extra={
                "agent": self.agent.name,
                "tool": tool_name,
                "input": input_str[:100]
            }
        )
    
    def on_tool_end(self, output: str, **kwargs):
        """Log tool execution completion with timing."""
        elapsed = (
            time.time() - self.tool_start_time
            if self.tool_start_time else 0
        )
        
        logger.info(
            "Tool execution completed",
            extra={
                "agent": self.agent.name,
                "elapsed_ms": round(elapsed * 1000, 2),
                "output_length": len(output)
            }
        )
    
    def on_tool_error(self, error, **kwargs):
        """Log tool execution errors."""
        logger.error(
            "Tool execution failed",
            extra={
                "agent": self.agent.name,
                "error": str(error)
            }
        )
    
    def on_agent_action(self, action, **kwargs):
        """Log agent action decisions."""
        logger.info(
            "Agent action decided",
            extra={
                "agent": self.agent.name,
                "tool": action.tool,
                "tool_input": str(action.tool_input)[:100]
            }
        )
    
    def on_agent_finish(self, finish, **kwargs):
        """Log agent completion."""
        logger.info(
            "Agent finished reasoning",
            extra={
                "agent": self.agent.name,
                "output_length": len(finish.return_values.get("output", ""))
            }
        )
