"""
Base Agent

Abstract base class for all governed agents.
Ensures all actions are evaluated against governance policies.
"""
from abc import ABC
from typing import Dict, Any, Optional
from governance.policy_engine import PolicyEngine
from governance.models import EvaluationResult, DecisionType


class BaseAgent(ABC):
    """
    Base class for agents that enforce governance policies.
    
    All agent actions must be evaluated through the policy engine
    before execution.
    """
    
    def __init__(self, name: str, policy_engine: PolicyEngine):
        """
        Initialize the base agent.
        
        Args:
            name: Unique identifier for this agent
            policy_engine: Policy engine instance for evaluating actions
        """
        self.name = name
        self.policy_engine = policy_engine
    
    def evaluate_action(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate an action against governance policies.
        
        Args:
            action: The action to evaluate (e.g., "query_documents")
            context: Optional context data for policy evaluation
            
        Returns:
            EvaluationResult with decision and reasoning
        """
        context = context or {}
        return self.policy_engine.evaluate(
            agent_id=self.name,
            action=action,
            context=context
        )
    