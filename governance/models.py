"""
Governance Models

Data structures for policies, rules, and decisions.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Possible decisions for a policy rule."""
    ALLOW = "allow"
    BLOCK = "block"
    VERIFY = "verify"
    FLAG = "flag"


class Rule(BaseModel):
    """
    Represents a single governance rule within a policy.

    Fields:
        id: Optional identifier for traceability (can be set in YAML)
        action: The action this rule applies to (e.g., "query_database")
        decision: What to do when this rule matches
        conditions: Optional conditions that must be met (defaults to {})
        reason: Human-readable explanation for this rule
    """
    id: Optional[str] = None
    action: str
    decision: DecisionType
    conditions: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None


class Policy(BaseModel):
    """
    Represents a governance policy for a specific agent.

    Fields:
        agent_id: Identifier of the agent this policy applies to
        description: Optional human readable description of the policy
        rules: List of rules to evaluate for this agent
        source_file: Optional source filename where the policy was defined
    """
    agent_id: str
    description: Optional[str] = None
    rules: List[Rule]
    source_file: Optional[str] = None


class PolicySet(BaseModel):
    """
    Container for multiple policies with version information.
    
    Attributes:
        version: Schema version of the policy file
        policies: List of policies for different agents
    """
    version: str
    policies: List[Policy]


class PolicyResult(BaseModel):
    """
    Structured result of a policy evaluation for tracing and metrics.
    """
    decision: DecisionType
    reason: str
    agent_id: str
    action: str
    policy_matched: Optional[str] = None
    rule_id: Optional[str] = None


class EvaluationResult(PolicyResult):
    """Backward-compatible alias for PolicyResult (same fields).
    Kept for compatibility with existing code that imports EvaluationResult.
    """
    # intentionally same as PolicyResult
