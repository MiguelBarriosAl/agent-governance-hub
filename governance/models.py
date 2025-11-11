"""
Governance Models

Data structures for policies, rules, and decisions.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


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
        id: Unique identifier for traceability (required)
        action: The action this rule applies to (e.g., "query_database")
        decision: What to do when this rule matches
        conditions: Conditions that must be met (required, can be empty dict)
        reason: Human-readable explanation for this rule (required)
    """
    id: str
    action: str
    decision: DecisionType
    conditions: Dict[str, Any]
    reason: str


class Policy(BaseModel):
    """
    Represents a governance policy for a specific agent.

    Fields:
        agent_id: Identifier of the agent this policy applies to
        description: Human readable description of the policy (required)
        rules: List of rules to evaluate for this agent (required)
        source_file: Source filename (set by loader after parsing)
    """
    agent_id: str
    description: str
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
