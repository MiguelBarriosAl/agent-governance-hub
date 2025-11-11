"""
Policy Engine

Core evaluation logic for governance policies.
"""

from typing import List, Dict, Any, Optional
from .models import Policy, Rule, DecisionType, EvaluationResult


class PolicyEngine:
    """
    Evaluates agent actions against loaded governance policies.
    
    Uses first-match strategy: the first rule that matches determines
    the decision.
    """
    
    def __init__(
        self,
        policies: List[Policy],
        default_decision: DecisionType = DecisionType.ALLOW,
    ):
        """
        Initialize the policy engine with a list of policies.
        
        Args:
            policies: List of Policy objects to evaluate against
        """
        self.policies = policies
        self._policy_map = {policy.agent_id: policy for policy in policies}
        self.default_decision = default_decision
    
    def evaluate(
        self,
        agent_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate an agent action against governance policies.
        
        Args:
            agent_id: Identifier of the agent performing the action
            action: The action being performed
            context: Optional context data for condition evaluation
            
        Returns:
            EvaluationResult with decision and reasoning
        """
        context = context or {}
        
        # Find policy for this agent
        policy = self._policy_map.get(agent_id)
        
        if not policy:
            # No policy found - return default decision with traceable reason
            return EvaluationResult(
                decision=self.default_decision,
                reason=(
                    f"No policy found for agent '{agent_id}'. "
                    f"Applying default decision: {self.default_decision.value}"
                ),
                policy_matched=None,
                agent_id=agent_id,
                action=action,
                rule_id=None
            )
        
        # Evaluate rules in order (first match wins)
        for rule in policy.rules:
            if self._rule_matches(rule, action, context):
                return EvaluationResult(
                    decision=rule.decision,
                    reason=rule.reason,
                    policy_matched=f"{agent_id}.{action}",
                    agent_id=agent_id,
                    action=action,
                    rule_id=rule.id,
                )
        
        # No rule matched - return the configured default decision
        return EvaluationResult(
            decision=self.default_decision,
            reason=(
                f"No matching rule for action '{action}'. "
                f"Using default decision."
            ),
            policy_matched=None,
            agent_id=agent_id,
            action=action,
            rule_id=None,
        )
    
    def _rule_matches(
        self,
        rule: Rule,
        action: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Check if a rule matches the given action and context.
        
        Args:
            rule: The rule to evaluate
            action: The action being performed
            context: Context data for condition evaluation
            
        Returns:
            True if the rule matches, False otherwise
        """
        # Check if action matches
        if rule.action != action:
            return False
        
        # Check if all conditions are met
        for condition_key, condition_value in rule.conditions.items():
            context_value = context.get(condition_key)
            cond_ok = self._check_condition(condition_value, context_value)
            if not cond_ok:
                return False
        
        return True

    def _check_condition(self, expected: Any, actual: Any) -> bool:
        """Evaluate a single condition.

        - Numeric expected values are treated as upper-bounds (<=).
        - List expected values mean membership.
        - Otherwise do equality comparison.
        """
        if expected is None:
            return True
        if isinstance(expected, (int, float)):
            if actual is None:
                return False
            if not isinstance(actual, (int, float)):
                return False
            return actual <= expected
        if isinstance(expected, list):
            return actual in expected
        return actual == expected
    
    def get_policy(self, agent_id: str) -> Optional[Policy]:
        """
        Get the policy for a specific agent.
        
        Args:
            agent_id: Identifier of the agent
            
        Returns:
            Policy object or None if not found
        """
        return self._policy_map.get(agent_id)