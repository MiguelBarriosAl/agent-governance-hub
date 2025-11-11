"""
Shared fixtures for governance tests.

Provides reusable test data and utilities for policy testing.
"""
import pytest
from pathlib import Path


@pytest.fixture
def policy_dir(tmp_path: Path) -> Path:
    """Create a temporary policy directory."""
    policies = tmp_path / "policies"
    policies.mkdir()
    return policies


@pytest.fixture
def valid_policy_yaml(policy_dir: Path) -> Path:
    """Create a valid policy YAML file for testing."""
    policy_file = policy_dir / "test_policy.yaml"
    policy_file.write_text("""
version: "1.0"
policies:
  - agent_id: "test_agent"
    description: "Test agent policy"
    rules:
      - id: "T001"
        action: "test_action"
        decision: "allow"
        conditions: {}
        reason: "Test reason"
""")
    return policy_dir


@pytest.fixture
def multi_policy_yaml(policy_dir: Path) -> Path:
    """Create multiple policy YAML files for testing."""
    # First policy file
    policy1 = policy_dir / "policy1.yaml"
    policy1.write_text("""
version: "1.0"
policies:
  - agent_id: "agent1"
    description: "First agent"
    rules:
      - id: "A001"
        action: "action1"
        decision: "allow"
        conditions: {}
        reason: "Reason 1"
""")
    
    # Second policy file
    policy2 = policy_dir / "policy2.yaml"
    policy2.write_text("""
version: "1.0"
policies:
  - agent_id: "agent2"
    description: "Second agent"
    rules:
      - id: "B001"
        action: "action2"
        decision: "block"
        conditions: {}
        reason: "Reason 2"
  - agent_id: "agent3"
    description: "Third agent"
    rules:
      - id: "C001"
        action: "action3"
        decision: "verify"
        conditions: {}
        reason: "Reason 3"
""")
    
    return policy_dir
