"""
Unit tests for PolicyLoader.

Tests validation of YAML policy files and Pydantic model enforcement.
"""
import pytest
from pathlib import Path
from governance.policy_loader import PolicyLoader, PolicyLoadError
from governance.models import PolicySet


class TestPolicyLoaderInitialization:
    """Tests for PolicyLoader initialization."""
    
    def test_init_with_valid_directory(self, policy_dir: Path):
        """Test PolicyLoader initializes with valid directory."""
        loader = PolicyLoader(policy_dir)
        assert loader.policy_dir == policy_dir
    
    def test_init_with_nonexistent_directory(self, tmp_path: Path):
        """Test that non-existent policy directory raises PolicyLoadError."""
        nonexistent_dir = tmp_path / "does_not_exist"
        
        with pytest.raises(
            PolicyLoadError, match="Policy directory not found"
        ):
            PolicyLoader(nonexistent_dir)


class TestPolicyFileLoading:
    """Tests for loading individual policy files."""
    
    def test_load_valid_policy_file(self, valid_policy_yaml: Path):
        """Test successful loading of a valid policy file."""
        loader = PolicyLoader(valid_policy_yaml)
        policy_set = loader.load_policy_file("test_policy.yaml")
        
        assert isinstance(policy_set, PolicySet)
        assert policy_set.version == "1.0"
        assert len(policy_set.policies) == 1
        assert policy_set.policies[0].agent_id == "test_agent"
        assert len(policy_set.policies[0].rules) == 1
        assert policy_set.policies[0].rules[0].id == "T001"
    
    def test_load_nonexistent_file(self, valid_policy_yaml: Path):
        """Test that loading a non-existent file raises PolicyLoadError."""
        loader = PolicyLoader(valid_policy_yaml)
        
        with pytest.raises(PolicyLoadError, match="Policy file not found"):
            loader.load_policy_file("nonexistent.yaml")
    


class TestVersionValidation:
    """Tests for policy version validation."""
    
    def test_unsupported_version(self, policy_dir: Path):
        """Test that unsupported policy version raises PolicyLoadError."""
        # Version allowed is defined in policy_loader.py as SUPPORTED_POLICY_VERSION
        policy_file = policy_dir / "bad_version.yaml"
        policy_file.write_text("""
version: "2.0"
policies: []
""")
        
        loader = PolicyLoader(policy_dir)
        
        with pytest.raises(
            PolicyLoadError, match="Incompatible policy version"
        ):
            loader.load_policy_file("bad_version.yaml")
    
    def test_missing_version_field(self, policy_dir: Path):
        """Test that missing version field raises PolicyLoadError."""
        policy_file = policy_dir / "no_version.yaml"
        policy_file.write_text("""
policies: []
""")
        
        loader = PolicyLoader(policy_dir)
        
        with pytest.raises(PolicyLoadError, match="Missing 'version' field"):
            loader.load_policy_file("no_version.yaml")


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""
    
    def test_missing_required_field_in_rule(self, policy_dir: Path):
        """Test that missing required fields in rule raises PolicyLoadError."""
        # Missing 'reason' field in rule
        policy_file = policy_dir / "incomplete_rule.yaml"
        policy_file.write_text("""
version: "1.0"
policies:
  - agent_id: "test_agent"
    description: "Test policy"
    rules:
      - id: "T001"
        action: "test_action"
        decision: "allow"
        conditions: {}
""")
        
        loader = PolicyLoader(policy_dir)
        
        with pytest.raises(PolicyLoadError) as exc_info:
            loader.load_policy_file("incomplete_rule.yaml")
        
        assert "validation error" in str(exc_info.value).lower()
    
    def test_missing_required_field_in_policy(self, policy_dir: Path):
        """Test that missing description in policy raises PolicyLoadError."""
        # Missing 'description' field in policy
        policy_file = policy_dir / "incomplete_policy.yaml"
        policy_file.write_text("""
version: "1.0"
policies:
  - agent_id: "test_agent"
    rules:
      - id: "T001"
        action: "test_action"
        decision: "allow"
        conditions: {}
        reason: "Test reason"
""")
        
        loader = PolicyLoader(policy_dir)
        
        with pytest.raises(PolicyLoadError) as exc_info:
            loader.load_policy_file("incomplete_policy.yaml")
        
        assert "validation error" in str(exc_info.value).lower()
    
    def test_policy_with_complex_conditions(self, policy_dir: Path):
        """Test loading policy with complex condition values."""
        policy_file = policy_dir / "complex.yaml"
        policy_file.write_text("""
version: "1.0"
policies:
  - agent_id: "analyzer"
    description: "Analyzer with conditions"
    rules:
      - id: "A001"
        action: "analyze"
        decision: "verify"
        conditions:
          max_tokens: 4000
          allowed_formats: ["json", "yaml"]
          requires_approval: true
        reason: "Complex conditions test"
""")
        
        loader = PolicyLoader(policy_dir)
        policy_set = loader.load_policy_file("complex.yaml")
        
        rule = policy_set.policies[0].rules[0]
        assert rule.conditions["max_tokens"] == 4000
        assert rule.conditions["allowed_formats"] == ["json", "yaml"]
        assert rule.conditions["requires_approval"] is True


class TestMultiplePolicyLoading:
    """Tests for loading multiple policy files."""
    
    def test_load_all_policies_multiple_files(self, multi_policy_yaml: Path):
        """Test loading multiple policy files returns all policies."""
        loader = PolicyLoader(multi_policy_yaml)
        all_policies = loader.load_all_policies()
        
        # Should have 3 policies total
        # (1 from policy1.yaml, 2 from policy2.yaml)
        assert len(all_policies) == 3
        agent_ids = [p.agent_id for p in all_policies]
        assert "agent1" in agent_ids
        assert "agent2" in agent_ids
        assert "agent3" in agent_ids
        
        # Verify source_file is set
        for policy in all_policies:
            assert policy.source_file in ["policy1.yaml", "policy2.yaml"]
    
    def test_load_all_policies_empty_directory(self, policy_dir: Path):
        """Test that loading from empty directory raises PolicyLoadError."""
        loader = PolicyLoader(policy_dir)
        
        with pytest.raises(PolicyLoadError, match="No policy files found"):
            loader.load_all_policies()
