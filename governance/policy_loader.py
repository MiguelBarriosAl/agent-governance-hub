"""
Policy Loader

Responsible for loading and parsing YAML policy files.
"""
import yaml
from pathlib import Path
from typing import List
from governance.models import PolicySet, Policy


# Supported policy schema version
SUPPORTED_POLICY_VERSION = "1.0"


class PolicyLoadError(Exception):
    """Raised when a policy file cannot be loaded or parsed."""
    pass


class PolicyLoader:
    """
    Loads governance policies from YAML files.
    
    Attributes:
        policy_dir: Directory containing policy YAML files
    """
    
    def __init__(self, policy_dir: Path):
        """
        Initialize the policy loader.
        
        Args:
            policy_dir: Path to directory containing policy files
        """
        self.policy_dir = Path(policy_dir)
        if not self.policy_dir.exists():
            raise PolicyLoadError(f"Policy directory not found: {policy_dir}")
    
    def _validate_policy_version(self, data: dict, filename: str) -> None:
        """
        Validate policy file version.
        
        Args:
            data: Parsed YAML data
            filename: Name of the policy file
            
        Raises:
            PolicyLoadError: If version is missing or incompatible
        """
        if "version" not in data:
            raise PolicyLoadError(f"Missing 'version' field in {filename}")
        
        version = str(data["version"])
        if version != SUPPORTED_POLICY_VERSION:
            raise PolicyLoadError(
                f"Incompatible policy version in {filename}: "
                f"expected {SUPPORTED_POLICY_VERSION!r}, got {version!r}"
            )
    
    def load_policy_file(self, filename: str) -> PolicySet:
        """
        Load a single policy file and parse it into a PolicySet.
        
        Args:
            filename: Name of the YAML file to load
            
        Returns:
            Parsed PolicySet object
            
        Raises:
            PolicyLoadError: If file cannot be read or parsed
        """
        file_path = self.policy_dir / filename
        
        if not file_path.exists():
            raise PolicyLoadError(f"Policy file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            # Validate policy version
            self._validate_policy_version(data, filename)

            # Validate and parse into Pydantic models
            policy_set = PolicySet(**data)

            # Ensure structural defaults and normalization
            for p in policy_set.policies:
                # guarantee rules list exists
                p.rules = p.rules or []
                for idx, r in enumerate(p.rules):
                    # ensure conditions exist
                    if r.conditions is None:
                        r.conditions = {}
                    # if rule id missing, create a stable fallback id
                    if not getattr(r, "id", None):
                        r.id = f"{p.agent_id}-{idx+1}"

            return policy_set

        except yaml.YAMLError as e:
            raise PolicyLoadError(
                f"Invalid YAML in {filename}: {e}"
            ) from e
        except PolicyLoadError:
            # re-raise our own errors
            raise
        except Exception as e:
            raise PolicyLoadError(
                f"Error loading policy {filename}: {e}"
            ) from e
    
    def load_all_policies(self) -> List[Policy]:
        """
        Load all policy files from the policy directory.
        
        Returns:
            Flat list of all policies from all files
            
        Raises:
            PolicyLoadError: If any file cannot be loaded
        """
        all_policies = []

        # Find all YAML files in the directory
        yaml_files = list(self.policy_dir.glob("*.yaml"))
        yaml_files += list(self.policy_dir.glob("*.yml"))

        if not yaml_files:
            raise PolicyLoadError(
                f"No policy files found in {self.policy_dir}"
            )

        for yaml_file in yaml_files:
            policy_set = self.load_policy_file(yaml_file.name)
            for policy in policy_set.policies:
                # attach source filename for traceability
                setattr(policy, "source_file", yaml_file.name)
                all_policies.append(policy)

        return all_policies
    
    def load_default_policy(self) -> PolicySet:
        """
        Load the default policy file.
        
        Returns:
            Default PolicySet
            
        Raises:
            PolicyLoadError: If default.yaml cannot be loaded
        """
        return self.load_policy_file("default.yaml")
