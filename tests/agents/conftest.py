"""
Shared fixtures for agent tests.

Provides reusable test data and utilities for agent testing.
"""
import pytest
from pathlib import Path
from governance.policy_loader import PolicyLoader
from governance.policy_engine import PolicyEngine


@pytest.fixture
def policy_engine() -> PolicyEngine:
    """Create a PolicyEngine with default policies."""
    loader = PolicyLoader(Path("config/policies"))
    policies = loader.load_all_policies()
    return PolicyEngine(policies)


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """Create sample documents for testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Create sample documents
    (docs_dir / "doc_001.txt").write_text(
        "Machine learning is a subset of artificial intelligence."
    )
    (docs_dir / "doc_002.txt").write_text(
        "Deep learning uses neural networks with multiple layers."
    )
    (docs_dir / "doc_003.txt").write_text(
        "Natural language processing enables computers to understand text."
    )
    
    return docs_dir
