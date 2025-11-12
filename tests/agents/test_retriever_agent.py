"""
Unit tests for RetrieverAgent.

Tests governance policy enforcement and document retrieval functionality.
"""
from pathlib import Path
from agents.retriever_agent import RetrieverAgent
from governance.policy_engine import PolicyEngine


class TestRetrieverAgentInitialization:
    """Tests for RetrieverAgent initialization."""
    
    def test_init_with_policy_engine(self, policy_engine: PolicyEngine):
        """Test RetrieverAgent initializes correctly."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine
        )
        
        assert agent.name == "retriever"
        assert agent.policy_engine == policy_engine
        assert agent.collection_name == "news_docs"
        assert agent.vectorstore is None
    
    def test_init_with_custom_collection(self, policy_engine: PolicyEngine):
        """Test initialization with custom collection name."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine,
            collection_name="custom_collection"
        )
        
        assert agent.collection_name == "custom_collection"


class TestDocumentLoading:
    """Tests for document loading functionality."""
    
    def test_load_documents_success(
        self,
        policy_engine: PolicyEngine,
        sample_docs_dir: Path
    ):
        """Test successful document loading."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine
        )
        
        result = agent.load_text_documents(sample_docs_dir)
        
        assert result["status"] == "success"
        assert result["documents_loaded"] == 3
        assert agent.vectorstore is not None
    
    def test_load_documents_nonexistent_dir(
        self,
        policy_engine: PolicyEngine,
        tmp_path: Path
    ):
        """Test loading from non-existent directory."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine
        )
        
        nonexistent = tmp_path / "nonexistent"
        result = agent.load_text_documents(nonexistent)
        
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()
    
    def test_load_documents_empty_dir(
        self,
        policy_engine: PolicyEngine,
        tmp_path: Path
    ):
        """Test loading from empty directory."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine
        )
        
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = agent.load_text_documents(empty_dir)
        
        assert result["status"] == "error"
        assert "no documents" in result["message"].lower()


class TestPolicyEnforcement:
    """Tests for governance policy enforcement."""
    
    def test_query_documents_allowed(
        self,
        policy_engine: PolicyEngine,
        sample_docs_dir: Path
    ):
        """Test that query_documents is allowed by policy."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine
        )
        agent.load_text_documents(sample_docs_dir)
        
        result = agent.query_documents("machine learning")
        
        assert result["status"] == "success"
        assert result["decision"] == "allow"
        assert "results" in result
        assert result["count"] > 0
    
    def test_delete_data_blocked(self, policy_engine: PolicyEngine):
        """Test that delete_data is blocked by policy."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine
        )
        
        result = agent.delete_data("doc_001")
        
        assert result["status"] == "rejected"
        assert result["decision"] == "block"
        assert "rule_id" in result
        assert result["doc_id"] == "doc_001"
    
    def test_query_without_loading_documents(
        self,
        policy_engine: PolicyEngine
    ):
        """Test query fails gracefully without loaded documents."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine
        )
        
        result = agent.query_documents("test query")
        
        assert result["status"] == "error"
        assert "not initialized" in result["message"].lower()


class TestSearchFunctionality:
    """Tests for document search functionality."""
    
    def test_search_returns_relevant_results(
        self,
        policy_engine: PolicyEngine,
        sample_docs_dir: Path
    ):
        """Test that search returns relevant results."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine
        )
        agent.load_text_documents(sample_docs_dir)
        
        result = agent.query_documents("neural networks", top_k=2)
        
        assert result["status"] == "success"
        assert len(result["results"]) <= 2
        assert all("content" in r for r in result["results"])
        assert all("source" in r for r in result["results"])
    
    def test_search_with_custom_top_k(
        self,
        policy_engine: PolicyEngine,
        sample_docs_dir: Path
    ):
        """Test search with custom result count."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine
        )
        agent.load_text_documents(sample_docs_dir)
        
        result = agent.query_documents("artificial intelligence", top_k=1)
        
        assert result["status"] == "success"
        assert result["count"] <= 1


class TestEvaluateAction:
    """Tests for action evaluation."""
    
    def test_evaluate_action_returns_result(
        self,
        policy_engine: PolicyEngine
    ):
        """Test that evaluate_action returns EvaluationResult."""
        agent = RetrieverAgent(
            name="retriever",
            policy_engine=policy_engine
        )
        
        evaluation = agent.evaluate_action("query_database")
        
        assert evaluation.decision is not None
        assert evaluation.agent_id == "retriever"
        assert evaluation.action == "query_database"
