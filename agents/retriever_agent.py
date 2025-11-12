"""
Retriever Agent

Agent specialized in document retrieval using vector search.
All actions are governed by policies defined in the policy engine.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from agents.base_agent import BaseAgent
from governance.policy_engine import PolicyEngine
from governance.models import DecisionType


class RetrieverAgent(BaseAgent):
    """
    Agent that retrieves documents using vector similarity search.
    
    Uses Qdrant in-memory vector store with local HuggingFace embeddings.
    All queries are evaluated against governance policies.
    """
    
    def __init__(
        self,
        name: str,
        policy_engine: PolicyEngine,
        collection_name: str = "news_docs"
    ):
        """
        Initialize the retriever agent.
        
        Args:
            name: Agent identifier (must match policy agent_id)
            policy_engine: Policy engine for action evaluation
            collection_name: Name of the Qdrant collection
        """
        super().__init__(name, policy_engine)
        self.collection_name = collection_name
        
        # Initialize embeddings model (local, no API calls)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Vector store (initialized via load_documents)
        self.vectorstore: Optional[Qdrant] = None
    
    def load_text_documents(self, docs_dir: Path) -> Dict[str, Any]:
        """
        Load documents from directory into in-memory vector store.
        
        Args:
            docs_dir: Path to directory containing .txt files
            
        Returns:
            Dict with status and count of loaded documents
        """
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
            return {
                "status": "error",
                "message": f"Directory not found: {docs_dir}"
            }
        
        # Read all .txt files
        texts = []
        metadatas = []
        
        for doc_file in sorted(docs_path.glob("*.txt")):
            content = doc_file.read_text(encoding="utf-8").strip()
            if content:
                texts.append(content)
                metadatas.append({
                    "source": doc_file.name,
                    "path": str(doc_file)
                })
        
        if not texts:
            return {
                "status": "error",
                "message": "No documents found in directory"
            }
        
        # Create in-memory Qdrant vector store
        self.vectorstore = Qdrant.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name=self.collection_name,
            location=":memory:"
        )
        
        return {
            "status": "success",
            "documents_loaded": len(texts),
            "collection": self.collection_name
        }
    
    def query_documents(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Query documents with governance policy evaluation.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            Dict with results or policy decision
        """
        # Evaluate action against policies
        evaluation = self.evaluate_action(
            action="query_database",
            context={"query": query, "top_k": top_k}
        )
        
        # If blocked or requires verification, return decision
        if evaluation.decision in [DecisionType.BLOCK, DecisionType.VERIFY]:
            return {
                "status": "rejected",
                "decision": evaluation.decision.value,
                "reason": evaluation.reason,
                "rule_id": evaluation.rule_id,
                "query": query
            }
        
        # Check if vector store is initialized
        if self.vectorstore is None:
            return {
                "status": "error",
                "message": "Vector store not initialized. Call load_documents() first."
            }
        
        # Execute search
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=top_k
            )
            
            # Format results
            formatted_results = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                }
                for doc in results
            ]
            
            return {
                "status": "success",
                "decision": evaluation.decision.value,
                "rule_id": evaluation.rule_id,
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results)
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}"
            }
    
    def delete_data(self, doc_id: str) -> Dict[str, Any]:
        """
        Attempt to delete data (should be blocked by policy).
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dict with policy decision (should be BLOCK)
        """
        # Evaluate action against policies
        evaluation = self.evaluate_action(
            action="delete_data",
            context={"doc_id": doc_id}
        )
        
        return {
            "status": "rejected",
            "decision": evaluation.decision.value,
            "reason": evaluation.reason,
            "rule_id": evaluation.rule_id,
            "doc_id": doc_id
        }
