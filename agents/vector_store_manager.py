"""
Vector Store Manager

Manages document loading, embeddings, and vector store lifecycle.
Separates vector store concerns from agent logic.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector store lifecycle and document operations.
    
    Responsibilities:
    - Initialize embeddings model
    - Load documents into vector store
    - Provide access to vector store
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store manager.
        
        Args:
            collection_name: Name for the Qdrant collection
            embedding_model: HuggingFace embedding model name
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize embeddings (local, no API calls)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Vector store (initialized via load_text_documents)
        self.vectorstore: Optional[Qdrant] = None
    
    def load_text_documents(self, docs_dir: Path) -> Dict[str, Any]:
        """
        Load documents from directory into in-memory vector store.
        
        Args:
            docs_dir: Path to directory containing .txt files
            
        Returns:
            Dict with status and count of loaded documents
        """
        logger.info(
            "Loading documents into vector store",
            extra={
                "collection": self.collection_name,
                "directory": str(docs_dir)
            }
        )
        
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
            logger.error(
                "Document directory not found",
                extra={
                    "collection": self.collection_name,
                    "directory": str(docs_dir)
                }
            )
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
            logger.warning(
                "No documents found in directory",
                extra={
                    "collection": self.collection_name,
                    "directory": str(docs_dir)
                }
            )
            return {
                "status": "error",
                "message": "No documents found in directory"
            }
        
        logger.info(
            "Creating vector store with embeddings",
            extra={
                "collection": self.collection_name,
                "documents": len(texts),
                "embedding_model": self.embedding_model
            }
        )
        
        # Create in-memory Qdrant vector store
        self.vectorstore = Qdrant.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name=self.collection_name,
            location=":memory:"
        )
        
        logger.info(
            "Documents loaded successfully",
            extra={"collection": self.collection_name, "documents": len(texts)}
        )
        
        return {
            "status": "success",
            "documents_loaded": len(texts),
            "collection": self.collection_name
        }
    
    def get_vectorstore(self) -> Optional[Qdrant]:
        """Get the initialized vector store."""
        return self.vectorstore
    
    def is_ready(self) -> bool:
        """Check if vector store is initialized and ready."""
        return self.vectorstore is not None
