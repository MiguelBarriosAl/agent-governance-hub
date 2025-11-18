"""
Document Pipeline

Handles document loading and vector store preparation.
Separates setup concerns from agent execution logic.
"""
from pathlib import Path
from typing import Dict, Any
import logging

from agents.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    Pipeline for document loading and vector store preparation.
    
    Responsibilities:
    - Load documents from directory
    - Create and initialize vector store
    - Provide ready-to-use VectorStoreManager
    
    Keeps agent focused on query processing only.
    """
    
    def __init__(self, collection_name: str):
        """
        Initialize document pipeline.
        
        Args:
            collection_name: Name for the vector store collection
        """
        self.collection_name = collection_name
        self.vector_manager: VectorStoreManager = None
    
    def load_documents(self, docs_dir: Path) -> Dict[str, Any]:
        """
        Load documents and prepare vector store.
        
        Args:
            docs_dir: Directory containing .txt files
            
        Returns:
            Dict with status and metadata
        """
        logger.info(
            "Document pipeline starting",
            extra={
                "collection": self.collection_name,
                "directory": str(docs_dir)
            }
        )
        
        # Create VectorStoreManager
        self.vector_manager = VectorStoreManager(
            collection_name=self.collection_name
        )
        
        # Load documents
        result = self.vector_manager.load_text_documents(docs_dir)
        
        if result["status"] == "success":
            logger.info(
                "Document pipeline completed",
                extra={
                    "collection": self.collection_name,
                    "documents": result["documents_loaded"]
                }
            )
        else:
            logger.error(
                "Document pipeline failed",
                extra={
                    "collection": self.collection_name,
                    "error": result.get("message")
                }
            )
        
        return result
    
    def get_vector_manager(self) -> VectorStoreManager:
        """
        Get the initialized vector store manager.
        
        Returns:
            VectorStoreManager ready for use
            
        Raises:
            RuntimeError: If documents not loaded yet
        """
        if self.vector_manager is None:
            raise RuntimeError(
                "Documents not loaded. Call load_documents() first."
            )
        return self.vector_manager
    
    def is_ready(self) -> bool:
        """Check if pipeline completed successfully."""
        return (
            self.vector_manager is not None
            and self.vector_manager.is_ready()
        )
