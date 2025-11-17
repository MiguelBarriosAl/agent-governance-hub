"""
Vector Retrieval Tool

LangChain tool for searching the vector database.
"""
from typing import Any, Type
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import Qdrant
import logging

logger = logging.getLogger(__name__)


class VectorRetrievalInput(BaseModel):
    """Input schema for vector retrieval tool."""
    query: str = Field(description="Natural language search query")


class VectorRetrievalTool(BaseTool):
    """Tool for retrieving documents from vector database."""
    
    name: str = "vector_retrieval"
    description: str = (
        "Search the vector database for relevant documents. "
        "Use this when you need factual information from documents."
    )
    args_schema: Type[BaseModel] = VectorRetrievalInput
    vectorstore: Any = Field(description="Qdrant vectorstore instance")
    
    # Policy metadata - governance callback uses this
    policy_action: str = "query_database"
    
    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        """Execute vector similarity search."""
        try:
            results = self.vectorstore.similarity_search(query=query, k=3)

            if not results:
                return "No relevant documents found."

            output = []
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "unknown")
                content = doc.page_content[:300]
                output.append(f"[{i}] Source: {source}\n{content}...")

            return "\n\n".join(output)

        except Exception as e:
            logger.error(
                "Vector retrieval failed",
                extra={"error": str(e), "query": query}
            )
            return f"Error during retrieval: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async execution not supported."""
        raise NotImplementedError("Async not supported")

