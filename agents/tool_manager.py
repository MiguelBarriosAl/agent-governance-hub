"""
Tool Manager

Manages tool creation and agent executor configuration.
Separates tool setup from agent orchestration.
"""
from typing import List, Optional
import logging

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.vectorstores import Qdrant

from agents.prompts import get_rag_prompt
from tools.vector_retrieval import VectorRetrievalTool

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Manages tool creation and agent executor setup.
    
    Responsibilities:
    - Create tools (e.g., VectorRetrievalTool)
    - Configure AgentExecutor with tools and LLM
    - Provide access to tools and executor
    """
    
    def __init__(self, vectorstore: Qdrant, llm: ChatOpenAI):
        """
        Initialize tool manager.
        
        Args:
            vectorstore: Qdrant vector store for retrieval tool
            llm: LLM for agent reasoning
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.tools: List = []
        self.agent_executor: Optional[AgentExecutor] = None
    
    def setup_tools(self):
        """Setup tools and configure agent executor."""
        logger.debug("Setting up tools and agent executor")
        
        # Create vector retrieval tool
        retrieval_tool = VectorRetrievalTool(vectorstore=self.vectorstore)
        self.tools = [retrieval_tool]
        
        # Create ReAct agent with prompt
        prompt = get_rag_prompt()
        
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        logger.debug("Tools and agent executor configured successfully")
    
    def get_executor(self) -> Optional[AgentExecutor]:
        """Get the configured agent executor."""
        return self.agent_executor
    
    def get_tools(self) -> List:
        """Get the list of configured tools."""
        return self.tools

