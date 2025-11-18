"""
Governed RAG Agent

ReAct-based agent with policy enforcement and observability.
All decisions go through the policy engine - no heuristics.
Refactored to use separated managers for clean responsibilities.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import time

from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from agents.vector_store_manager import VectorStoreManager
from agents.tool_manager import ToolManager
from agents.execution_coordinator import ExecutionCoordinator
from governance.policy_engine import PolicyEngine
from governance.models import DecisionType

logger = logging.getLogger(__name__)


class GovernedRAGAgent(BaseAgent):
    """
    RAG agent with LLM-driven reasoning and policy enforcement.
    
    Uses ReAct pattern internally:
    - Thought: LLM decides strategy
    - Action: Calls tools (if needed)
    - Observation: Processes results
    - Final Answer: Returns response
    
    All tool calls are evaluated against policies before execution.
    Orchestrates VectorStoreManager, ToolManager, and ExecutionCoordinator.
    """
    
    def __init__(
        self,
        name: str,
        policy_engine: PolicyEngine,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.0
    ):
        """
        Initialize the governed RAG agent.
        
        Args:
            name: Agent identifier (must match policy agent_id)
            policy_engine: Policy engine for governance
            llm_model: OpenAI model name
            temperature: LLM temperature (0=deterministic)
        """
        super().__init__(name, policy_engine)
        
        # LLM for reasoning
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature
        )
        
        # Managers (always created explicitly for POC clarity)
        self.vector_manager = VectorStoreManager(
            collection_name=f"{name}_documents"
        )
        self.tool_manager: Optional[ToolManager] = None
        self.execution_coordinator: Optional[ExecutionCoordinator] = None

    def load_documents(self, docs_dir: Path) -> Dict[str, Any]:
        """
        Load documents into vector store.
        
        Args:
            docs_dir: Directory containing .txt files
            
        Returns:
            Status dict
        """
        logger.info(
            "Loading documents into vector store",
            extra={"agent": self.name, "directory": str(docs_dir)}
        )
        
        # Delegate to VectorStoreManager
        result = self.vector_manager.load_text_documents(docs_dir)
        
        if result["status"] == "success":
            # Setup tools and execution coordinator
            self._setup_execution()
        
        return result
    
    def _setup_execution(self):
        """Setup tools and execution coordinator after documents loaded."""
        logger.debug("Setting up execution components")
        
        vectorstore = self.vector_manager.get_vectorstore()
        
        # Create and setup ToolManager
        self.tool_manager = ToolManager(vectorstore, self.llm)
        self.tool_manager.setup_tools()
        
        # Create ExecutionCoordinator
        self.execution_coordinator = ExecutionCoordinator(
            self,
            self.tool_manager.get_tools()
        )
        
        logger.debug("Execution components ready")

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Process user query with LLM reasoning and policy enforcement.
        
        The LLM decides whether to use tools or answer directly.
        All actions are evaluated through the policy engine.
        No heuristics - pure policy-based governance.
        
        Args:
            query: Natural language query from user
            
        Returns:
            Dict with answer and metadata
        """
        logger.info(
            "Processing user query",
            extra={"agent": self.name, "query": query[:100]}
        )
        
        if not self.tool_manager or not self.tool_manager.is_ready():
            logger.error(
                "Agent not initialized",
                extra={"agent": self.name}
            )
            return {
                "status": "error",
                "message": "Agent not initialized. Load documents first."
            }
        
        # Evaluate query through policy engine
        evaluation = self.evaluate_action(
            action="ask_question",
            context={"query": query}
        )
        
        logger.info(
            "Policy evaluation for query",
            extra={
                "agent": self.name,
                "action": "ask_question",
                "decision": evaluation.decision.value,
                "rule_id": evaluation.rule_id
            }
        )
        
        if evaluation.decision == DecisionType.BLOCK:
            logger.warning(
                "Query blocked by policy",
                extra={
                    "agent": self.name,
                    "rule_id": evaluation.rule_id,
                    "reason": evaluation.reason
                }
            )
            return {
                "status": "blocked",
                "reason": evaluation.reason,
                "rule_id": evaluation.rule_id
            }
        
        # Execute agent with governance
        start_time = time.time()
        try:
            executor = self.tool_manager.get_executor()
            result = self.execution_coordinator.execute(executor, query)
            
            elapsed = time.time() - start_time
            
            logger.info(
                "Query processed successfully",
                extra={
                    "agent": self.name,
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "answer_length": len(result)
                }
            )
            
            return {
                "status": "success",
                "answer": result,
                "decision": evaluation.decision.value,
                "rule_id": evaluation.rule_id
            }
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "Query processing failed",
                extra={
                    "agent": self.name,
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "error": str(e)
                }
            )
            return {
                "status": "error",
                "message": f"Agent execution failed: {str(e)}"
            }


