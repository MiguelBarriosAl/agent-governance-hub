"""
Governed RAG Agent

ReAct-based agent with policy enforcement and observability.
All decisions go through the policy engine - no heuristics.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import time

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

from agents.base_agent import BaseAgent
from agents.prompts import get_rag_prompt
from agents.callbacks import (
    PolicyEnforcementCallback,
    ObservabilityCallback
)
from governance.policy_engine import PolicyEngine
from governance.models import DecisionType
from tools.vector_retrieval import VectorRetrievalTool

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
    """
    
    def __init__(
        self,
        name: str,
        policy_engine: PolicyEngine,
        vectorstore: Optional[Qdrant] = None,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.0
    ):
        """
        Initialize the governed RAG agent.
        
        Args:
            name: Agent identifier (must match policy agent_id)
            policy_engine: Policy engine for governance
            vectorstore: Qdrant vectorstore (created if None)
            llm_model: OpenAI model name
            temperature: LLM temperature (0=deterministic)
        """
        super().__init__(name, policy_engine)
        
        # LLM for reasoning
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature
        )
        
        # Vector store
        self.vectorstore = vectorstore
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Tools (only vector retrieval for now)
        self.tools = []
        if self.vectorstore:
            self._setup_tools()
        
        # Agent executor (created when tools are ready)
        self.agent_executor = None
    
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
        
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
            logger.error(
                "Document directory not found",
                extra={"agent": self.name, "directory": str(docs_dir)}
            )
            return {
                "status": "error",
                "message": f"Directory not found: {docs_dir}"
            }
        
        # Read documents
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
                extra={"agent": self.name, "directory": str(docs_dir)}
            )
            return {
                "status": "error",
                "message": "No documents found"
            }
        
        logger.info(
            "Creating vector store with embeddings",
            extra={
                "agent": self.name,
                "documents": len(texts),
                "embedding_model": "all-MiniLM-L6-v2"
            }
        )
        
        # Create vector store
        self.vectorstore = Qdrant.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name=self.name,
            location=":memory:"
        )
        
        # Setup tools now that vectorstore exists
        self._setup_tools()
        
        logger.info(
            "Documents loaded successfully",
            extra={"agent": self.name, "documents": len(texts)}
        )
        
        return {
            "status": "success",
            "documents_loaded": len(texts)
        }
    
    def _setup_tools(self):
        """Setup tools for the agent."""
        if self.vectorstore is None:
            return
        
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
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True
        )
    
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
        
        if not self.agent_executor:
            logger.error(
                "Agent not initialized",
                extra={"agent": self.name}
            )
            return {
                "status": "error",
                "message": "Agent not initialized. Load documents first."
            }
        
        # Evaluate query through policy engine
        # (No heuristics - policy decides everything)
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
            result = self._execute_with_governance(query)
            
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
    
    def _execute_with_governance(self, query: str) -> str:
        """
        Execute agent with separated policy enforcement and observability.
        
        Uses two independent callbacks:
        - PolicyEnforcementCallback: validates tool calls against policies
        - ObservabilityCallback: logs all agent behavior
        
        Args:
            query: User query
            
        Returns:
            Agent's final answer
        """
        # Create separated callbacks
        policy_callback = PolicyEnforcementCallback(self, self.tools)
        observability_callback = ObservabilityCallback(self)
        
        # Execute with both callbacks
        result = self.agent_executor.invoke(
            {"input": query},
            config={"callbacks": [policy_callback, observability_callback]}
        )
        
        return result.get("output", "No response generated")

