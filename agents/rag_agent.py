"""
Governed RAG Agent

ReAct-based agent that uses OpenAI for reasoning and tool calling,
with policy enforcement before executing any action.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler

from agents.base_agent import BaseAgent
from agents.prompts import get_rag_prompt
from governance.policy_engine import PolicyEngine
from governance.models import DecisionType
from tools.vector_retrieval import VectorRetrievalTool


class GovernanceCallback(BaseCallbackHandler):
    """Callback that enforces policies on tool executions."""
    
    def __init__(self, agent_ref):
        super().__init__()
        self.agent = agent_ref
    
    def on_tool_start(self, serialized: Dict[str, Any],
                     input_str: str, **kwargs):
        """Check policy before tool execution."""
        tool_name = serialized.get("name", "")
        
        if tool_name == "vector_retrieval":
            evaluation = self.agent.evaluate_action(
                action="vector_retrieval",
                context={"query": input_str}
            )
            
            if evaluation.decision == DecisionType.BLOCK:
                raise Exception(
                    f"Action blocked by policy: {evaluation.reason} "
                    f"(rule: {evaluation.rule_id})"
                )



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
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
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
            return {
                "status": "error",
                "message": "No documents found"
            }
        
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
        
        This is the main interface for the agent. The LLM decides
        whether to use tools or answer directly. All actions are
        governed by policies.
        
        Args:
            query: Natural language query from user
            
        Returns:
            Dict with answer and metadata
        """
        if not self.agent_executor:
            return {
                "status": "error",
                "message": "Agent not initialized. Load documents first."
            }
        
        # Check if query itself is allowed
        evaluation = self.evaluate_action(
            action="ask_question",
            context={"query": query}
        )
        
        if evaluation.decision == DecisionType.BLOCK:
            return {
                "status": "blocked",
                "reason": evaluation.reason,
                "rule_id": evaluation.rule_id
            }
        
        # Execute agent (LLM decides strategy internally)
        try:
            # Intercept tool calls for policy enforcement
            result = self._execute_with_governance(query)
            
            return {
                "status": "success",
                "answer": result,
                "decision": evaluation.decision.value,
                "rule_id": evaluation.rule_id
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Agent execution failed: {str(e)}"
            }
    
    def _execute_with_governance(self, query: str) -> str:
        """
        Execute agent with policy checks on tool calls via callback.
        
        Args:
            query: User query
            
        Returns:
            Agent's final answer
        """
        # Create callback that intercepts tool calls
        governance_callback = GovernanceCallback(self)
        
        # Execute with callback
        result = self.agent_executor.invoke(
            {"input": query},
            config={"callbacks": [governance_callback]}
        )
        
        return result.get("output", "No response generated")