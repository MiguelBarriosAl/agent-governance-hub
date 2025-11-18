# Agent Governance Hub

A lightweight, policy-driven governance framework for LLM agents with ReAct reasoning. Control agent behavior through declarative YAML policies and structured observability—ensuring safe, compliant, and traceable AI operations.

## Overview

Agent Governance Hub provides a **policy-first architecture** for governing LLM agents. Every action is evaluated against declarative policies before execution, with complete separation between policy enforcement and observability logging.

**What makes this project unique**: We investigate how governance becomes critical when agents make autonomous decisions about tool usage. Our RAG agent uses an LLM that independently decides whether to retrieve documents or answer directly—and governance policies control both the query permission AND the retrieval tool execution. This demonstrates real-world scenarios where AI agents need guardrails on their decision-making process, not just their final actions.

### Key Features

- **🔒 Policy Enforcement**: Automatic evaluation before every agent action
- **📊 Separated Observability**: Independent callbacks for governance and logging
- **🤖 ReAct Agent**: OpenAI-powered reasoning with LangChain tool calling (gpt-3.5-turbo)
- **🔍 Vector Retrieval**: Built-in semantic search with Qdrant (in-memory) + HuggingFace embeddings
- **📝 Structured Logging**: Complete traceability of LLM decisions, tool usage, and policy evaluations
- **🎯 Autonomous RAG Decision-Making**: LLM independently decides when to use retrieval tools vs. direct answers
- **👁️ RAG Usage Visibility**: Clear indicators showing whether the agent used RAG (🔍) or answered directly (💬)
- **⚡ FastAPI Integration**: RESTful API for policy evaluation and agent orchestration
- **🛡️ Type Safety**: Pydantic validation throughout the stack

## Architecture

### High-Level Flow

The architecture demonstrates governance at two critical decision points:

```
┌──────────────────┐
│  User Query      │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  GovernedRAGAgent                                    │
│  ┌──────────────────────────────────────────────┐   │
│  │ 1. Policy Check (ask_question action)       │   │
│  │    ↓ PolicyEngine.evaluate()                │   │
│  │    ↓ Decision: ALLOW/BLOCK                  │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ 2. LLM Reasoning (ReAct pattern)            │   │
│  │    ↓ OpenAI GPT-3.5-turbo                   │   │
│  │    ↓ Decides: Direct Answer vs Tool Call    │   │
│  │    ⚠️ AUTONOMOUS DECISION - NOT CONTROLLED   │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ 3. Tool Execution (if needed)               │   │
│  │    ↓ PolicyEnforcementCallback intercepts   │   │
│  │    ↓ Gets tool.policy_action metadata       │   │
│  │    ↓ PolicyEngine.evaluate(query_database)  │   │
│  │    ↓ VectorRetrievalTool executes           │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ 4. Observability (throughout)               │   │
│  │    ↓ ObservabilityCallback logs all events  │   │
│  │    ↓ Timing, decisions, tool calls          │   │
│  │    ↓ Tracks: 🔍 RAG vs 💬 Direct mode      │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Final Answer    │
│  + Metadata      │
│  (RAG used?)     │
└──────────────────┘
```

**Why This Matters**: The LLM's autonomous decision-making (step 2) is the reason governance is critical. We can't predict when the agent will use tools, so we must:
- Enforce policies at the tool execution layer (step 3)
- Track which decision path was taken (step 4)
- Provide visibility into agent behavior patterns

### Core Components

```
┌─────────────────────────────────────────────────────┐
│  YAML Policies (config/policies/default.yaml)      │
│  • Declarative rules: allow/block/verify/flag      │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  PolicyEngine (governance/policy_engine.py)        │
│  • Evaluates agent_id + action + context          │
│  • First-match rule strategy                       │
│  • Returns EvaluationResult with decision          │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  PolicyEnforcementCallback (agents/callbacks.py)   │
│  • Intercepts tool calls before execution          │
│  • Reads tool.policy_action metadata               │
│  • Blocks execution if policy denies               │
└─────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  VectorRetrievalTool (tools/vector_retrieval.py)   │
│  • Metadata: policy_action = "query_database"      │
│  • Executes Qdrant similarity search               │
│  • Returns top-k relevant documents                │
└─────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  ObservabilityCallback (agents/callbacks.py)       │
│  • Logs LLM reasoning steps                        │
│  • Tracks tool execution timing                    │
│  • Records policy evaluation results               │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (dependency management)
- OpenAI API key

### Installation

```bash
# Clone repository
git clone https://github.com/MiguelBarriosAl/agent-governance-hub.git
cd agent-governance-hub

# Install dependencies
poetry install

# Configure OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Run the demo
poetry run python main.py

# Run tests
poetry run pytest -v
```

### Demo Output

The demo showcases governed RAG with autonomous LLM decision-making:

```bash
Query: What information do you have about machine learning?
------------------------------------------------------------
🔍 RAG | Answer: I found some information about machine learning:
Machine learning models require large amounts of training data...
Decision: allow (R007)

Query: Hello! How are you?
------------------------------------------------------------
💬 Direct | Answer: Hello! I'm here and ready to assist you. How can I help you today?
Decision: allow (R007)

Query: Find details about vector databases
------------------------------------------------------------
🔍 RAG | Answer: Vector databases store data as high-dimensional vectors...
Decision: allow (R007)
```

**Key Observations:**
- 🔍 **RAG Mode**: The LLM autonomously decided to use the vector retrieval tool for technical questions
- 💬 **Direct Mode**: The LLM answered the greeting directly without tool usage
- ✅ **Governance Applied**: Both the `ask_question` action and `query_database` tool calls were evaluated by policies

**What This Demonstrates:**
This simple demo reveals the complexity of governing autonomous agents. The LLM makes real-time decisions about when to use RAG, and our governance framework must control:
1. Whether the agent can process the query (`ask_question` policy)
2. Whether the agent can execute retrieval tools (`query_database` policy)
3. Complete observability of which path the LLM chose

This two-layer governance is critical in production systems where AI agents have multiple tools and make autonomous decisions about when to use them.

## Usage Examples

### 1. Define Policies (YAML)

```yaml
version: "1.0"
policies:
  - agent_id: "retriever"
    description: "Rules for RAG agent with vector retrieval"
    rules:
      - id: "R007"
        action: "ask_question"
        decision: "allow"
        conditions: {}
        reason: "Users can ask questions to the agent"

      - id: "R008"
        action: "query_database"
        decision: "allow"
        conditions: {}
        reason: "Agent can query the vector database for information"

      - id: "R003"
        action: "delete_data"
        decision: "block"
        conditions: {}
        reason: "Destructive operations are forbidden"
```

### 2. Create a Tool with Policy Metadata

```python
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field

class VectorRetrievalTool(BaseTool):
    name: str = "vector_retrieval"
    description: str = "Search the vector database for documents"
    
    # Policy metadata - governance callback uses this
    policy_action: str = "query_database"
    
    def _run(self, query: str) -> str:
        # Execute search
        results = self.vectorstore.similarity_search(query, k=3)
        return format_results(results)
```

### 3. Use the Governed Agent

```python
from pathlib import Path
from governance.policy_loader import PolicyLoader
from governance.policy_engine import PolicyEngine
from governance.models import DecisionType
from pipelines.document_pipeline import DocumentPipeline
from agents.rag_agent import GovernedRAGAgent

# Step 1: Load governance policies
loader = PolicyLoader(Path("config/policies"))
policies = loader.load_all_policies()
engine = PolicyEngine(policies=policies, default_decision=DecisionType.BLOCK)

# Step 2: Setup document pipeline (separate from agent)
pipeline = DocumentPipeline(collection_name="my_documents")
pipeline.load_documents(Path("data/docs"))

# Step 3: Create governed agent with pre-loaded vector store
agent = GovernedRAGAgent(
    name="retriever",
    policy_engine=engine,
    vector_manager=pipeline.get_vector_manager(),
    llm_model="gpt-3.5-turbo",
    temperature=0.0
)

# Step 4: Query the agent - LLM decides RAG vs Direct
result = agent.ask("What is machine learning?")

# Result includes answer + governance metadata + RAG usage
print(f"{result['answer']}")
print(f"Decision: {result['decision']} (rule: {result['rule_id']})")
print(f"Used RAG: {result['used_rag']}")  # True if vector retrieval was used
print(f"Tools: {result['tools_used']}")   # List of tools the LLM invoked
```

**Architecture Explanation:**
- **Separation of Concerns**: Document loading happens in `DocumentPipeline`, not the agent
- **Policy-First**: Engine with `default_decision=BLOCK` denies everything not explicitly allowed
- **Autonomous Decision**: The LLM independently chooses when to use RAG—governance tracks this
- **Observable Behavior**: Every result includes metadata about what the agent actually did

### 4. Structured Logging Output

```log
15:29:07 | INFO | agents.rag_agent | Processing user query | agent=retriever
15:29:07 | INFO | agents.rag_agent | Policy evaluation for query | decision=allow | rule_id=R007
15:29:07 | INFO | agents.callbacks | LLM reasoning started | model=gpt-3.5-turbo
15:29:08 | INFO | agents.callbacks | Agent action decided | tool=vector_retrieval
15:29:08 | INFO | agents.callbacks | Tool execution requested by LLM | tool=vector_retrieval
15:29:08 | INFO | agents.callbacks | Tool execution completed | elapsed_ms=12.32
15:29:09 | INFO | agents.rag_agent | Query processed successfully | elapsed_ms=2100.16
```

## Project Structure

```
agent-governance-hub/
├── agents/                    # Agent execution logic
│   ├── base_agent.py          # Abstract base with policy evaluation
│   ├── rag_agent.py           # ReAct RAG agent with OpenAI
│   ├── vector_store_manager.py # Manages embeddings and vectorstore
│   ├── tool_manager.py        # Configures tools and AgentExecutor
│   ├── execution_coordinator.py # Executes with governance callbacks
│   ├── callbacks.py           # Separated callbacks
│   │   ├── PolicyEnforcementCallback
│   │   └── ObservabilityCallback
│   └── prompts.py             # LLM prompt templates
├── pipelines/                 # Setup and preparation logic
│   └── document_pipeline.py   # Document loading (separate from agent)
├── tools/                     # LangChain tools with policy metadata
│   └── vector_retrieval.py    # Vector search tool
├── governance/                # Policy engine core
│   ├── models.py              # Pydantic models (Policy, Rule, DecisionType)
│   ├── policy_loader.py       # YAML loading & validation
│   └── policy_engine.py       # Rule evaluation logic
├── config/                    # Configuration
│   ├── settings.py            # Application settings
│   └── policies/              # YAML policy files
│       └── default.yaml       # Default governance rules
├── data/docs/                 # Sample documents for vector search
├── tests/                     # Test suite (11 passing tests)
│   ├── agents/                # Agent tests
│   └── governance/            # Policy tests (11 tests)
├── main.py                    # Clean demo showing RAG decision-making (69 lines)
├── pyproject.toml             # Dependencies (Poetry)
└── .env                       # API keys (gitignored)
```

**Key Design Decisions:**
- `pipelines/` handles setup; `agents/` handles execution (clear separation)
- `ExecutionCoordinator` tracks tool usage to provide RAG visibility
- `main.py` kept minimal and readable (69 lines) to demonstrate architecture clearly

## Policy System

### Decision Types

- **`allow`**: Action proceeds without restrictions
- **`block`**: Action is rejected immediately
- **`verify`**: Requires human approval (future: integration with approval workflows)
- **`flag`**: Action proceeds but is logged for audit

### Conditions

Add dynamic conditions to rules:

```yaml
- id: "R005"
  action: "analyze"
  decision: "verify"
  conditions:
    max_tokens: 4000  # Only verify if exceeds threshold
  reason: "Large analysis requires human oversight"
```

### Policy Evaluation Flow

```
1. Agent calls method (e.g., query_documents())
2. BaseAgent.evaluate_action() invoked
3. PolicyEngine searches for matching rule
4. Conditions evaluated against action context
5. Decision returned (ALLOW/BLOCK/VERIFY/FLAG)
6. Action executed or rejected based on decision
```

## API Endpoints

Run the FastAPI server:

```bash
poetry run uvicorn api.main:app --reload
```

Access documentation at: `http://localhost:8000/docs`

## Testing

```bash
# Run all tests
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=agents --cov=governance --cov=api

# Run specific test file
poetry run pytest tests/agents/test_retriever_agent.py -v
```

**Test Results:**
- ✅ 11 tests passing
- ✅ Governance: 11 tests (policy loading, evaluation logic)
- ✅ Demo: End-to-end validation with RAG decision visibility

## Technology Stack

- **LLM Framework**: LangChain 0.1.0+ with OpenAI integration
- **LLM Model**: GPT-3.5-turbo (temperature=0.0 for deterministic reasoning)
- **Agent Pattern**: ReAct (Reasoning + Acting)
- **Vector Store**: Qdrant (in-memory mode)
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **API Framework**: FastAPI 0.109+
- **Validation**: Pydantic v2
- **Configuration**: YAML policies with runtime loading
- **Logging**: Structured logging with extra fields

## Roadmap

### Immediate Priorities

- [ ] **Observability Integration**: Connect structured logs to monitoring platforms (Datadog, Grafana, Prometheus)
- [ ] **Metrics Dashboard**: Export key metrics (policy violations, RAG usage rate, tool execution time, LLM token consumption)
- [ ] **Production Monitoring**: Add alerting for governance failures, blocked actions, and abnormal agent behavior patterns

### Future Enhancements

- [ ] Add more sophisticated policy conditions (regex patterns, context-aware rules)
- [ ] Implement VERIFY decision workflow (human-in-the-loop)
- [ ] Add policy versioning and A/B testing
- [ ] Multi-agent orchestration with shared governance
- [ ] Policy analytics dashboard (Streamlit/Gradio)
- [ ] Performance benchmarks for policy evaluation overhead

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/new-feature`)
3. Write tests for new functionality (maintain >90% coverage)
4. Ensure all tests pass (`poetry run pytest -v`)
5. Update documentation as needed
6. Submit a pull request with clear description

### Development Setup

```bash
# Install dev dependencies
poetry install --with dev

# Run linter
poetry run ruff check .

# Format code
poetry run black .

# Type checking
poetry run mypy agents/ governance/
```


