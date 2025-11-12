# Agent Governance Hub

A lightweight, policy-driven governance middleware for LLM agents. Control agent behavior through declarative YAML policies, ensuring safe and compliant AI operations.

## Overview

Agent Governance Hub provides a **policy engine** that evaluates every agent action before execution. Define rules in YAML, and the system automatically enforces them—allowing, blocking, or flagging actions based on configurable conditions.

### Key Features

- **Declarative Policies**: Define governance rules in simple YAML files
- **Policy Enforcement**: Automatic evaluation before every agent action
- **Vector-Based Retrieval**: Built-in document search with Qdrant + HuggingFace embeddings
- **FastAPI Integration**: RESTful API for policy evaluation and agent management
- **Type Safety**: Pydantic validation for policies and configurations
- **Extensible Architecture**: Base agent class for creating governed agents

## Architecture

```
┌─────────────────┐
│  YAML Policies  │  ← Define rules (allow/block/verify/flag)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Policy Engine   │  ← Evaluates actions against rules
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Base Agent    │  ← Enforces decisions automatically
└────────┬────────┘
         │
         ├─────────► RetrieverAgent
         └─────────► AnalyzerAgent
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/MiguelBarriosAl/agent-governance-hub.git
cd agent-governance-hub

# Install dependencies
poetry install

# Run tests
poetry run pytest -v
```

### Basic Usage

#### 1. Define Policies

Create rules in `config/policies/default.yaml`:

```yaml
version: "1.0"
policies:
  - agent_id: "retriever"
    description: "Rules for data retrieval agent"
    rules:
      - id: "R001"
        action: "query_database"
        decision: "allow"
        conditions: {}
        reason: "Read-only queries are permitted"

      - id: "R002"
        action: "delete_data"
        decision: "block"
        conditions: {}
        reason: "Destructive operations are forbidden"
```

## Project Structure

```
agent-governance-hub/
├── agents/              # Governed agent implementations
│   ├── base_agent.py    # Abstract base with policy enforcement
│   └── retriever_agent.py  # Vector search agent
├── api/                 # FastAPI application
│   ├── main.py          # App factory and endpoints
│   └── exceptions.py    # Exception handlers
├── config/              # Configuration
│   ├── settings.py      # Application settings
│   └── policies/        # YAML policy files
│       └── default.yaml
├── governance/          # Policy engine core
│   ├── models.py        # Pydantic models
│   ├── policy_loader.py # YAML loading & validation
│   └── policy_engine.py # Rule evaluation logic
├── tests/               # Test suite
│   ├── agents/          # Agent tests (11 tests)
│   └── governance/      # Policy tests (11 tests)
├── data/docs/           # Sample documents
├── demo_retriever.py    # Interactive demo
└── pyproject.toml       # Dependencies
```

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

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/new-agent`)
3. Write tests for new functionality
4. Ensure all tests pass (`poetry run pytest -v`)
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

**Author**: Miguel Barrios  
**Repository**: [github.com/MiguelBarriosAl/agent-governance-hub](https://github.com/MiguelBarriosAl/agent-governance-hub)

---
