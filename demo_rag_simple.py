#!/usr/bin/env python3
"""
Simple RAG Agent Demo

Demonstrates GovernedRAGAgent with OpenAI LLM and policy enforcement.
The LLM decides whether to use vector_retrieval or answer directly.
Includes structured logging for complete traceability.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from governance.policy_loader import PolicyLoader
from governance.policy_engine import PolicyEngine
from agents.rag_agent import GovernedRAGAgent


def setup_logging():
    """Configure structured logging for agent traceability."""
    
    class StructuredFormatter(logging.Formatter):
        """Custom formatter that adds structured fields to logs."""
        
        def format(self, record):
            base = super().format(record)
            extras = []
            
            # Extract structured fields
            for key in ['agent', 'model', 'tool', 'decision', 'rule_id',
                       'elapsed_ms', 'documents', 'query', 'action',
                       'reason', 'error']:
                if hasattr(record, key):
                    value = getattr(record, key)
                    if key == 'query' and isinstance(value, str):
                        # Truncate long queries
                        value = value[:50] + '...' if len(value) > 50 else value
                    extras.append(f"{key}={value}")
            
            if extras:
                return f"{base} | {' | '.join(extras)}"
            return base
    
    # Setup root logger
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]
    
    # Reduce noise from external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)


def main():
    # Load environment variables
    load_dotenv()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Set it in .env file or export OPENAI_API_KEY='your-key'")
        return
    
    # Setup logging first
    setup_logging()
    
    print("\n" + "=" * 70)
    print("GOVERNED RAG AGENT - Demo with Structured Logging")
    print("=" * 70)
    print("\n[INFO] Check logs below for detailed agent behavior traceability\n")
    
    # Setup
    print("[Setup] Initializing agent...")
    policy_dir = Path("config/policies")
    loader = PolicyLoader(policy_dir)
    policies = loader.load_all_policies()
    
    from governance.models import DecisionType
    engine = PolicyEngine(
        policies=policies,
        default_decision=DecisionType.BLOCK
    )
    
    agent = GovernedRAGAgent(
        name="retriever",
        policy_engine=engine,
        llm_model="gpt-3.5-turbo",
        temperature=0.0
    )
    
    # Load documents
    print("[Setup] Loading documents...")
    docs_dir = Path("data/docs")
    result = agent.load_documents(docs_dir)
    
    if result["status"] != "success":
        print(f"❌ Failed to load documents: {result['message']}")
        return
    
    print(f"✓ Loaded {result['documents_loaded']} documents\n")
    
    # Test queries
    queries = [
        "What information do you have about machine learning?",
        "Hello! How are you?",
        "Find details about vector databases",
    ]
    
    print("=" * 70)
    print("TESTING QUERIES")
    print("=" * 70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: {query}")
        print(f"{'─' * 70}")
        
        result = agent.ask(query)
        
        if result["status"] == "success":
            print(f"\n✅ Answer:\n{result['answer']}\n")
            print(f"Decision: {result['decision']} (rule: {result['rule_id']})")
        
        elif result["status"] == "blocked":
            print(f"\n❌ Blocked: {result['reason']}")
            print(f"Rule: {result['rule_id']}")
        
        elif result["status"] == "error":
            print(f"\n❌ Error: {result['message']}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETED")
    print("=" * 70)
    print("\nRefactored architecture demonstrated:")
    print("  ✓ No heuristics - pure policy-based decisions")
    print("  ✓ Separated callbacks (PolicyEnforcement + Observability)")
    print("  ✓ Dynamic tool metadata (policy_action attribute)")
    print("  ✓ Homogeneous policy evaluation interface")
    print("  ✓ New governed agent prompt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
