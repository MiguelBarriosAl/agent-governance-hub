#!/usr/bin/env python3
"""
Simple RAG Agent Demo

Demonstrates GovernedRAGAgent with OpenAI LLM and policy enforcement.
The LLM decides whether to use vector_retrieval or answer directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from governance.policy_loader import PolicyLoader
from governance.policy_engine import PolicyEngine
from agents.rag_agent import GovernedRAGAgent


def main():
    # Load environment variables
    load_dotenv()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Set it in .env file or export OPENAI_API_KEY='your-key'")
        return
    
    print("\n" + "=" * 70)
    print("GOVERNED RAG AGENT - Simple Demo")
    print("=" * 70)
    
    # Setup
    print("\n[Setup] Initializing agent...")
    policy_dir = Path("config/policies")
    loader = PolicyLoader(policy_dir)
    policies = loader.load_all_policies()
    engine = PolicyEngine(policies=policies, default_decision="BLOCK")
    
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
    print("Demo completed!")
    print("The LLM decided autonomously when to use vector_retrieval.")
    print("All actions were governed by policies.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
