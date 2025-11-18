#!/usr/bin/env python3
"""
Governed RAG Agent - Main Demo

Architecture:
  1. DocumentPipeline: Loads docs → Creates vector store
  2. PolicyEngine: Evaluates actions against YAML rules
  3. GovernedRAGAgent: Processes queries with LLM + policy enforcement
"""
from pathlib import Path
from dotenv import load_dotenv

from governance.policy_loader import PolicyLoader
from governance.policy_engine import PolicyEngine
from governance.models import DecisionType
from pipelines.document_pipeline import DocumentPipeline
from agents.rag_agent import GovernedRAGAgent


def main():
    load_dotenv()
    
    # Step 1: Load policies from YAML
    loader = PolicyLoader(Path("config/policies"))
    policies = loader.load_all_policies()
    engine = PolicyEngine(policies, default_decision=DecisionType.BLOCK)
    
    # Step 2: Load documents into vector store (setup phase)
    pipeline = DocumentPipeline(collection_name="retriever_documents")
    result = pipeline.load_documents(Path("data/docs"))
    
    if result["status"] != "success":
        print(f"Error loading documents: {result['message']}")
        return
    
    print(f"Loaded {result['documents_loaded']} documents\n")
    
    # Step 3: Create agent with pre-loaded vector store
    agent = GovernedRAGAgent(
        name="retriever",
        policy_engine=engine,
        vector_manager=pipeline.get_vector_manager(),
        llm_model="gpt-3.5-turbo",
        temperature=0.0
    )
    
    # Step 4: Process queries
    queries = [
        "What information do you have about machine learning?",
        "Hello! How are you?",
        "Find details about vector databases",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        result = agent.ask(query)
        
        if result["status"] == "success":
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Decision: {result['decision']} ({result['rule_id']})")
        else:
            print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
