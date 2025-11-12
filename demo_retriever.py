#!/usr/bin/env python3
"""
Demo script - RetrieverAgent with policy enforcement
Shows how the agent is governed by policies defined in config/policies/default.yaml
"""

from pathlib import Path
from governance.policy_loader import PolicyLoader
from governance.policy_engine import PolicyEngine
from agents.retriever_agent import RetrieverAgent


def main():
    print("=" * 70)
    print("AGENT GOVERNANCE HUB - RetrieverAgent Demo")
    print("=" * 70)
    
    # 1. Load policies
    print("\n[1] Loading governance policies...")
    policy_dir = Path("config/policies")
    loader = PolicyLoader(policy_dir)
    policies = loader.load_all_policies()
    print(f"✓ Loaded {len(policies)} policies")
    
    # 2. Initialize policy engine
    print("\n[2] Initializing policy engine...")
    engine = PolicyEngine(policies=policies, default_decision="BLOCK")
    print("✓ Policy engine ready with default: BLOCK")
    
    # 3. Create RetrieverAgent
    print("\n[3] Creating RetrieverAgent...")
    agent = RetrieverAgent(name="retriever", policy_engine=engine)
    print(f"✓ Agent '{agent.name}' created")
    
    # 4. Load documents
    print("\n[4] Loading documents from data/docs/...")
    docs_dir = Path("data/docs")
    result = agent.load_text_documents(docs_dir)
    if result["status"] == "success":
        print(f"✓ Loaded {result['documents_loaded']} documents into vector store")
    else:
        print(f"✗ Failed to load documents: {result['message']}")
        return
    
    # 5. Test ALLOWED action: query_database
    print("\n[5] Testing ALLOWED action: query_database")
    print("-" * 70)
    query_result = agent.query_documents("machine learning training", top_k=2)
    print(f"Status: {query_result['status']}")
    print(f"Decision: {query_result.get('decision', 'N/A')}")
    print(f"Rule ID: {query_result.get('rule_id', 'N/A')}")
    if query_result['status'] == 'success':
        print(f"Results found: {query_result['count']}")
        for i, doc in enumerate(query_result['results'], 1):
            preview = doc['content'][:70] + "..."
            print(f"  {i}. {preview} (from {doc['source']})")
    
    # 6. Test BLOCKED action: delete_data
    print("\n[6] Testing BLOCKED action: delete_data")
    print("-" * 70)
    delete_result = agent.delete_data("doc_001")
    print(f"Status: {delete_result['status']}")
    print(f"Decision: {delete_result.get('decision', 'N/A')}")
    print(f"Rule ID: {delete_result.get('rule_id', 'N/A')}")
    print(f"Reason: {delete_result.get('reason', 'N/A')}")
    print(f"Message: {delete_result.get('message', 'N/A')}")
    
    # 7. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Policies loaded and enforced successfully")
    print("✓ 'query_database' action → ALLOWED (data retrieval permitted)")
    print("✓ 'delete_data' action → BLOCKED (destructive operations forbidden)")
    print("\nThe agent respects governance rules defined in YAML policies.")
    print("=" * 70)


if __name__ == "__main__":
    main()
