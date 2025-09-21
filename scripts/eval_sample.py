#!/usr/bin/env python3
"""
Quick test script for BiasLens bias analysis.
"""

import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.retriever import DocumentRetriever
from app.heuristics import BiasHeuristics
from app.icl_explainer import ICLExplainer


def test_components():
    """Test individual components."""
    print("=" * 80)
    print("BIASLENS COMPONENT TESTING")
    print("=" * 80)
    
    # Test queries
    test_queries = [
        "The economy is booming under this administration with record job growth and rising wages.",
        "Corporate greed continues to exploit working families while the wealthy hoard resources.",
        "The bipartisan infrastructure bill represents a balanced approach to critical national needs.",
        "Free market solutions are driving innovation and creating prosperity for all Americans.",
        "Climate change demands immediate action to prevent catastrophic environmental damage."
    ]
    
    print("\n1. Testing Document Retriever...")
    print("-" * 40)
    try:
        retriever = DocumentRetriever()
        print("✓ Retriever loaded successfully")
        
        for i, query in enumerate(test_queries[:2], 1):
            print(f"\nQuery {i}: {query}")
            results = retriever.retrieve(query, top_k=2)
            for j, doc in enumerate(results, 1):
                print(f"  {j}. {doc['title']} ({doc['bias_label']}) - Score: {doc['score']:.3f}")
    except Exception as e:
        print(f"✗ Retriever failed: {e}")
    
    print("\n2. Testing Bias Heuristics...")
    print("-" * 40)
    try:
        heuristics = BiasHeuristics()
        print("✓ Heuristics loaded successfully")
        
        for i, query in enumerate(test_queries[:3], 1):
            print(f"\nQuery {i}: {query}")
            result = heuristics.calculate_bias_score(query)
            print(f"  Label: {result['tentative_label']}")
            print(f"  Score: {result['bias_score']:.3f}")
            print(f"  Confidence: {result['confidence']:.3f}")
            
            indicators = heuristics.get_bias_indicators(query)
            if indicators:
                print(f"  Indicators: {', '.join(indicators[:2])}")
    except Exception as e:
        print(f"✗ Heuristics failed: {e}")
    
    print("\n3. Testing ICL Explainer...")
    print("-" * 40)
    try:
        icl_explainer = ICLExplainer()
        print("✓ ICL Explainer loaded successfully")
        
        for i, query in enumerate(test_queries[:2], 1):
            print(f"\nQuery {i}: {query}")
            result = icl_explainer.analyze(query)
            
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Label: {result['tentative_label']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Rationale: {result['rationale'][:100]}...")
    except Exception as e:
        print(f"✗ ICL Explainer failed: {e}")


def test_integration():
    """Test integrated analysis pipeline."""
    print("\n" + "=" * 80)
    print("INTEGRATED ANALYSIS TESTING")
    print("=" * 80)
    
    test_queries = [
        "The administration's economic policies have created unprecedented prosperity for all Americans.",
        "Corporate greed is destroying the middle class while the wealthy get richer.",
        "The bipartisan infrastructure bill represents a compromise that addresses critical needs."
    ]
    
    try:
        # Initialize components
        print("Initializing components...")
        retriever = DocumentRetriever()
        heuristics = BiasHeuristics()
        icl_explainer = ICLExplainer()
        
        print("✓ All components loaded")
        
        # Test integrated analysis
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i} ---")
            print(f"Query: {query}")
            print("-" * 60)
            
            start_time = time.time()
            
            # Retrieve documents
            retrieved = retriever.retrieve(query, top_k=3)
            print(f"Retrieved {len(retrieved)} documents")
            
            # Heuristic analysis
            heuristic_result = heuristics.calculate_bias_score(query)
            print(f"Heuristic: {heuristic_result['tentative_label']} (score: {heuristic_result['bias_score']:.3f})")
            
            # ICL analysis
            icl_result = icl_explainer.analyze(query)
            if "error" not in icl_result:
                print(f"ICL: {icl_result['tentative_label']} (confidence: {icl_result['confidence']:.3f})")
            else:
                print(f"ICL Error: {icl_result['error']}")
            
            # Show retrieved documents
            print("\nRetrieved Documents:")
            for j, doc in enumerate(retrieved, 1):
                print(f"  {j}. {doc['title']} ({doc['bias_label']}) - {doc['score']:.3f}")
                print(f"     {doc['text'][:100]}...")
            
            elapsed = time.time() - start_time
            print(f"\nAnalysis completed in {elapsed:.2f} seconds")
            
    except Exception as e:
        print(f"✗ Integration test failed: {e}")


def test_api_simulation():
    """Simulate API calls."""
    print("\n" + "=" * 80)
    print("API SIMULATION TESTING")
    print("=" * 80)
    
    try:
        # Initialize components
        retriever = DocumentRetriever()
        heuristics = BiasHeuristics()
        icl_explainer = ICLExplainer()
        
        # Simulate API request
        query = "The economy is experiencing unprecedented growth with record job creation and rising wages across all sectors."
        
        print(f"Simulating API call for: {query}")
        print("-" * 60)
        
        # Retrieve documents
        retrieved_docs = retriever.retrieve(query, top_k=5)
        
        # Get heuristic analysis
        heuristic_result = heuristics.calculate_bias_score(query)
        
        # Get ICL analysis
        icl_result = icl_explainer.analyze(query, num_shots=2)
        
        # Format response (similar to API)
        formatted_retrieved = []
        for doc in retrieved_docs:
            formatted_retrieved.append({
                "text": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                "title": doc['title'],
                "source": doc['source'],
                "bias_label": doc['bias_label'],
                "score": doc['score']
            })
        
        formatted_score = {
            "heuristic_score": heuristic_result['bias_score'],
            "heuristic_label": heuristic_result['tentative_label'],
            "heuristic_confidence": heuristic_result['confidence'],
            "sentiment": "positive" if heuristic_result['sentiment_analysis']['polarity'] > 0 else "negative",
            "keywords": {
                "left": heuristic_result['keyword_analysis']['found_left'][:3],
                "right": heuristic_result['keyword_analysis']['found_right'][:3],
                "neutral": heuristic_result['keyword_analysis']['found_neutral'][:3]
            }
        }
        
        # Display results
        print("\nAPI Response Simulation:")
        print("=" * 40)
        
        print("\nAnswer (ICL Analysis):")
        print(json.dumps(icl_result, indent=2))
        
        print(f"\nRetrieved Documents ({len(formatted_retrieved)}):")
        for i, doc in enumerate(formatted_retrieved, 1):
            print(f"  {i}. {doc['title']} ({doc['bias_label']}) - Score: {doc['score']:.3f}")
        
        print("\nScore (Heuristics):")
        print(json.dumps(formatted_score, indent=2))
        
    except Exception as e:
        print(f"✗ API simulation failed: {e}")


def main():
    """Main test function."""
    print("BiasLens Evaluation Script")
    print("This script tests all components and the integrated pipeline.")
    print()
    
    # Check if index exists
    index_path = Path(__file__).parent.parent / 'app' / 'index'
    if not index_path.exists():
        print("⚠️  Warning: FAISS index not found.")
        print("   Run 'python scripts/index_docs.py' first to create the index.")
        print()
    
    # Run tests
    test_components()
    test_integration()
    test_api_simulation()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("If all tests passed, you can start the API server with:")
    print("  python app/server.py")
    print()
    print("Then visit http://localhost:8000/docs for API documentation.")


if __name__ == "__main__":
    main()
