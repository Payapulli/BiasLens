"""
FAISS-based document retriever for bias analysis.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentRetriever:
    """Retrieves relevant documents using FAISS similarity search."""
    
    def __init__(self, index_dir: str = None):
        """Initialize the retriever with FAISS index."""
        if index_dir is None:
            index_dir = Path(__file__).parent / 'index'
        
        self.index_dir = Path(index_dir)
        self.model = None
        self.index = None
        self.metadata = None
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        index_path = self.index_dir / 'faiss_index.bin'
        metadata_path = self.index_dir / 'metadata.json'
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Index files not found. Run 'python scripts/index_docs.py' first.\n"
                f"Expected files: {index_path}, {metadata_path}"
            )
        
        # Load sentence transformer model
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load FAISS index
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded index with {self.index.ntotal} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar documents to the query.
        
        Args:
            query: Text query to search for
            top_k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing retrieved documents with scores
        """
        if self.model is None or self.index is None:
            raise RuntimeError("Retriever not properly initialized")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
                
            doc_metadata = self.metadata[idx]
            results.append({
                'text': doc_metadata['content'],
                'title': doc_metadata['title'],
                'source': doc_metadata['source'],
                'date': doc_metadata['date'],
                'bias_label': doc_metadata['bias_label'],
                'score': float(score),
                'rank': i + 1
            })
        
        return results
    
    def get_similar_by_bias(self, query: str, bias_label: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve documents with specific bias label.
        
        Args:
            query: Text query to search for
            bias_label: Target bias label ('leans_left', 'center', 'leans_right')
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with the specified bias
        """
        all_results = self.retrieve(query, top_k * 3)  # Get more to filter
        filtered_results = [
            doc for doc in all_results 
            if doc['bias_label'] == bias_label
        ][:top_k]
        
        return filtered_results
    
    def get_balanced_retrieval(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Retrieve balanced set of documents across different bias labels.
        
        Args:
            query: Text query to search for
            top_k: Total number of documents to retrieve
            
        Returns:
            List of retrieved documents balanced across bias labels
        """
        all_results = self.retrieve(query, top_k * 2)  # Get more to balance
        
        # Group by bias label
        by_bias = {}
        for doc in all_results:
            bias = doc['bias_label']
            if bias not in by_bias:
                by_bias[bias] = []
            by_bias[bias].append(doc)
        
        # Balance results
        balanced_results = []
        labels = list(by_bias.keys())
        per_label = max(1, top_k // len(labels)) if labels else 0
        
        for label in labels:
            balanced_results.extend(by_bias[label][:per_label])
        
        # Fill remaining slots with highest scoring documents
        remaining = top_k - len(balanced_results)
        if remaining > 0:
            used_indices = {i for doc in balanced_results for i in range(len(all_results)) 
                          if all_results[i] == doc}
            additional = [doc for i, doc in enumerate(all_results) 
                         if i not in used_indices][:remaining]
            balanced_results.extend(additional)
        
        return balanced_results[:top_k]


def main():
    """Test the retriever."""
    try:
        retriever = DocumentRetriever()
        
        # Test queries
        test_queries = [
            "economic growth and job creation",
            "climate change and environmental policy",
            "tax cuts and corporate profits"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 50)
            
            results = retriever.retrieve(query, top_k=3)
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc['title']} ({doc['bias_label']}) - Score: {doc['score']:.3f}")
                print(f"   Source: {doc['source']}")
                print(f"   Text: {doc['text'][:100]}...")
                print()
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run 'python scripts/index_docs.py' first")


if __name__ == "__main__":
    main()
