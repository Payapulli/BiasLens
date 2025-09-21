#!/usr/bin/env python3
"""
Build FAISS index from articles.jsonl for efficient similarity search.
"""

import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def load_articles(file_path):
    """Load articles from JSONL file."""
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles

def create_embeddings(articles, model):
    """Create embeddings for article content."""
    texts = []
    metadata = []
    
    for article in articles:
        # Combine title and content for better representation
        text = f"{article['title']} {article['content']}"
        texts.append(text)
        metadata.append({
            'title': article['title'],
            'source': article['source'],
            'date': article['date'],
            'bias_label': article.get('bias_label', 'unknown'),
            'content': article['content']
        })
    
    print(f"Creating embeddings for {len(texts)} articles...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, metadata

def build_faiss_index(embeddings, metadata, output_dir):
    """Build and save FAISS index."""
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    # Save index and metadata
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, 'faiss_index.bin')
    metadata_path = os.path.join(output_dir, 'metadata.json')
    
    faiss.write_index(index, index_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Index saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")
    return index_path, metadata_path

def main():
    """Main function to build FAISS index."""
    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    articles_path = data_dir / 'articles.jsonl'
    output_dir = Path(__file__).parent.parent / 'app' / 'index'
    
    # Check if articles file exists
    if not articles_path.exists():
        print(f"Error: Articles file not found at {articles_path}")
        sys.exit(1)
    
    # Load model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load articles
    print(f"Loading articles from {articles_path}")
    articles = load_articles(articles_path)
    print(f"Loaded {len(articles)} articles")
    
    # Create embeddings
    embeddings, metadata = create_embeddings(articles, model)
    
    # Build and save index
    index_path, metadata_path = build_faiss_index(embeddings, metadata, output_dir)
    
    print("Indexing complete!")
    print(f"Index dimension: {embeddings.shape[1]}")
    print(f"Number of vectors: {embeddings.shape[0]}")

if __name__ == "__main__":
    main()
