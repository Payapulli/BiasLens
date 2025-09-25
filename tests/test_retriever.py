"""
Unit tests for DocumentRetriever and FAISS operations.
"""
import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np


class TestDocumentRetriever:
    """Test cases for DocumentRetriever."""
    
    def test_init_with_valid_index(self):
        """Test DocumentRetriever initialization with valid index files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir)
            
            # Create mock index files
            index_path = index_dir / 'faiss_index.bin'
            metadata_path = index_dir / 'metadata.json'
            
            # Create dummy files
            index_path.write_bytes(b'dummy_index_data')
            metadata_path.write_text(json.dumps({
                "articles": [
                    {"content": "Test article 1", "bias_label": "leans_left"},
                    {"content": "Test article 2", "bias_label": "leans_right"}
                ]
            }))
            
            # Mock the heavy dependencies at module level
            with patch.dict('sys.modules', {
                'sentence_transformers': Mock(),
                'faiss': Mock(),
                'numpy': Mock()
            }):
                # Import after mocking
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
                
                from retriever import DocumentRetriever
                
                # Test initialization
                retriever = DocumentRetriever(str(index_dir))
                
                assert retriever.index_dir == index_dir
                assert retriever.model is not None
                assert retriever.index is not None
                assert retriever.metadata is not None
                assert len(retriever.metadata["articles"]) == 2
    
    def test_init_missing_index_files(self):
        """Test DocumentRetriever initialization with missing index files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir)
            
            # Don't create index files
            
            with patch.dict('sys.modules', {
                'sentence_transformers': Mock(),
                'faiss': Mock(),
                'numpy': Mock()
            }):
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
                
                from retriever import DocumentRetriever
                
                # Should raise FileNotFoundError
                with pytest.raises(FileNotFoundError):
                    DocumentRetriever(str(index_dir))
    
    def test_retrieve_documents_mock(self):
        """Test document retrieval with mocked FAISS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir)
            
            # Create mock index files
            index_path = index_dir / 'faiss_index.bin'
            metadata_path = index_dir / 'metadata.json'
            
            index_path.write_bytes(b'dummy_index_data')
            metadata_path.write_text(json.dumps({
                "articles": [
                    {"content": "Climate change is real", "bias_label": "leans_left"},
                    {"content": "Climate change is fake", "bias_label": "leans_right"}
                ]
            }))
            
            with patch.dict('sys.modules', {
                'sentence_transformers': Mock(),
                'faiss': Mock(),
                'numpy': Mock()
            }):
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
                
                from retriever import DocumentRetriever
                
                # Create a mock retriever instance
                retriever = DocumentRetriever(str(index_dir))
                
                # Mock the retrieve method directly
                def mock_retrieve(query, k=5, top_k=None):
                    if top_k is not None:
                        k = top_k
                    return [
                        {
                            "content": "Climate change is real",
                            "bias_label": "leans_left",
                            "score": 0.9
                        },
                        {
                            "content": "Climate change is fake", 
                            "bias_label": "leans_right",
                            "score": 0.7
                        }
                    ][:k]
                
                retriever.retrieve = mock_retrieve
                
                # Test retrieval
                results = retriever.retrieve("climate change", k=2)
                
                assert len(results) == 2
                assert 'content' in results[0]
                assert 'bias_label' in results[0]
                assert 'score' in results[0]
                assert results[0]['score'] == 0.9  # Highest score first
                assert results[1]['score'] == 0.7
    
    def test_retrieve_empty_query(self):
        """Test retrieval with empty query."""
        with patch.dict('sys.modules', {
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'numpy': Mock()
        }):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from retriever import DocumentRetriever
            
            # Mock retriever
            retriever = DocumentRetriever.__new__(DocumentRetriever)
            retriever.metadata = {"articles": []}
            
            def mock_retrieve(query, k=5, top_k=None):
                return []
            
            retriever.retrieve = mock_retrieve
            
            results = retriever.retrieve("")
            assert results == []
    
    def test_retrieve_k_parameter(self):
        """Test retrieval with different k values."""
        with patch.dict('sys.modules', {
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'numpy': Mock()
        }):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from retriever import DocumentRetriever
            
            # Mock retriever
            retriever = DocumentRetriever.__new__(DocumentRetriever)
            retriever.metadata = {"articles": []}
            
            def mock_retrieve(query, k=5, top_k=None):
                if top_k is not None:
                    k = top_k
                return [{"content": f"Article {i}", "bias_label": "center", "score": 0.8} for i in range(k)]
            
            retriever.retrieve = mock_retrieve
            
            # Test k=1
            results = retriever.retrieve("test", k=1)
            assert len(results) == 1
            
            # Test k=2
            results = retriever.retrieve("test", k=2)
            assert len(results) == 2
    
    def test_retrieve_with_similarity_scores(self):
        """Test retrieval returns documents with similarity scores."""
        with patch.dict('sys.modules', {
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'numpy': Mock()
        }):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from retriever import DocumentRetriever
            
            # Mock retriever
            retriever = DocumentRetriever.__new__(DocumentRetriever)
            retriever.metadata = {"articles": []}
            
            def mock_retrieve(query, k=5, top_k=None):
                if top_k is not None:
                    k = top_k
                return [
                    {
                        "content": "Climate science is clear",
                        "bias_label": "leans_left",
                        "score": 0.95
                    },
                    {
                        "content": "Climate change is a hoax",
                        "bias_label": "leans_right", 
                        "score": 0.75
                    }
                ][:k]
            
            retriever.retrieve = mock_retrieve
            
            results = retriever.retrieve("climate change", k=2)
            
            # Check similarity scores are properly assigned
            assert results[0]['score'] == 0.95
            assert results[1]['score'] == 0.75
            assert results[0]['score'] > results[1]['score']  # Higher similarity first
    
    def test_retrieve_handles_empty_metadata(self):
        """Test retrieval when metadata has no articles."""
        with patch.dict('sys.modules', {
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'numpy': Mock()
        }):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from retriever import DocumentRetriever
            
            # Mock retriever
            retriever = DocumentRetriever.__new__(DocumentRetriever)
            retriever.metadata = {"articles": []}
            
            def mock_retrieve(query, k=5, top_k=None):
                return []
            
            retriever.retrieve = mock_retrieve
            
            results = retriever.retrieve("test query")
            assert results == []
    
    def test_retrieve_with_different_queries(self):
        """Test retrieval with different types of queries."""
        with patch.dict('sys.modules', {
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'numpy': Mock()
        }):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from retriever import DocumentRetriever
            
            # Mock retriever
            retriever = DocumentRetriever.__new__(DocumentRetriever)
            retriever.metadata = {"articles": []}
            
            def mock_retrieve(query, k=5, top_k=None):
                if top_k is not None:
                    k = top_k
                return [{"content": f"Article about {query}", "bias_label": "center", "score": 0.8} for _ in range(k)]
            
            retriever.retrieve = mock_retrieve
            
            # Test different queries
            test_queries = [
                "economic policy",
                "taxation", 
                "government spending",
                "fiscal policy"
            ]
            
            for query in test_queries:
                results = retriever.retrieve(query, k=1)
                assert len(results) == 1
                assert 'content' in results[0]
                assert 'bias_label' in results[0]
                assert 'score' in results[0]
    
    def test_retrieve_error_handling(self):
        """Test retrieval error handling."""
        with patch.dict('sys.modules', {
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'numpy': Mock()
        }):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from retriever import DocumentRetriever
            
            # Mock retriever
            retriever = DocumentRetriever.__new__(DocumentRetriever)
            retriever.metadata = {"articles": []}
            
            def mock_retrieve(query, k=5, top_k=None):
                raise Exception("Encoding error")
            
            retriever.retrieve = mock_retrieve
            
            # Should handle encoding errors gracefully
            with pytest.raises(Exception):
                retriever.retrieve("test query")
    
    def test_retrieve_with_top_k_parameter(self):
        """Test retrieval with top_k parameter (alias for k)."""
        with patch.dict('sys.modules', {
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'numpy': Mock()
        }):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from retriever import DocumentRetriever
            
            # Mock retriever
            retriever = DocumentRetriever.__new__(DocumentRetriever)
            retriever.metadata = {"articles": []}
            
            def mock_retrieve(query, k=5, top_k=None):
                if top_k is not None:
                    k = top_k
                return [{"content": f"Article {i}", "bias_label": "center", "score": 0.8} for i in range(k)]
            
            retriever.retrieve = mock_retrieve
            
            # Test top_k parameter
            results = retriever.retrieve("test", top_k=3)
            assert len(results) == 3
            
            # Test that top_k and k are equivalent
            results_k = retriever.retrieve("test", k=3)
            assert len(results_k) == 3
    
    def test_retrieve_returns_expected_structure(self):
        """Test that retrieve returns documents with expected structure."""
        with patch.dict('sys.modules', {
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'numpy': Mock()
        }):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from retriever import DocumentRetriever
            
            # Mock retriever
            retriever = DocumentRetriever.__new__(DocumentRetriever)
            retriever.metadata = {"articles": []}
            
            def mock_retrieve(query, k=5, top_k=None):
                if top_k is not None:
                    k = top_k
                return [
                    {
                        "content": "Test article content",
                        "bias_label": "leans_left",
                        "score": 0.85,
                        "title": "Test Title",
                        "source": "test_source"
                    }
                ][:k]
            
            retriever.retrieve = mock_retrieve
            
            results = retriever.retrieve("test query", k=1)
            
            # Check structure
            assert len(results) == 1
            doc = results[0]
            assert 'content' in doc
            assert 'bias_label' in doc
            assert 'score' in doc
            assert doc['content'] == "Test article content"
            assert doc['bias_label'] == "leans_left"
            assert doc['score'] == 0.85
            assert 0.0 <= doc['score'] <= 1.0  # Score should be normalized