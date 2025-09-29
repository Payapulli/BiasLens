"""
Unit tests for FastAPI server functions.
"""
import pytest
from unittest.mock import Mock, patch
import asyncio


class TestServerFunctions:
    """Test cases for FastAPI server functions."""
    
    def test_parse_distilgpt2_response_valid(self):
        """Test parsing valid DistilGPT2 response."""
        with patch.dict('sys.modules', {
            'transformers': Mock(),
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'torch': Mock(),
            'retriever': Mock()
        }):
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from server import parse_distilgpt2_response
            
            response = """BIAS: leans_right
            CONFIDENCE: 0.85
            EVIDENCE: climate hoax, fake news
            RATIONALE: The text contains climate denial language"""
            
            result = parse_distilgpt2_response(response)
            
            assert result['bias'] == 'leans_right'
            assert 0.0 <= result['confidence'] <= 1.0
            assert result['confidence'] > 0.5
            assert 'climate' in result['evidence']
            assert any('hoax' in word for word in result['evidence'])  # Handle comma in "hoax,"
            assert 'climate denial' in result['rationale']
    
    def test_parse_distilgpt2_response_missing_fields(self):
        """Test parsing response with missing fields."""
        with patch.dict('sys.modules', {
            'transformers': Mock(),
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'torch': Mock(),
            'retriever': Mock()
        }):
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from server import parse_distilgpt2_response
            
            response = """BIAS: leans_left
            CONFIDENCE: 0.7"""
            
            result = parse_distilgpt2_response(response)
            
            assert result['bias'] == 'leans_left'
            assert 0.0 <= result['confidence'] <= 1.0
            assert result['evidence'] == []
            assert result['rationale'] == ""
    
    def test_extract_evidence_spans(self):
        """Test evidence extraction from response."""
        with patch.dict('sys.modules', {
            'transformers': Mock(),
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'torch': Mock(),
            'retriever': Mock()
        }):
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from server import extract_evidence_spans
            
            response = """BIAS: leans_right
            EVIDENCE: climate hoax, fake news, conspiracy
            RATIONALE: Denial language detected"""
            
            evidence = extract_evidence_spans(response)
            
            assert len(evidence) == 5  # Should be 5 words after splitting
            assert 'climate' in evidence
            assert any('hoax' in word for word in evidence)  # Handle comma in "hoax,"
            assert 'fake' in evidence
    
    def test_build_analysis_prompt(self):
        """Test building analysis prompt."""
        with patch.dict('sys.modules', {
            'transformers': Mock(),
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'torch': Mock(),
            'retriever': Mock()
        }):
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from server import build_analysis_prompt
            
            query = "climate change is a hoax"
            context = "Climate change is a serious threat..."
            few_shot_examples = []
            
            prompt = build_analysis_prompt(query, context, few_shot_examples)
            
            assert isinstance(prompt, str)
            assert query in prompt
            assert context in prompt
            assert "BIAS:" in prompt
            assert "CONFIDENCE:" in prompt
            assert "EVIDENCE:" in prompt
            assert "RATIONALE:" in prompt
    
    def test_analyze_with_distilgpt2_function_exists(self):
        """Test that analyze_with_distilgpt2 function exists and is callable."""
        with patch.dict('sys.modules', {
            'transformers': Mock(),
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'torch': Mock(),
            'retriever': Mock()
        }):
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from server import analyze_with_distilgpt2
            
            # Test the function exists and is callable
            assert callable(analyze_with_distilgpt2)
            
            # Test function signature
            import inspect
            sig = inspect.signature(analyze_with_distilgpt2)
            assert len(sig.parameters) == 1  # Should take only query parameter
    
    def test_health_check_function(self):
        """Test health check function."""
        with patch.dict('sys.modules', {
            'transformers': Mock(),
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'torch': Mock(),
            'retriever': Mock()
        }):
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from server import health_check
            
            # Test the health check function
            response = asyncio.run(health_check())
            
            # Should return a dict with status info
            assert isinstance(response, dict)
            assert 'status' in response
            assert response['status'] == 'healthy'
    
    def test_confidence_normalization(self):
        """Test confidence value normalization."""
        with patch.dict('sys.modules', {
            'transformers': Mock(),
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'torch': Mock(),
            'retriever': Mock()
        }):
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from server import parse_distilgpt2_response
            
            test_cases = [
                ("CONFIDENCE: 0.85", True),  # Valid confidence
                ("CONFIDENCE: 85", True),    # Percentage should be converted
                ("CONFIDENCE: 1.2", False),  # Invalid, should default
                ("CONFIDENCE: -0.1", False), # Invalid, should default
                ("CONFIDENCE: 0.05", False), # Too low, should default
            ]
            
            for confidence_text, should_be_valid in test_cases:
                response = f"BIAS: center\n{confidence_text}\nEVIDENCE: test\nRATIONALE: test"
                result = parse_distilgpt2_response(response)
                
                # Always check it's in valid range
                assert 0.0 <= result['confidence'] <= 1.0, f"Confidence {result['confidence']} not in valid range for {confidence_text}"
                
                # Check if it matches expected behavior
                if should_be_valid:
                    assert result['confidence'] > 0.5, f"Expected high confidence for {confidence_text}, got {result['confidence']}"
                else:
                    assert result['confidence'] == 0.7, f"Expected default confidence for {confidence_text}, got {result['confidence']}"
    
    def test_bias_label_normalization(self):
        """Test bias label normalization."""
        with patch.dict('sys.modules', {
            'transformers': Mock(),
            'sentence_transformers': Mock(),
            'faiss': Mock(),
            'torch': Mock(),
            'retriever': Mock()
        }):
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            
            from server import parse_distilgpt2_response
            
            test_cases = [
                ("BIAS: leans_left", "leans_left"),
                ("BIAS: LEANS_LEFT", "leans_left"),
                ("BIAS: leans_right", "leans_right"),
                ("BIAS: center", "center"),
                ("BIAS: CENTER", "center"),
                ("BIAS: unknown", "center"),  # Should default to center
            ]
            
            for bias_text, expected in test_cases:
                response = f"{bias_text}\nCONFIDENCE: 0.8\nEVIDENCE: test\nRATIONALE: test"
                result = parse_distilgpt2_response(response)
                assert result['bias'] == expected, f"Failed for {bias_text}"
