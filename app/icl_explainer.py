"""
In-Context Learning (ICL) explainer for bias analysis using few-shot examples.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class ICLExplainer:
    """In-context learning explainer for bias analysis."""
    
    def __init__(self, model_name: str = "distilgpt2", shots_file: str = None):
        """
        Initialize the ICL explainer.
        
        Args:
            model_name: Hugging Face model name for text generation
            shots_file: Path to few-shot examples JSON file
        """
        self.model_name = model_name
        self.shots_file = shots_file or Path(__file__).parent.parent / 'examples' / 'shots.json'
        
        # Load few-shot examples
        self.shots = self._load_shots()
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_shots(self) -> List[Dict[str, Any]]:
        """Load few-shot examples from JSON file."""
        try:
            with open(self.shots_file, 'r', encoding='utf-8') as f:
                shots = json.load(f)
            print(f"Loaded {len(shots)} few-shot examples")
            return shots
        except FileNotFoundError:
            print(f"Warning: Shots file not found at {self.shots_file}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error loading shots file: {e}")
            return []
    
    def _load_model(self):
        """Load the language model for text generation."""
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.pipeline = None
    
    def _format_shot_example(self, shot: Dict[str, Any]) -> str:
        """Format a single shot example for the prompt."""
        text = shot['text']
        analysis = shot['analysis']
        
        formatted = f"""Text: "{text}"
Analysis: {{
  "evidence_spans": {json.dumps(analysis['evidence_spans'])},
  "indicators": {json.dumps(analysis['indicators'])},
  "tentative_label": "{analysis['tentative_label']}",
  "confidence": {analysis['confidence']},
  "rationale": "{analysis['rationale']}"
}}

"""
        return formatted
    
    def _build_prompt(self, query: str, num_shots: int = 2) -> str:
        """
        Build the ICL prompt with few-shot examples.
        
        Args:
            query: Text to analyze for bias
            num_shots: Number of few-shot examples to include
            
        Returns:
            Formatted prompt string
        """
        # Start with task description
        prompt = """You are an expert political bias analyst. Analyze the given text for political bias and provide a structured JSON response.

Task: Identify evidence of political bias in the text and classify it as "leans_left", "center", or "leans_right".

Response format:
{
  "evidence_spans": ["specific phrases that indicate bias"],
  "indicators": ["bias indicators found"],
  "tentative_label": "leans_left|center|leans_right",
  "confidence": 0.85,
  "rationale": "Explanation of the analysis"
}

Examples:

"""
        
        # Add few-shot examples
        shots_to_use = self.shots[:num_shots] if self.shots else []
        for shot in shots_to_use:
            prompt += self._format_shot_example(shot)
        
        # Add the query
        prompt += f"""Text: "{query}"
Analysis:"""
        
        return prompt
    
    def analyze(self, query: str, num_shots: int = 2) -> Dict[str, Any]:
        """
        Analyze text for bias using ICL.
        
        Args:
            query: Text to analyze
            num_shots: Number of few-shot examples to use
            
        Returns:
            Dictionary with bias analysis results
        """
        if not self.pipeline:
            return {
                "error": "Model not loaded",
                "evidence_spans": [],
                "indicators": [],
                "tentative_label": "center",
                "confidence": 0.0,
                "rationale": "Model unavailable"
            }
        
        # Build prompt
        prompt = self._build_prompt(query, num_shots)
        
        try:
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                truncation=True
            )
            
            generated_text = response[0]['generated_text']
            
            # Extract the analysis part (after "Analysis:")
            analysis_start = generated_text.find('Analysis:') + len('Analysis:')
            analysis_text = generated_text[analysis_start:].strip()
            
            # Try to parse JSON from the generated text
            try:
                # Look for JSON-like structure in the response
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = analysis_text[json_start:json_end]
                    analysis = json.loads(json_str)
                    
                    # Validate required fields
                    required_fields = ['evidence_spans', 'indicators', 'tentative_label', 'confidence', 'rationale']
                    for field in required_fields:
                        if field not in analysis:
                            analysis[field] = self._get_default_value(field)
                    
                    return analysis
                else:
                    raise ValueError("No JSON structure found")
                    
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: extract information using simple parsing
                return self._parse_fallback_response(analysis_text)
                
        except Exception as e:
            return {
                "error": f"Generation failed: {str(e)}",
                "evidence_spans": [],
                "indicators": [],
                "tentative_label": "center",
                "confidence": 0.0,
                "rationale": f"Analysis failed: {str(e)}"
            }
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields."""
        defaults = {
            'evidence_spans': [],
            'indicators': [],
            'tentative_label': 'center',
            'confidence': 0.5,
            'rationale': 'Unable to determine bias'
        }
        return defaults.get(field, None)
    
    def _parse_fallback_response(self, text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails."""
        # Simple keyword-based fallback analysis
        text_lower = text.lower()
        
        # Look for bias indicators
        left_indicators = ['progressive', 'liberal', 'left', 'social justice', 'inequality']
        right_indicators = ['conservative', 'traditional', 'right', 'free market', 'liberty']
        
        left_count = sum(1 for indicator in left_indicators if indicator in text_lower)
        right_count = sum(1 for indicator in right_indicators if indicator in text_lower)
        
        if left_count > right_count:
            label = "leans_left"
            confidence = min(0.3 + left_count * 0.1, 0.8)
        elif right_count > left_count:
            label = "leans_right"
            confidence = min(0.3 + right_count * 0.1, 0.8)
        else:
            label = "center"
            confidence = 0.3
        
        return {
            "evidence_spans": [],
            "indicators": [],
            "tentative_label": label,
            "confidence": confidence,
            "rationale": "Fallback analysis due to parsing error"
        }
    
    def get_available_shots(self) -> List[Dict[str, Any]]:
        """Get list of available few-shot examples."""
        return self.shots.copy()
    
    def add_shot_example(self, text: str, analysis: Dict[str, Any]) -> bool:
        """
        Add a new few-shot example.
        
        Args:
            text: Example text
            analysis: Analysis result for the text
            
        Returns:
            True if successfully added
        """
        try:
            new_shot = {
                "text": text,
                "analysis": analysis
            }
            self.shots.append(new_shot)
            
            # Save to file
            with open(self.shots_file, 'w', encoding='utf-8') as f:
                json.dump(self.shots, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error adding shot example: {e}")
            return False


def main():
    """Test the ICL explainer."""
    explainer = ICLExplainer()
    
    # Test queries
    test_queries = [
        "The economy is booming under this administration with record job growth and rising wages.",
        "Corporate greed is destroying the middle class while the wealthy get richer.",
        "The bipartisan infrastructure bill represents a balanced approach to national needs."
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 80)
        
        result = explainer.analyze(query)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Label: {result['tentative_label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Evidence: {result['evidence_spans']}")
            print(f"Indicators: {result['indicators']}")
            print(f"Rationale: {result['rationale']}")
        print()


if __name__ == "__main__":
    main()
