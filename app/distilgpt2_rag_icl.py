"""
REAL RAG + ICL implementation using actual DistilGPT2 LLM inference.
"""

import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

from retriever import DocumentRetriever
from heuristics import BiasHeuristics
from icl_explainer import ICLExplainer


class DistilGPT2RAGICL:
    """Real RAG + ICL using actual DistilGPT2 LLM inference with retrieved context."""
    
    def __init__(self):
        """Initialize the real RAG + ICL system with DistilGPT2."""
        self.retriever = DocumentRetriever()
        self.heuristics = BiasHeuristics()
        
        # Initialize DistilGPT2 ICL explainer
        print("Loading DistilGPT2 for ICL...")
        self.icl_explainer = ICLExplainer()
        print("✅ DistilGPT2 loaded successfully")
        
        # Load few-shot examples for ICL
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _load_few_shot_examples(self) -> List[Dict]:
        """Load few-shot examples for ICL."""
        examples_path = Path(__file__).parent.parent / "examples" / "shots.json"
        try:
            with open(examples_path, 'r') as f:
                examples = json.load(f)
                # Convert to simpler format for ICL
                formatted_examples = []
                for ex in examples:
                    formatted_examples.append({
                        "text": ex["text"],
                        "bias": ex["analysis"]["tentative_label"],
                        "reasoning": ex["analysis"]["rationale"]
                    })
                return formatted_examples
        except Exception as e:
            print(f"Warning: Could not load few-shot examples: {e}")
            # Fallback examples
            return [
                {
                    "text": "Corporate greed is destroying our economy",
                    "bias": "leans_left",
                    "reasoning": "Uses 'corporate greed' and 'destroying' - classic left-wing economic critique"
                },
                {
                    "text": "The free market will solve all our problems",
                    "bias": "leans_right", 
                    "reasoning": "Promotes free market ideology without government intervention"
                },
                {
                    "text": "We need bipartisan solutions to address climate change",
                    "bias": "center",
                    "reasoning": "Calls for cooperation and compromise, avoiding partisan language"
                }
            ]
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Real RAG + ICL analysis using actual DistilGPT2 LLM inference.
        
        Args:
            query: Text to analyze for bias
            
        Returns:
            Complete analysis with real RAG + ICL reasoning
        """
        print(f"🔍 DISTILGPT2 RAG + ICL Analysis: {query[:50]}...")
        
        # Step 1: RAG - Retrieve relevant documents
        print("📚 Retrieving relevant documents...")
        retrieved_docs = self.retriever.retrieve(query, top_k=5)
        print(f"Found {len(retrieved_docs)} relevant documents")
        
        # Step 2: Build context from retrieved documents
        context = self._build_rag_context(retrieved_docs)
        
        # Step 3: ICL - Use DistilGPT2 with few-shot examples + retrieved context
        print("🧠 Running DISTILGPT2 ICL with retrieved context...")
        icl_result = self._distilgpt2_icl_analysis(query, context, retrieved_docs)
        
        # Step 4: Combine with heuristics for validation
        print("🔧 Running heuristic validation...")
        heuristic_result = self.heuristics.calculate_bias_score(query)
        
        # Step 5: Final result
        return self._combine_results(query, icl_result, heuristic_result, retrieved_docs)
    
    def _build_rag_context(self, retrieved_docs: List[Dict]) -> str:
        """Build rich context from retrieved documents for RAG."""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs[:3], 1):
            context_parts.append(f"""
Document {i}:
- Source: {doc['source']}
- Bias Label: {doc['bias_label']}
- Similarity Score: {doc['score']:.3f}
- Content: {doc['text'][:400]}...
""")
        
        return "\n".join(context_parts)
    
    def _distilgpt2_icl_analysis(self, query: str, context: str, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """
        REAL ICL analysis using actual DistilGPT2 LLM inference.
        """
        
        # Build the ICL prompt with few-shot examples and RAG context
        icl_prompt = self._build_distilgpt2_prompt(query, context, retrieved_docs)
        
        print("🤖 Running DistilGPT2 inference...")
        print(f"Prompt length: {len(icl_prompt)} characters")
        
        try:
            # Use the actual DistilGPT2 ICL explainer
            result = self.icl_explainer.analyze(icl_prompt, num_shots=2)
            
            if result and result.get('confidence', 0) > 0.1:
                # Enhance result with RAG context information
                result['rag_context_used'] = len(retrieved_docs)
                result['few_shot_examples_used'] = len(self.few_shot_examples)
                result['distilgpt2_inference'] = True
                
                # Add context-aware rationale
                if 'rationale' not in result:
                    result['rationale'] = "DistilGPT2 analysis using retrieved context documents."
                else:
                    result['rationale'] = f"DistilGPT2 RAG + ICL analysis using {len(retrieved_docs)} retrieved articles and {len(self.few_shot_examples)} few-shot examples. " + result['rationale']
                
                print("✅ DistilGPT2 inference successful")
                return result
            else:
                print("⚠️ DistilGPT2 returned low confidence, using fallback")
                return self._fallback_analysis(query, retrieved_docs)
                
        except Exception as e:
            print(f"❌ DistilGPT2 inference failed: {e}")
            return self._fallback_analysis(query, retrieved_docs)
    
    def _build_distilgpt2_prompt(self, query: str, context: str, retrieved_docs: List[Dict]) -> str:
        """Build ICL prompt for DistilGPT2 with few-shot examples and RAG context."""
        
        # Few-shot examples
        examples_text = ""
        for i, example in enumerate(self.few_shot_examples, 1):
            examples_text += f"""
Example {i}:
Text: "{example['text']}"
Bias: {example['bias']}
Reasoning: {example['reasoning']}
"""
        
        # RAG context
        rag_context = f"""
Retrieved Context Documents:
{context}
"""
        
        # Main prompt for DistilGPT2
        prompt = f"""Analyze political bias in text using examples and context.

{examples_text}

{rag_context}

Analyze this text: "{query}"

Provide:
- evidence_spans: specific phrases indicating bias
- indicators: types of bias detected  
- tentative_label: leans_left, center, or leans_right
- confidence: 0-1 score
- rationale: explanation using context and examples

Analysis:"""
        
        return prompt
    
    def _fallback_analysis(self, query: str, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """Fallback analysis when DistilGPT2 fails."""
        print("Using fallback analysis...")
        
        # Simple context-aware analysis
        query_lower = query.lower()
        
        # Analyze against retrieved documents
        evidence_spans = []
        indicators = []
        reasoning_parts = []
        
        for doc in retrieved_docs:
            doc_text = doc['text'].lower()
            doc_bias = doc['bias_label']
            doc_similarity = doc['score']
            
            # Find common phrases
            common_words = set(query_lower.split()) & set(doc_text.split())
            if len(common_words) > 2:
                evidence_spans.extend(list(common_words))
                reasoning_parts.append(f"Similar to {doc['source']} ({doc_bias})")
                
                if doc_bias == "leans_left":
                    indicators.append(f"Similar to left-leaning source: {doc['source']}")
                elif doc_bias == "leans_right":
                    indicators.append(f"Similar to right-leaning source: {doc['source']}")
        
        # Simple bias determination
        if any(word in query_lower for word in ['fake', 'hoax', 'scam']):
            tentative_label = "leans_right"
            confidence = 0.7
        elif any(word in query_lower for word in ['greed', 'exploit', 'inequality']):
            tentative_label = "leans_left"
            confidence = 0.7
        else:
            tentative_label = "center"
            confidence = 0.5
        
        rationale = f"Fallback analysis using {len(retrieved_docs)} retrieved documents. " + " ".join(reasoning_parts[:2])
        
        return {
            "evidence_spans": list(set(evidence_spans))[:5],
            "indicators": indicators[:5],
            "tentative_label": tentative_label,
            "confidence": confidence,
            "rationale": rationale,
            "distilgpt2_inference": False,
            "fallback_used": True
        }
    
    def _combine_results(self, query: str, icl_result: Dict, heuristic_result: Dict, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """Combine ICL and heuristic results."""
        
        # Use ICL result as primary (it's the "real" analysis)
        answer = icl_result
        
        # Format retrieved documents
        formatted_retrieved = []
        for doc in retrieved_docs:
            formatted_retrieved.append({
                "text": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                "title": doc['title'],
                "source": doc['source'],
                "bias_label": doc['bias_label'],
                "score": doc['score']
            })
        
        # Format score
        formatted_score = {
            "heuristic_score": heuristic_result['bias_score'],
            "heuristic_label": heuristic_result['tentative_label'],
            "heuristic_confidence": heuristic_result['confidence'],
            "sentiment": "positive" if heuristic_result['sentiment_analysis']['polarity'] > 0 else "negative",
            "keywords": {
                "left": heuristic_result['keyword_analysis']['found_left'][:3],
                "right": heuristic_result['keyword_analysis']['found_right'][:3],
                "neutral": heuristic_result['keyword_analysis']['found_neutral'][:3]
            },
            "analysis_method": "DISTILGPT2 RAG + ICL" if icl_result.get('distilgpt2_inference') else "Fallback + RAG",
            "rag_context_used": len(retrieved_docs),
            "few_shot_examples_used": icl_result.get('few_shot_examples_used', 0),
            "distilgpt2_inference": icl_result.get('distilgpt2_inference', False),
            "fallback_used": icl_result.get('fallback_used', False)
        }
        
        return {
            "answer": answer,
            "retrieved": formatted_retrieved,
            "score": formatted_score
        }


def main():
    """Test the real DistilGPT2 RAG + ICL system."""
    analyzer = DistilGPT2RAGICL()
    
    test_queries = [
        "climate change is fake",
        "The economy is booming under this administration", 
        "Corporate greed continues to exploit working families"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"DISTILGPT2 RAG + ICL TEST: {query}")
        print('='*60)
        
        result = analyzer.analyze(query)
        
        print(f"Label: {result['answer']['tentative_label']}")
        print(f"Confidence: {result['answer']['confidence']:.3f}")
        print(f"Method: {result['score']['analysis_method']}")
        print(f"DistilGPT2 Used: {result['answer'].get('distilgpt2_inference', False)}")
        print(f"Few-shot Examples: {result['score']['few_shot_examples_used']}")
        print(f"Evidence: {result['answer']['evidence_spans']}")
        print(f"Rationale: {result['answer']['rationale'][:100]}...")
        print(f"Retrieved: {len(result['retrieved'])} documents")


if __name__ == "__main__":
    main()
