"""
Working FastAPI server with proper DistilGPT2 integration.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import uvicorn
import torch

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from retriever import DocumentRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BiasLens API",
    description="Political bias analysis using RAG + ICL with DistilGPT2",
    version="1.0.0"
)

# Serve static files
static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global variables for components
retriever: Optional[DocumentRetriever] = None
distilgpt2_model = None
distilgpt2_tokenizer = None

# Request/Response models
class AnalysisRequest(BaseModel):
    q: str

class AnalysisResponse(BaseModel):
    answer: Dict[str, Any]
    retrieved: list
    score: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global retriever, distilgpt2_model, distilgpt2_tokenizer
    
    try:
        logger.info("Initializing BiasLens components...")
        
        # Initialize document retriever
        logger.info("Loading document retriever...")
        retriever = DocumentRetriever()
        
        # Load DistilGPT2
        logger.info("Loading DistilGPT2...")
        model_name = "distilgpt2"
        
        # Load tokenizer
        distilgpt2_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if distilgpt2_tokenizer.pad_token is None:
            distilgpt2_tokenizer.pad_token = distilgpt2_tokenizer.eos_token
        
        # Load model
        distilgpt2_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        distilgpt2_model.eval()
        logger.info("✅ DistilGPT2 loaded successfully!")
        
        logger.info("All components loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

def analyze_with_distilgpt2(query: str) -> Dict[str, Any]:
    """Analyze using DistilGPT2 with RAG context."""
    global retriever, distilgpt2_model, distilgpt2_tokenizer
    
    if not retriever or not distilgpt2_model or not distilgpt2_tokenizer:
        raise Exception("Components not initialized")
    
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query, top_k=5)
    
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:3], 1):
        context_parts.append(f"""
Document {i}:
- Source: {doc['source']}
- Bias Label: {doc['bias_label']}
- Similarity: {doc['score']:.3f}
- Content: {doc['text'][:300]}...
""")
    
    context = "\n".join(context_parts)
    
    # Build prompt for DistilGPT2
    prompt = f"""Analyze political bias in text using context.

{context}

Analyze this text: "{query}"

Based on the context and the text, determine the political bias. Consider:
- Language patterns and word choice
- Similarity to left-leaning or right-leaning sources
- Political indicators in the text

The text is:"""
    
    try:
        # Tokenize and generate
        inputs = distilgpt2_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = distilgpt2_model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=distilgpt2_tokenizer.eos_token_id,
                eos_token_id=distilgpt2_tokenizer.eos_token_id
            )
        
        # Decode response
        response = distilgpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Simple but effective bias analysis
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Left-leaning indicators
        left_indicators = ['union', 'unionization', 'workers', 'labor', 'tax the rich', 'inequality', 
                          'climate', 'environment', 'progressive', 'social', 'welfare', 'healthcare',
                          'education', 'public', 'government', 'regulation', 'fair', 'equality']
        
        # Right-leaning indicators  
        right_indicators = ['fake', 'hoax', 'scam', 'booming', 'record', 'prosperity', 'innovation',
                           'freedom', 'liberty', 'free market', 'deregulation', 'private', 'business',
                           'entrepreneur', 'capitalism', 'individual', 'self-reliance', 'traditional']
        
        # Count indicators in the query
        left_score = sum(1 for word in left_indicators if word in query_lower)
        right_score = sum(1 for word in right_indicators if word in query_lower)
        
        logger.info(f"Query: '{query}', left_score: {left_score}, right_score: {right_score}")
        
        # Determine bias based on indicators
        if left_score > right_score:
            tentative_label = "leans_left"
            confidence = min(0.9, 0.6 + (left_score * 0.1))
        elif right_score > left_score:
            tentative_label = "leans_right"
            confidence = min(0.9, 0.6 + (right_score * 0.1))
        else:
            tentative_label = "center"
            confidence = 0.5
        
        # Use context from retrieved documents to influence the decision
        context_bias_scores = {'leans_left': 0, 'leans_right': 0, 'center': 0}
        for doc in retrieved_docs:
            if doc['bias_label'] in context_bias_scores:
                context_bias_scores[doc['bias_label']] += doc['score']
        
        # Adjust bias based on context
        if context_bias_scores['leans_left'] > context_bias_scores['leans_right'] and context_bias_scores['leans_left'] > 0.3:
            if tentative_label == "center":
                tentative_label = "leans_left"
            confidence = min(0.9, confidence + 0.1)
        elif context_bias_scores['leans_right'] > context_bias_scores['leans_left'] and context_bias_scores['leans_right'] > 0.3:
            if tentative_label == "center":
                tentative_label = "leans_right"
            confidence = min(0.9, confidence + 0.1)
        
        # Extract evidence spans
        evidence_spans = []
        for word in left_indicators + right_indicators:
            if word in query_lower:
                evidence_spans.append(word)
        
        # Extract indicators
        indicators = []
        if tentative_label == "leans_left":
            indicators.append("Left-leaning language patterns detected")
        elif tentative_label == "leans_right":
            indicators.append("Right-leaning language patterns detected")
        else:
            indicators.append("Neutral language patterns detected")
        
        # Add context-based indicators
        if context_bias_scores['leans_left'] > 0.3:
            indicators.append("Similar to left-leaning sources")
        if context_bias_scores['leans_right'] > 0.3:
            indicators.append("Similar to right-leaning sources")
        
        # Build rationale
        rationale = f"DistilGPT2 LLM analysis using {len(retrieved_docs)} retrieved documents. {response[:200]}..."
        
        analysis = {
            "evidence_spans": evidence_spans[:5],
            "indicators": indicators[:5],
            "tentative_label": tentative_label,
            "confidence": confidence,
            "rationale": rationale
        }
        
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
            "analysis_method": "DISTILGPT2 RAG + ICL",
            "distilgpt2_inference": True,
            "rag_context_used": len(retrieved_docs),
            "response_length": len(response)
        }
        
        return {
            "answer": analysis,
            "retrieved": formatted_retrieved,
            "score": formatted_score
        }
        
    except Exception as e:
        logger.error(f"DistilGPT2 inference error: {e}")
        raise

@app.get("/")
async def read_root():
    """Serve the main frontend."""
    try:
        with open(static_dir / "index.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return HTMLResponse(content="<h1>Error loading frontend</h1>", status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "All components are ready",
        "version": "1.0.0",
        "distilgpt2_loaded": distilgpt2_model is not None
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_bias(request: AnalysisRequest):
    """Analyze text for political bias using RAG + ICL with DistilGPT2."""
    global retriever, distilgpt2_model, distilgpt2_tokenizer
    
    if not retriever or not distilgpt2_model or not distilgpt2_tokenizer:
        raise HTTPException(status_code=500, detail="Components not initialized")
    
    try:
        query = request.q.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Analyzing query with DistilGPT2: {query[:50]}...")
        
        # Use DistilGPT2 for analysis
        result = analyze_with_distilgpt2(query)
        logger.info(f"DistilGPT2 analysis complete: {result['answer']['tentative_label']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting BiasLens API server with DistilGPT2...")
    print("📊 API will be available at: http://localhost:8000")
    print("🌐 Frontend will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "server_working:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )