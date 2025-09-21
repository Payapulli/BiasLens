"""
FastAPI server for BiasLens bias analysis API.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from retriever import DocumentRetriever
from heuristics import BiasHeuristics
from icl_explainer import ICLExplainer


# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    q: str

class AnalysisResponse(BaseModel):
    answer: Dict[str, Any]
    retrieved: list
    score: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    message: str

# Initialize FastAPI app
app = FastAPI(
    title="BiasLens API",
    description="Political bias analysis using RAG and ICL",
    version="1.0.0"
)

# Global variables for components
retriever: Optional[DocumentRetriever] = None
heuristics: Optional[BiasHeuristics] = None
icl_explainer: Optional[ICLExplainer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global retriever, heuristics, icl_explainer
    
    try:
        print("Initializing BiasLens components...")
        
        # Initialize retriever
        print("Loading document retriever...")
        retriever = DocumentRetriever()
        
        # Initialize heuristics
        print("Loading bias heuristics...")
        heuristics = BiasHeuristics()
        
        # Initialize ICL explainer
        print("Loading ICL explainer...")
        icl_explainer = ICLExplainer()
        
        print("All components loaded successfully!")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        print("Some components may not be available")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic info."""
    return HealthResponse(
        status="ok",
        message="BiasLens API - Political bias analysis using RAG and ICL"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components_status = {
        "retriever": retriever is not None,
        "heuristics": heuristics is not None,
        "icl_explainer": icl_explainer is not None
    }
    
    all_ready = all(components_status.values())
    
    if all_ready:
        return HealthResponse(
            status="healthy",
            message="All components are ready"
        )
    else:
        missing = [k for k, v in components_status.items() if not v]
        return HealthResponse(
            status="degraded",
            message=f"Missing components: {', '.join(missing)}"
        )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_bias(request: AnalysisRequest):
    """
    Analyze text for political bias.
    
    Args:
        request: Analysis request with query text
        
    Returns:
        Analysis results with LLM output, retrieved documents, and heuristic scores
    """
    if not all([retriever, heuristics, icl_explainer]):
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Some components failed to load."
        )
    
    try:
        query = request.q.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Retrieve relevant documents
        print(f"Retrieving documents for query: {query[:50]}...")
        retrieved_docs = retriever.retrieve(query, top_k=5)
        
        # Get heuristic analysis
        print("Running heuristic analysis...")
        heuristic_result = heuristics.calculate_bias_score(query)
        
        # Get ICL analysis
        print("Running ICL analysis...")
        icl_result = icl_explainer.analyze(query, num_shots=2)
        
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
        
        # Format heuristic score
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
        
        return AnalysisResponse(
            answer=icl_result,
            retrieved=formatted_retrieved,
            score=formatted_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/retrieve")
async def retrieve_documents(q: str, top_k: int = 5):
    """
    Retrieve relevant documents for a query.
    
    Args:
        q: Query text
        top_k: Number of documents to retrieve
        
    Returns:
        List of retrieved documents
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not available")
    
    try:
        docs = retriever.retrieve(q, top_k)
        return {"query": q, "documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.get("/heuristics")
async def analyze_heuristics(q: str):
    """
    Get heuristic bias analysis for a query.
    
    Args:
        q: Query text
        
    Returns:
        Heuristic analysis results
    """
    if not heuristics:
        raise HTTPException(status_code=503, detail="Heuristics not available")
    
    try:
        result = heuristics.calculate_bias_score(q)
        indicators = heuristics.get_bias_indicators(q)
        result['indicators'] = indicators
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heuristic analysis failed: {str(e)}")


@app.get("/shots")
async def get_shots():
    """Get available few-shot examples."""
    if not icl_explainer:
        raise HTTPException(status_code=503, detail="ICL explainer not available")
    
    try:
        shots = icl_explainer.get_available_shots()
        return {"shots": shots, "count": len(shots)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get shots: {str(e)}")


def main():
    """Run the server."""
    import uvicorn
    
    print("Starting BiasLens API server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
