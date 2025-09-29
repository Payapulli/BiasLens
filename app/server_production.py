"""
Production-ready BiasLens server with RAG + ICL.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from retriever import DocumentRetriever
from heuristics import BiasHeuristics
from icl_explainer import ICLExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    version: str

# Initialize FastAPI app
app = FastAPI(
    title="BiasLens API",
    description="Political bias analysis using RAG and In-Context Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global variables for components
retriever: Optional[DocumentRetriever] = None
heuristics: Optional[BiasHeuristics] = None
icl_explainer: Optional[ICLExplainer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global retriever, heuristics, icl_explainer
    
    try:
        logger.info("Initializing BiasLens components...")
        
        # Initialize retriever
        logger.info("Loading document retriever...")
        retriever = DocumentRetriever()
        
        # Initialize heuristics
        logger.info("Loading bias heuristics...")
        heuristics = BiasHeuristics()
        
        # Initialize ICL explainer
        logger.info("Loading ICL explainer...")
        icl_explainer = ICLExplainer()
        
        logger.info("All components loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.error("Some components may not be available")


@app.get("/")
async def root():
    """Serve the frontend."""
    static_dir = Path(__file__).parent.parent / "static"
    return FileResponse(static_dir / "index.html")

@app.get("/api", response_model=HealthResponse)
async def api_info():
    """API info endpoint."""
    return HealthResponse(
        status="ok",
        message="BiasLens API - Political bias analysis using RAG and ICL",
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components_status = {
        "retriever": retriever is not None,
        "heuristics": heuristics is not None,
        "icl_explainer": icl_explainer is not None,
    }
    
    all_ready = all(components_status.values())
    
    if all_ready:
        return HealthResponse(
            status="healthy",
            message="All components are ready",
            version="1.0.0"
        )
    else:
        missing = [k for k, v in components_status.items() if not v]
        return HealthResponse(
            status="degraded",
            message=f"Missing components: {', '.join(missing)}",
            version="1.0.0"
        )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_bias(request: AnalysisRequest):
    """
    Analyze text for political bias using ICL.
    
    Args:
        request: Analysis request with query text
        
    Returns:
        Analysis results with ICL analysis
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
        
        if len(query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long. Maximum 1000 characters.")
        
        logger.info(f"Analyzing query: {query[:50]}...")
        
        # ICL Analysis
        icl_result = icl_explainer.analyze(query)
        
        # Heuristic validation
        heuristic_result = heuristics.calculate_bias_score(query)
        
        # Retrieve relevant documents for context
        retrieved_docs = retriever.retrieve(query, top_k=5)
        
        logger.info(f"Analysis complete. Label: {icl_result['tentative_label']}")
        
        return AnalysisResponse(
            answer=icl_result,
            retrieved=retrieved_docs,
            score=heuristic_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/retrieve")
async def retrieve_documents(q: str, top_k: int = 5):
    """Retrieve relevant documents for a query."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not available")
    
    try:
        if len(q) > 500:
            raise HTTPException(status_code=400, detail="Query too long. Maximum 500 characters.")
        
        docs = retriever.retrieve(q, min(top_k, 10))  # Limit to 10 max
        return {"query": q, "documents": docs}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.get("/heuristics")
async def analyze_heuristics(q: str):
    """Get heuristic bias analysis for a query."""
    if not heuristics:
        raise HTTPException(status_code=503, detail="Heuristics not available")
    
    try:
        if len(q) > 1000:
            raise HTTPException(status_code=400, detail="Query too long. Maximum 1000 characters.")
        
        result = heuristics.calculate_bias_score(q)
        indicators = heuristics.get_bias_indicators(q)
        result['indicators'] = indicators
        return result
    except Exception as e:
        logger.error(f"Heuristic analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Heuristic analysis failed: {str(e)}")


def main():
    """Run the server."""
    import uvicorn
    
    logger.info("Starting BiasLens API server...")
    logger.info("API will be available at: http://localhost:8000")
    logger.info("Frontend will be available at: http://localhost:8000")
    logger.info("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "server_production:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()