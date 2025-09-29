# BiasLens

A local, CPU-friendly tool for analyzing political and news bias in text using Retrieval-Augmented Generation (RAG) and In-Context Learning (ICL).

## Overview

BiasLens combines:
- **FAISS vector search** for retrieving relevant article snippets
- **DistilGPT2 LLM inference** for intelligent bias analysis
- **In-context learning** with few-shot examples for structured bias analysis
- **Small transformer models** optimized for CPU inference

## Quick Start

### Local Development

1. **Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare Data:**
   ```bash
   # Index the sample articles
   python scripts/index_docs.py
   ```

3. **Run the Development Server:**
   ```bash
   python app/server.py
   ```

4. **Access the Application:**
   - Frontend: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Production Deployment

For production deployment, see [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

**Quick deployment:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Docker deployment:**
```bash
docker-compose up -d
```

### Test Analysis

```bash
# Quick test
python -m uvicorn app.server:app --reload

# Or use the API directly
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"q": "The economy is booming under this administration"}'
```

## API Usage

### POST /analyze

Analyze text for political bias.

**Request:**
```json
{
  "q": "Your text to analyze here"
}
```

**Response:**
```json
{
  "answer": {
    "evidence_spans": ["specific phrases that indicate bias"],
    "indicators": ["bias indicators found"],
    "tentative_label": "leans_left|center|leans_right",
    "confidence": 0.85,
    "rationale": "Explanation of the analysis"
  },
  "retrieved": [
    {
      "text": "retrieved article snippet",
      "score": 0.92,
      "source": "article source"
    }
  ],
  "score": {
    "heuristic_score": 0.3,
    "sentiment": "positive",
    "keywords": ["economy", "booming"]
  }
}
```

## Project Structure

```
biaslens/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/
│   └── articles.jsonl        # Sample article dataset
├── scripts/
│   └── index_docs.py         # Build FAISS index from articles
├── app/
│   ├── retriever.py          # FAISS retrieval logic
│   ├── server.py             # FastAPI application with DistilGPT2
│   └── index/                # FAISS index files
└── examples/
    └── shots.json            # Few-shot examples for ICL
```

## Limitations

- **CPU-only**: Designed for local inference, not production scale
- **Small models**: Uses lightweight models (distilgpt2) for demo purposes
- **Limited dataset**: Sample data only - replace with your own articles
- **Heuristic-based**: Bias detection relies on simple rules and patterns
- **No training**: Uses pre-trained models without fine-tuning

## Customization

1. **Add your own articles**: Replace `data/articles.jsonl` with your dataset
2. **Modify analysis**: Edit `app/server.py` to adjust DistilGPT2 prompting
3. **Update examples**: Modify `examples/shots.json` for different ICL patterns
4. **Change models**: Swap models in the code (requires updating requirements.txt)

## Requirements

- Python 3.8+
- 4GB+ RAM recommended
- No GPU required (CPU-only)
