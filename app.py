"""
FastAPI application for RAG evaluation using RAGAS metrics.
Complements ai-qa-decision-engine with RAG quality validation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List

from schemas import RAGEvaluationRequest, RAGEvaluationResponse, BatchEvaluationRequest
from evaluator import RAGEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Evaluator Mini",
    description="RAGAS-based evaluation for Retrieval-Augmented Generation pipelines",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize evaluator
evaluator = RAGEvaluator()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "RAG Evaluator Mini",
        "version": "0.1.0"
    }


@app.get("/detectors")
async def list_detectors():
    """List available RAGAS metrics."""
    return {
        "metrics": [
            {
                "name": "faithfulness",
                "description": "How faithfully does the answer adhere to retrieved context?",
                "threshold": 0.85
            },
            {
                "name": "answer_relevancy",
                "description": "How relevant is the answer to the question?",
                "threshold": 0.80
            },
            {
                "name": "context_precision",
                "description": "Is the retrieved context precise and useful?",
                "threshold": 0.90
            },
            {
                "name": "context_recall",
                "description": "Is all necessary context retrieved?",
                "threshold": 0.75
            }
        ]
    }


@app.post("/evaluate/rag", response_model=RAGEvaluationResponse)
async def evaluate_rag(request: RAGEvaluationRequest):
    """
    Evaluate a RAG pipeline output using RAGAS metrics.
    
    Validates:
    - Faithfulness: Does the answer stick to context?
    - Answer Relevancy: Is the answer relevant to the question?
    - Context Precision: Is retrieved context actually useful?
    - Context Recall: Is all needed context retrieved?
    """
    try:
        result = await evaluator.evaluate_single(request)
        logger.info(f"Evaluation successful: {result.evaluation_id}")
        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/batch")
async def evaluate_batch(request: BatchEvaluationRequest):
    """
    Batch evaluate multiple RAG outputs with statistical analysis.
    
    Returns:
    - Per-sample scores
    - Aggregate statistics (mean, std_dev, confidence_interval)
    - Distribution analysis
    - Statistical significance
    """
    try:
        results = await evaluator.evaluate_batch(request.evaluations)
        logger.info(f"Batch evaluation successful: {len(results['evaluations'])} items")
        return results
    except Exception as e:
        logger.error(f"Batch evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/with-deepeval")
async def evaluate_with_deepeval(request: RAGEvaluationRequest):
    """
    Cross-validate RAGAS metrics with DeepEval framework.
    
    Useful for:
    - Comparing metric frameworks
    - Validating against multiple evaluation approaches
    - Building confidence in scoring
    """
    try:
        result = await evaluator.evaluate_with_deepeval(request)
        logger.info(f"Cross-validation successful: {result.evaluation_id}")
        return result
    except Exception as e:
        logger.error(f"Cross-validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docs", include_in_schema=False)
async def get_docs():
    """Swagger UI documentation."""
    return {"message": "Visit /docs for interactive API documentation"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
