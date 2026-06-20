"""
RAG Evaluation Orchestrator: FastAPI application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from .schemas import RAGEvaluationRequest, RAGEvaluationResponse
from .evaluator import RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Evaluation Orchestrator",
    description="Coordinates RAGAS, DeepEval, and LangSmith",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

evaluator = RAGEvaluator()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAG Evaluation Orchestrator"}


@app.get("/detectors")
async def list_detectors():
    """List available RAGAS metrics."""
    return {
        "metrics": [
            {
                "name": "faithfulness",
                "description": "Does answer adhere to context?",
                "threshold": 0.85,
            },
            {
                "name": "answer_relevancy",
                "description": "Is answer relevant to question?",
                "threshold": 0.80,
            },
            {
                "name": "context_precision",
                "description": "Is retrieved context useful?",
                "threshold": 0.90,
            },
            {
                "name": "context_recall",
                "description": "Is all needed context retrieved?",
                "threshold": 0.75,
            },
        ]
    }


@app.post("/evaluate/rag", response_model=RAGEvaluationResponse)
async def evaluate_rag(request: RAGEvaluationRequest):
    """Evaluate a RAG pipeline output."""
    try:
        result = await evaluator.evaluate_single(request)
        logger.info(f"Evaluation successful: {result.evaluation_id}")
        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
