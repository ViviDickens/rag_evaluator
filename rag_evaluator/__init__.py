"""
RAG Evaluation Orchestrator: Coordinates RAGAS, DeepEval, and LangSmith.
"""

__version__ = "0.1.0"
__author__ = "Viviana Pérez"

from .app import app
from .evaluator import RAGEvaluator
from .observability import LangSmithObserver
from .schemas import (
    RAGEvaluationRequest,
    RAGEvaluationResponse,
    RAGMetrics,
    OverallQuality,
)

__all__ = [
    "app",
    "RAGEvaluator",
    "LangSmithObserver",
    "RAGEvaluationRequest",
    "RAGEvaluationResponse",
    "RAGMetrics",
    "OverallQuality",
]
