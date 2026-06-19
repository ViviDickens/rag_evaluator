"""
Pydantic schemas for RAG evaluation requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RAGEvaluationStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class MetricScore(BaseModel):
    """Individual metric score with confidence."""
    score: float = Field(..., ge=0.0, le=1.0, description="Metric score 0-1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the score")
    
    class Config:
        example = {"score": 0.92, "confidence": 0.88}


class RAGMetrics(BaseModel):
    """RAGAS evaluation metrics."""
    faithfulness: MetricScore = Field(..., description="Does answer stick to context?")
    answer_relevancy: MetricScore = Field(..., description="Is answer relevant to question?")
    context_precision: MetricScore = Field(..., description="Is retrieved context useful?")
    context_recall: MetricScore = Field(..., description="Is all needed context retrieved?")
    
    class Config:
        example = {
            "faithfulness": {"score": 0.92, "confidence": 0.88},
            "answer_relevancy": {"score": 0.87, "confidence": 0.85},
            "context_precision": {"score": 0.95, "confidence": 0.90},
            "context_recall": {"score": 0.80, "confidence": 0.78}
        }


class OverallQuality(BaseModel):
    """Overall RAG output quality assessment."""
    score: float = Field(..., ge=0.0, le=1.0, description="Aggregate quality score")
    status: RAGEvaluationStatus = Field(..., description="Pass/Warn/Fail status")
    threshold: float = Field(..., description="Quality threshold used")
    passed: bool = Field(..., description="Whether quality meets threshold")
    
    class Config:
        example = {
            "score": 0.885,
            "status": "PASS",
            "threshold": 0.75,
            "passed": True
        }


class RAGEvaluationRequest(BaseModel):
    """Request to evaluate a RAG pipeline output."""
    question: str = Field(..., description="The user question")
    retrieved_context: List[str] = Field(..., description="Retrieved document chunks")
    generated_answer: str = Field(..., description="LLM-generated answer")
    ground_truth_answer: Optional[str] = Field(None, description="Reference answer for validation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    class Config:
        example = {
            "question": "What is RAG?",
            "retrieved_context": [
                "RAG is a technique that combines...",
                "It improves LLM accuracy by retrieving..."
            ],
            "generated_answer": "RAG is a method where an LLM retrieves relevant documents...",
            "ground_truth_answer": "RAG augments LLM outputs with retrieved context..."
        }


class RAGEvaluationResponse(BaseModel):
    """Response with RAG evaluation results."""
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: RAGMetrics = Field(..., description="RAGAS metric scores")
    overall_quality: OverallQuality = Field(..., description="Overall quality assessment")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    metadata: Optional[Dict[str, Any]] = Field(None)
    
    class Config:
        example = {
            "evaluation_id": "eval-abc123",
            "timestamp": "2026-06-19T01:51:00Z",
            "metrics": {
                "faithfulness": {"score": 0.92, "confidence": 0.88},
                "answer_relevancy": {"score": 0.87, "confidence": 0.85},
                "context_precision": {"score": 0.95, "confidence": 0.90},
                "context_recall": {"score": 0.80, "confidence": 0.78}
            },
            "overall_quality": {
                "score": 0.885,
                "status": "PASS",
                "threshold": 0.75,
                "passed": True
            },
            "recommendations": [
                "Context recall slightly below optimal—consider expanding retrieval window"
            ]
        }


class BatchEvaluationRequest(BaseModel):
    """Request for batch RAG evaluations."""
    evaluations: List[RAGEvaluationRequest] = Field(..., description="List of evaluations to perform")
    calculate_confidence_intervals: bool = Field(default=True, description="Calculate statistical CIs")
    
    class Config:
        example = {
            "evaluations": [
                {
                    "question": "Question 1",
                    "retrieved_context": ["Context 1"],
                    "generated_answer": "Answer 1",
                    "ground_truth_answer": "Ground truth 1"
                }
            ],
            "calculate_confidence_intervals": True
        }


class BatchStatistics(BaseModel):
    """Statistical analysis for batch evaluations."""
    metric_name: str
    mean: float
    std_dev: float
    min: float
    max: float
    confidence_interval_95: tuple = Field(..., description="95% CI as (lower, upper)")
    sample_size: int


class BatchEvaluationResponse(BaseModel):
    """Response with batch evaluation results and statistics."""
    batch_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    evaluations: List[RAGEvaluationResponse]
    statistics: Optional[Dict[str, BatchStatistics]] = None
    overall_pass_rate: float = Field(..., ge=0.0, le=1.0)
    summary: str
