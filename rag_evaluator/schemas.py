"""
Pydantic schemas para RAG evaluation requests y responses.
"""

from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RAGEvaluationStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class MetricScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class RAGMetrics(BaseModel):
    faithfulness: MetricScore
    answer_relevancy: MetricScore
    context_precision: MetricScore
    context_recall: MetricScore


class OverallQuality(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    status: RAGEvaluationStatus
    threshold: float
    passed: bool


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RAGEvaluationRequest(BaseModel):
    question: NonBlank
    retrieved_context: List[NonBlank] = Field(..., min_length=1)
    generated_answer: NonBlank
    ground_truth_answer: NonBlank
    metadata: Optional[Dict[str, Any]] = None


class RAGEvaluationResponse(BaseModel):
    evaluation_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: RAGMetrics
    overall_quality: OverallQuality
    recommendations: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
