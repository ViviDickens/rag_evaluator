import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from rag_evaluator.app import app
from rag_evaluator.evaluator import RAGEvaluator
from rag_evaluator.schemas import MetricScore, RAGMetrics, RAGEvaluationRequest

def test_reference_required():
    with pytest.raises(ValidationError):
        RAGEvaluationRequest(question="Q", generated_answer="A", retrieved_context=["C"])

def test_empty_context_rejected():
    with pytest.raises(ValidationError):
        RAGEvaluationRequest(question="Q", generated_answer="A", retrieved_context=[], ground_truth_answer="A")

@pytest.mark.parametrize("score, status", [(0.9, "PASS"), (0.7, "WARN"), (0.5, "FAIL")])
def test_quality_gate(score, status):
    metrics = RAGMetrics(**{name: MetricScore(score=score) for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")})
    result = RAGEvaluator()._assess_overall_quality(metrics)
    assert result.status.value == status
    assert result.passed == (status == "PASS")

def test_http_contract():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert len(client.get("/detectors").json()["metrics"]) == 4
        assert client.post("/evaluate/rag", json={}).status_code == 422
