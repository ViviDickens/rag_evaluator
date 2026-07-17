"""Integration tests for RAG Evaluator"""

import os
import pytest
from rag_evaluator.schemas import RAGEvaluationRequest, RAGEvaluationStatus
from rag_evaluator.evaluator import RAGEvaluator

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — evaluator calls the real OpenAI API",
)


@pytest.fixture
def rag_evaluator():
    return RAGEvaluator()


@pytest.fixture
def sample_rag_request():
    return RAGEvaluationRequest(
        question="What is RAG?",
        retrieved_context=[
            "RAG is a technique that combines retrieval and generation.",
            "It improves factuality by grounding responses in context.",
        ],
        generated_answer="RAG augments LLM prompts with retrieved context to improve accuracy.",
        ground_truth_answer="RAG combines retrieval with generation for better answers.",
    )


@requires_openai_key
@pytest.mark.asyncio
async def test_evaluate_single(rag_evaluator, sample_rag_request):
    """
    End-to-end test: hits real RAGAS (OpenAI) + DeepEval, not a mock.
    Skipped automatically if OPENAI_API_KEY isn't set, so CI doesn't
    fail just because the secret isn't configured on that runner.
    """
    response = await rag_evaluator.evaluate_single(sample_rag_request)

    assert response.evaluation_id
    assert 0.0 <= response.metrics.faithfulness.score <= 1.0
    assert 0.0 <= response.metrics.answer_relevancy.score <= 1.0
    assert 0.0 <= response.metrics.context_precision.score <= 1.0
    assert 0.0 <= response.metrics.context_recall.score <= 1.0
    assert response.overall_quality.status in [
        RAGEvaluationStatus.PASS,
        RAGEvaluationStatus.WARN,
        RAGEvaluationStatus.FAIL,
    ]
    assert len(response.recommendations) >= 1


def test_evaluator_thresholds_configured(rag_evaluator):
    """No API call needed — just checks the evaluator wired up its threshold config."""
    assert rag_evaluator.thresholds["faithfulness"] == 0.85
    assert rag_evaluator.thresholds["answer_relevancy"] == 0.80
    assert rag_evaluator.thresholds["context_precision"] == 0.90
    assert rag_evaluator.thresholds["context_recall"] == 0.75


def test_recommendations_flag_low_scores(rag_evaluator, sample_rag_request):
    """No API call — feeds fabricated low scores straight into the recommendation logic."""
    from rag_evaluator.schemas import RAGMetrics, MetricScore

    low_metrics = RAGMetrics(
        faithfulness=MetricScore(score=0.3, confidence=0.9),
        answer_relevancy=MetricScore(score=0.4, confidence=0.9),
        context_precision=MetricScore(score=0.5, confidence=0.9),
        context_recall=MetricScore(score=0.2, confidence=0.9),
    )

    recommendations = rag_evaluator._generate_recommendations(low_metrics, sample_rag_request)

    assert any("Faithfulness" in r for r in recommendations)
    assert any("relevancy" in r for r in recommendations)
    assert any("precision" in r for r in recommendations)
    assert any("recall" in r for r in recommendations)
