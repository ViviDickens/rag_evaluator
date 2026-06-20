"""Integration tests for RAG Evaluator"""

import pytest
from rag_evaluator.schemas import RAGEvaluationRequest, RAGEvaluationStatus
from rag_evaluator.evaluator import RAGEvaluator


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


@pytest.mark.asyncio
async def test_evaluate_single(rag_evaluator, sample_rag_request):
    """Test single evaluation."""
    response = await rag_evaluator.evaluate_single(sample_rag_request)

    assert response.evaluation_id
    assert response.metrics.faithfulness.score >= 0.0
    assert response.metrics.answer_relevancy.score >= 0.0
    assert response.overall_quality.status in [
        RAGEvaluationStatus.PASS,
        RAGEvaluationStatus.WARN,
        RAGEvaluationStatus.FAIL,
    ]


def test_faithfulness_calculation(rag_evaluator, sample_rag_request):
    """Test faithfulness metric."""
    score = rag_evaluator._calculate_faithfulness(
        sample_rag_request.generated_answer,
        sample_rag_request.retrieved_context,
        sample_rag_request.ground_truth_answer,
    )

    assert 0.0 <= score <= 1.0


def test_answer_relevancy_calculation(rag_evaluator, sample_rag_request):
    """Test answer relevancy metric."""
    score = rag_evaluator._calculate_answer_relevancy(
        sample_rag_request.question, sample_rag_request.generated_answer
    )

    assert 0.0 <= score <= 1.0


def test_context_precision_calculation(rag_evaluator, sample_rag_request):
    """Test context precision metric."""
    score = rag_evaluator._calculate_context_precision(
        sample_rag_request.question, sample_rag_request.retrieved_context
    )

    assert 0.0 <= score <= 1.0
