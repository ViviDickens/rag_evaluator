"""
Integration tests for RAGAS + DeepEval + LangSmith RAG Evaluator.
Demonstrates three-framework evaluation pipeline.
"""

import pytest
import asyncio
from typing import List

from schemas import RAGEvaluationRequest, RAGEvaluationStatus
from evaluator import RAGEvaluator
from observability import LangSmithObserver


# Test fixtures
@pytest.fixture
def rag_evaluator():
    """Initialize RAG evaluator."""
    return RAGEvaluator()


@pytest.fixture
def langsmith_observer():
    """Initialize LangSmith observer (disabled for tests without API key)."""
    return LangSmithObserver(project_name="rag_evaluator_tests", api_key=None)


@pytest.fixture
def sample_rag_request():
    """Sample RAG evaluation request."""
    return RAGEvaluationRequest(
        question="What is Retrieval-Augmented Generation?",
        retrieved_context=[
            "RAG is a technique that combines retrieval and generation. It first retrieves relevant documents "
            "from a knowledge base, then uses them to augment the prompt for an LLM.",
            "RAG improves factuality and reduces hallucinations by grounding responses in retrieved context. "
            "It's particularly useful for domain-specific QA systems."
        ],
        generated_answer="RAG (Retrieval-Augmented Generation) is a method that retrieves relevant documents "
                         "from a knowledge base and uses them to augment LLM prompts, improving factuality and "
                         "reducing hallucinations.",
        ground_truth_answer="RAG augments large language models with retrieved context to improve answer accuracy "
                           "and reduce hallucinations in domain-specific question answering."
    )


# RAGAS Tests
class TestRASGMetrics:
    """Test RAGAS metric calculation."""
    
    @pytest.mark.asyncio
    async def test_evaluate_single_with_ragas(self, rag_evaluator, sample_rag_request):
        """Test single evaluation with RAGAS metrics."""
        response = await rag_evaluator.evaluate_single(sample_rag_request)
        
        # Verify all RAGAS metrics are present
        assert response.metrics.faithfulness.score >= 0.0
        assert response.metrics.answer_relevancy.score >= 0.0
        assert response.metrics.context_precision.score >= 0.0
        assert response.metrics.context_recall.score >= 0.0
        
        # Verify confidence scores
        assert 0.0 <= response.metrics.faithfulness.confidence <= 1.0
        assert 0.0 <= response.metrics.answer_relevancy.confidence <= 1.0
        
        # Verify evaluation ID
        assert response.evaluation_id.startswith("eval-")
        
        # Verify timestamp
        assert response.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_overall_quality_assessment(self, rag_evaluator, sample_rag_request):
        """Test overall quality assessment logic."""
        response = await rag_evaluator.evaluate_single(sample_rag_request)
        
        # Verify overall quality
        assert 0.0 <= response.overall_quality.score <= 1.0
        assert response.overall_quality.status in [
            RAGEvaluationStatus.PASS,
            RAGEvaluationStatus.WARN,
            RAGEvaluationStatus.FAIL
        ]
        assert response.overall_quality.threshold > 0.0
    
    def test_faithfulness_calculation(self, rag_evaluator, sample_rag_request):
        """Test faithfulness metric calculation."""
        score = rag_evaluator._calculate_faithfulness(
            sample_rag_request.generated_answer,
            sample_rag_request.retrieved_context,
            sample_rag_request.ground_truth_answer
        )
        
        assert 0.0 <= score <= 1.0
        # With good context alignment, should be high
        assert score > 0.5
    
    def test_answer_relevancy_calculation(self, rag_evaluator, sample_rag_request):
        """Test answer relevancy calculation."""
        score = rag_evaluator._calculate_answer_relevancy(
            sample_rag_request.question,
            sample_rag_request.generated_answer
        )
        
        assert 0.0 <= score <= 1.0
        # Answer contains key question terms, should be high
        assert score > 0.5
    
    def test_context_precision_calculation(self, rag_evaluator, sample_rag_request):
        """Test context precision calculation."""
        score = rag_evaluator._calculate_context_precision(
            sample_rag_request.question,
            sample_rag_request.retrieved_context
        )
        
        assert 0.0 <= score <= 1.0
        # Both context docs are relevant to question
        assert score > 0.7


# DeepEval Cross-Validation Tests
class TestDeepEvalIntegration:
    """Test DeepEval integration with RAGAS."""
    
    @pytest.mark.asyncio
    async def test_deepeval_cross_validation(self, rag_evaluator, sample_rag_request):
        """Test DeepEval cross-validation with RAGAS."""
        response = await rag_evaluator.evaluate_single(sample_rag_request)
        
        # Check that cross-validation results are present
        assert "cross_validation" in response.metadata
        cross_val = response.metadata["cross_validation"]
        
        # Verify DeepEval checks
        assert "faithfulness" in cross_val
        assert "answer_relevancy" in cross_val
        assert "bias" in cross_val
        
        # Verify agreement scores
        assert all(0.0 <= result.agreement <= 1.0 
                   for result in cross_val.values())
    
    def test_deepeval_hallucination_check(self, rag_evaluator):
        """Test DeepEval hallucination detection."""
        # Answer with suspicious phrases
        answer_with_hallucination = "Obviously, it is undoubtedly clear that RAG is amazing."
        score = rag_evaluator._deepeval_check_hallucination(answer_with_hallucination)
        
        # Should be lower than normal answer
        assert 0.0 <= score <= 1.0
        assert score < 0.9
        
        # Normal answer
        normal_answer = "RAG is a technique that combines retrieval and generation."
        score = rag_evaluator._deepeval_check_hallucination(normal_answer)
        assert score > 0.9
    
    def test_deepeval_bias_check(self, rag_evaluator):
        """Test DeepEval bias detection."""
        answer = "RAG is an important technique in AI."
        score = rag_evaluator._deepeval_check_bias(answer)
        
        # Should be high (minimal bias)
        assert 0.9 <= score <= 1.0


# LangSmith Observability Tests
class TestLangSmithObservability:
    """Test LangSmith observability integration."""
    
    def test_langsmith_observer_initialization(self, langsmith_observer):
        """Test LangSmith observer initialization."""
        assert langsmith_observer.project_name == "rag_evaluator_tests"
        assert langsmith_observer.enabled is False  # No API key in tests
    
    def test_langsmith_metric_logging(self, langsmith_observer):
        """Test metric logging to LangSmith."""
        # Should not raise exception
        langsmith_observer.log_metric("faithfulness_score", 0.92)
        langsmith_observer.log_metric("answer_relevancy", 0.87, 
                                     {"context_size": 2})
    
    def test_langsmith_chain_logging(self, langsmith_observer):
        """Test chain execution logging."""
        langsmith_observer.log_chain(
            chain_name="retrieval_step",
            inputs={"query": "What is RAG?"},
            outputs={"retrieved_docs": ["doc1", "doc2"]},
            metadata={"retrieval_model": "bm25"}
        )
    
    def test_langsmith_llm_call_logging(self, langsmith_observer):
        """Test LLM call logging."""
        langsmith_observer.log_llm_call(
            model="gpt-3.5-turbo",
            prompt="Answer this question: What is RAG?",
            response="RAG is a technique...",
            tokens_used=150,
            latency_ms=1200.5
        )
    
    def test_langsmith_evaluation_result_logging(self, langsmith_observer):
        """Test evaluation result logging."""
        langsmith_observer.log_evaluation_result(
            evaluation_id="eval-123",
            metrics={
                "faithfulness": 0.92,
                "answer_relevancy": 0.87
            },
            recommendations=["Improve context precision"],
            status="PASS"
        )
    
    def test_langsmith_error_logging(self, langsmith_observer):
        """Test error context logging."""
        langsmith_observer.log_error_context(
            error_id="err-001",
            error_message="Faithfulness check failed",
            context={
                "question": "What is RAG?",
                "answer_length": 150
            },
            trace_id="trace-abc123"
        )


# Integration Tests
class TestThreeFrameworkIntegration:
    """Test RAGAS + DeepEval + LangSmith integration."""
    
    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(self, rag_evaluator, langsmith_observer, sample_rag_request):
        """Test complete evaluation pipeline with all three frameworks."""
        # Step 1: RAGAS evaluation
        response = await rag_evaluator.evaluate_single(sample_rag_request)
        
        # Step 2: Log metrics with LangSmith
        langsmith_observer.log_metric("faithfulness", response.metrics.faithfulness.score)
        langsmith_observer.log_metric("answer_relevancy", response.metrics.answer_relevancy.score)
        langsmith_observer.log_metric("context_precision", response.metrics.context_precision.score)
        langsmith_observer.log_metric("context_recall", response.metrics.context_recall.score)
        
        # Step 3: Log evaluation result
        langsmith_observer.log_evaluation_result(
            evaluation_id=response.evaluation_id,
            metrics={
                "faithfulness": response.metrics.faithfulness.score,
                "answer_relevancy": response.metrics.answer_relevancy.score,
                "context_precision": response.metrics.context_precision.score,
                "context_recall": response.metrics.context_recall.score,
                "overall": response.overall_quality.score
            },
            recommendations=response.recommendations,
            status=response.overall_quality.status.value
        )
        
        # Verify all steps completed
        assert response.evaluation_id
        assert response.metrics
        assert response.overall_quality
    
    @pytest.mark.asyncio
    async def test_batch_evaluation_with_observability(self, rag_evaluator, langsmith_observer):
        """Test batch evaluation with LangSmith batch tracing."""
        requests = [
            RAGEvaluationRequest(
                question=f"Question {i}",
                retrieved_context=[f"Context {i}"],
                generated_answer=f"Answer to question {i}",
                ground_truth_answer=f"Ground truth {i}"
            )
            for i in range(3)
        ]
        
        # Evaluate batch
        batch_response = await rag_evaluator.evaluate_batch(requests)
        
        # Log batch metrics
        for eval_result in batch_response.evaluations:
            langsmith_observer.log_metric(
                f"eval_{eval_result.evaluation_id}_overall",
                eval_result.overall_quality.score
            )
        
        # Verify batch results
        assert batch_response.batch_id.startswith("batch-")
        assert len(batch_response.evaluations) == 3
        assert batch_response.overall_pass_rate >= 0.0
        assert batch_response.statistics is not None


# Edge Case Tests
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_evaluation_with_no_context(self, rag_evaluator):
        """Test evaluation with empty retrieved context."""
        request = RAGEvaluationRequest(
            question="What is RAG?",
            retrieved_context=[],
            generated_answer="RAG is a technique...",
            ground_truth_answer="RAG augments LLMs with retrieval..."
        )
        
        response = await rag_evaluator.evaluate_single(request)
        
        # Should still produce evaluation
        assert response.evaluation_id
        assert response.metrics.context_precision.score == 0.0
    
    @pytest.mark.asyncio
    async def test_evaluation_with_no_ground_truth(self, rag_evaluator):
        """Test evaluation without ground truth."""
        request = RAGEvaluationRequest(
            question="What is RAG?",
            retrieved_context=["Context about RAG"],
            generated_answer="RAG is a technique...",
            ground_truth_answer=None  # No ground truth
        )
        
        response = await rag_evaluator.evaluate_single(request)
        
        # Should still produce valid evaluation
        assert response.evaluation_id
        assert 0.0 <= response.overall_quality.score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
