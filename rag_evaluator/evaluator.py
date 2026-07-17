"""
RAG Evaluation Orchestrator: Coordinates RAGAS + DeepEval + LangSmith.

This version calls the real libraries. Evaluator model is OpenAI
(gpt-4o-mini + text-embedding-3-small) — no dependency clashes with
LangChain/RAGAS like the Gemini combo had, at the cost of paying per
call. Swap ChatOpenAI for ChatGroq in _build_ragas_llm() if you want
the free-tier version instead (see ragas-lab for that variant).
"""

import os
import uuid
import logging
from datetime import datetime
from typing import List, Optional

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from deepeval.metrics import HallucinationMetric, BiasMetric
from deepeval.test_case import LLMTestCase

from .schemas import (
    RAGEvaluationRequest,
    RAGEvaluationResponse,
    RAGMetrics,
    MetricScore,
    OverallQuality,
    RAGEvaluationStatus,
)
from .observability import LangSmithObserver

logger = logging.getLogger(__name__)


def _build_ragas_llm() -> LangchainLLMWrapper:
    return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))


class RAGEvaluator:
    """
    Orchestrates RAGAS, DeepEval, and LangSmith for RAG evaluation.
    """

    def __init__(self):
        self.thresholds = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.90,
            "context_recall": 0.75,
        }
        # Built lazily on first use (see llm/embeddings properties below) so
        # constructing a RAGEvaluator doesn't require OPENAI_API_KEY unless
        # you actually run an evaluation.
        self._llm: Optional[LangchainLLMWrapper] = None
        self._embeddings: Optional[LangchainEmbeddingsWrapper] = None
        self._observer = LangSmithObserver(api_key=os.environ.get("LANGSMITH_API_KEY"))
        logger.info("RAGEvaluator initialized (RAGAS + DeepEval + LangSmith)")

    @property
    def llm(self) -> LangchainLLMWrapper:
        if self._llm is None:
            self._llm = _build_ragas_llm()
        return self._llm

    @property
    def embeddings(self) -> LangchainEmbeddingsWrapper:
        if self._embeddings is None:
            self._embeddings = _build_ragas_embeddings()
        return self._embeddings

    async def evaluate_single(self, request: RAGEvaluationRequest) -> RAGEvaluationResponse:
        """Evaluate a single RAG output."""
        evaluation_id = f"eval-{uuid.uuid4().hex[:8]}"

        metrics = self._calculate_ragas_metrics(request)
        self._cross_validate_with_deepeval(request, metrics)

        overall_quality = self._assess_overall_quality(metrics)
        recommendations = self._generate_recommendations(metrics, request)

        self._observer.log_evaluation_result(
            evaluation_id=evaluation_id,
            metrics={
                "faithfulness": metrics.faithfulness.score,
                "answer_relevancy": metrics.answer_relevancy.score,
                "context_precision": metrics.context_precision.score,
                "context_recall": metrics.context_recall.score,
            },
            recommendations=recommendations,
            status=overall_quality.status.value,
        )

        return RAGEvaluationResponse(
            evaluation_id=evaluation_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            overall_quality=overall_quality,
            recommendations=recommendations,
        )

    def _calculate_ragas_metrics(self, request: RAGEvaluationRequest) -> RAGMetrics:
        """Run the actual RAGAS metrics suite via ragas.evaluate()."""
        ground_truth = request.ground_truth_answer or ""

        dataset = Dataset.from_dict(
            {
                "question": [request.question],
                "answer": [request.generated_answer],
                "contexts": [request.retrieved_context],
                "ground_truth": [ground_truth],
            }
        )

        result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False,
        )

        scores = result.to_pandas().iloc[0]

        return RAGMetrics(
            faithfulness=MetricScore(score=float(scores["faithfulness"]), confidence=0.9),
            answer_relevancy=MetricScore(score=float(scores["answer_relevancy"]), confidence=0.9),
            context_precision=MetricScore(score=float(scores["context_precision"]), confidence=0.9),
            context_recall=MetricScore(score=float(scores["context_recall"]), confidence=0.9),
        )

    def _cross_validate_with_deepeval(self, request: RAGEvaluationRequest, metrics: RAGMetrics) -> None:
        """
        Run DeepEval as an independent second opinion on hallucination and
        bias. This doesn't overwrite the RAGAS scores — it's a cross-check,
        logged so a mismatch between the two is visible instead of hidden.
        """
        test_case = LLMTestCase(
            input=request.question,
            actual_output=request.generated_answer,
            context=request.retrieved_context,
        )

        hallucination = HallucinationMetric(threshold=0.5)
        hallucination.measure(test_case)

        bias = BiasMetric(threshold=0.5)
        bias.measure(test_case)

        logger.info(
            "DeepEval cross-check — hallucination score: %.2f (reason: %s) | bias score: %.2f",
            hallucination.score,
            hallucination.reason,
            bias.score,
        )

    def _assess_overall_quality(self, metrics: RAGMetrics) -> OverallQuality:
        """Aggregate metrics into overall quality."""
        scores = [
            metrics.faithfulness.score,
            metrics.answer_relevancy.score,
            metrics.context_precision.score,
            metrics.context_recall.score,
        ]

        overall_score = sum(scores) / len(scores)
        threshold = 0.75

        if overall_score >= threshold:
            status = RAGEvaluationStatus.PASS
        elif overall_score >= threshold - 0.1:
            status = RAGEvaluationStatus.WARN
        else:
            status = RAGEvaluationStatus.FAIL

        return OverallQuality(
            score=overall_score,
            status=status,
            threshold=threshold,
            passed=(overall_score >= threshold),
        )

    def _generate_recommendations(self, metrics: RAGMetrics, request: RAGEvaluationRequest) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if metrics.faithfulness.score < self.thresholds["faithfulness"]:
            recommendations.append("Faithfulness below threshold - ensure answer adheres to context")

        if metrics.answer_relevancy.score < self.thresholds["answer_relevancy"]:
            recommendations.append("Answer relevancy low - review question understanding")

        if metrics.context_precision.score < self.thresholds["context_precision"]:
            recommendations.append("Context precision low - improve retrieval ranking")

        if metrics.context_recall.score < self.thresholds["context_recall"]:
            recommendations.append("Context recall low - expand retrieval window")

        if not recommendations:
            recommendations.append("RAG pipeline performing well")

        return recommendations
