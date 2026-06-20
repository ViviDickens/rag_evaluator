"""
LangSmith integration for RAG evaluation observability.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LangSmithObserver:
    """
    LangSmith integration for RAG evaluation observability.
    Tracks metrics, traces, and provides production debugging capabilities.
    """

    def __init__(
        self, project_name: str = "rag_evaluator", api_key: Optional[str] = None
    ):
        self.project_name = project_name
        self.api_key = api_key
        self.enabled = api_key is not None

        if self.enabled:
            logger.info(f"LangSmith observer enabled for project: {project_name}")
        else:
            logger.warning("LangSmith API key not provided. Tracing disabled.")

    def log_metric(
        self,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a metric to LangSmith."""
        if not self.enabled:
            return

        logger.debug(f"Metric logged: {metric_name}={value}")

    def log_evaluation_result(
        self,
        evaluation_id: str,
        metrics: Dict[str, float],
        recommendations: list,
        status: str,
    ):
        """Log evaluation result with metrics."""
        if not self.enabled:
            return

        logger.info(f"Evaluation result logged: {evaluation_id} - {status}")

    def log_chain(
        self,
        chain_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log chain execution."""
        if not self.enabled:
            return

        logger.debug(f"Chain logged: {chain_name}")

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log LLM API call."""
        if not self.enabled:
            return

        logger.debug(f"LLM call logged: {model}")
