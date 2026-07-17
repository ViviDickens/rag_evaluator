"""
LangSmith integration for RAG evaluation observability.

Uses the real langsmith Client to create runs. If no LANGSMITH_API_KEY
is set, falls back to local-only logging instead of silently pretending
to have sent data — that's the difference from the previous version.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LangSmithObserver:
    """
    Thin wrapper around the LangSmith SDK for evaluation observability.
    """

    def __init__(self, project_name: str = "rag_evaluator", api_key: Optional[str] = None):
        self.project_name = project_name
        self.api_key = api_key
        self.enabled = api_key is not None
        self._client = None

        if self.enabled:
            from langsmith import Client

            self._client = Client(api_key=api_key)
            logger.info(f"LangSmith observer enabled for project: {project_name}")
        else:
            logger.warning("LangSmith API key not provided. Tracing disabled — logging locally only.")

    def log_evaluation_result(
        self,
        evaluation_id: str,
        metrics: Dict[str, float],
        recommendations: list,
        status: str,
    ):
        """Log an evaluation result as a LangSmith run."""
        if not self.enabled:
            logger.info(f"[local-only] Evaluation result: {evaluation_id} - {status} - {metrics}")
            return

        self._client.create_run(
            name="rag_evaluation",
            run_type="chain",
            project_name=self.project_name,
            inputs={"evaluation_id": evaluation_id},
            outputs={"metrics": metrics, "recommendations": recommendations, "status": status},
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
        )

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log an individual LLM call as a LangSmith run."""
        if not self.enabled:
            logger.debug(f"[local-only] LLM call: {model}")
            return

        self._client.create_run(
            name=f"llm_call:{model}",
            run_type="llm",
            project_name=self.project_name,
            inputs={"prompt": prompt},
            outputs={"response": response},
            extra={"tokens_used": tokens_used, "latency_ms": latency_ms, **(metadata or {})},
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
        )
