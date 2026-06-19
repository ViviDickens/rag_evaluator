"""
LangSmith integration for observability and debugging of RAG evaluation.
Tracks metrics, traces, and provides production debugging capabilities.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from functools import wraps
import json

logger = logging.getLogger(__name__)


class LangSmithObserver:
    """
    LangSmith integration for RAG evaluation pipeline observability.
    
    Provides:
    - Request/response tracing
    - Metric logging
    - Performance tracking
    - Latency analysis
    - Error tracking with full context
    """
    
    def __init__(self, project_name: str = "rag_evaluator", api_key: Optional[str] = None):
        """
        Initialize LangSmith observer.
        
        Args:
            project_name: Project name in LangSmith (e.g., "rag_evaluator")
            api_key: LangSmith API key (defaults to LANGSMITH_API_KEY env var)
        """
        self.project_name = project_name
        self.api_key = api_key
        self.enabled = api_key is not None
        
        if self.enabled:
            logger.info(f"LangSmith observer enabled for project: {project_name}")
        else:
            logger.warning("LangSmith API key not provided. Tracing will be disabled.")
    
    def trace_evaluation(self, func):
        """
        Decorator to trace RAG evaluation execution.
        
        Logs:
        - Function inputs/outputs
        - Execution time
        - Exceptions
        - Nested calls
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace_id = self._generate_trace_id()
            start_time = datetime.utcnow()
            
            try:
                logger.info(f"[TRACE {trace_id}] Starting {func.__name__}")
                
                result = await func(*args, **kwargs)
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                self._log_trace(
                    trace_id=trace_id,
                    operation=func.__name__,
                    status="success",
                    duration_ms=duration_ms,
                    inputs=self._sanitize_inputs(*args, **kwargs),
                    outputs=self._sanitize_output(result)
                )
                
                return result
            
            except Exception as e:
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                self._log_trace(
                    trace_id=trace_id,
                    operation=func.__name__,
                    status="error",
                    duration_ms=duration_ms,
                    error=str(e),
                    error_type=type(e).__name__
                )
                
                logger.error(f"[TRACE {trace_id}] Error in {func.__name__}: {str(e)}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            trace_id = self._generate_trace_id()
            start_time = datetime.utcnow()
            
            try:
                logger.info(f"[TRACE {trace_id}] Starting {func.__name__}")
                
                result = func(*args, **kwargs)
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                self._log_trace(
                    trace_id=trace_id,
                    operation=func.__name__,
                    status="success",
                    duration_ms=duration_ms,
                    inputs=self._sanitize_inputs(*args, **kwargs),
                    outputs=self._sanitize_output(result)
                )
                
                return result
            
            except Exception as e:
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                self._log_trace(
                    trace_id=trace_id,
                    operation=func.__name__,
                    status="error",
                    duration_ms=duration_ms,
                    error=str(e),
                    error_type=type(e).__name__
                )
                
                logger.error(f"[TRACE {trace_id}] Error in {func.__name__}: {str(e)}")
                raise
        
        # Return async or sync wrapper depending on function type
        if hasattr(func, '__self__'):
            return async_wrapper if hasattr(func, '__await__') else sync_wrapper
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def log_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Log a metric to LangSmith.
        
        Examples:
        - log_metric("faithfulness_score", 0.92)
        - log_metric("context_precision", 0.95, {"context_size": 3})
        """
        self._send_to_langsmith({
            "type": "metric",
            "metric_name": metric_name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        })
        
        logger.debug(f"Metric logged: {metric_name}={value}")
    
    def log_chain(self, chain_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any], 
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Log a chain execution (useful for RAG pipeline steps).
        
        Examples:
        - Retrieval step
        - LLM generation step
        - Evaluation step
        """
        self._send_to_langsmith({
            "type": "chain",
            "chain_name": chain_name,
            "inputs": self._sanitize_inputs(inputs),
            "outputs": self._sanitize_output(outputs),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        })
        
        logger.debug(f"Chain logged: {chain_name}")
    
    def log_llm_call(self, model: str, prompt: str, response: str, 
                     tokens_used: Optional[int] = None, latency_ms: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Log LLM API call for debugging and performance tracking.
        
        Useful for understanding:
        - Which model was used
        - Token consumption
        - Response latency
        - Cost estimation
        """
        self._send_to_langsmith({
            "type": "llm_call",
            "model": model,
            "prompt_length": len(prompt.split()),
            "response_length": len(response.split()),
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        })
        
        logger.debug(f"LLM call logged: {model} ({tokens_used or '?'} tokens, {latency_ms or '?'}ms)")
    
    def log_evaluation_result(self, evaluation_id: str, metrics: Dict[str, float], 
                             recommendations: List[str], status: str):
        """
        Log complete evaluation result with all metrics.
        
        Useful for:
        - Tracking quality over time
        - Debugging metric changes
        - Identifying patterns in failures
        """
        self._send_to_langsmith({
            "type": "evaluation_result",
            "evaluation_id": evaluation_id,
            "metrics": metrics,
            "recommendations_count": len(recommendations),
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Evaluation result logged: {evaluation_id} - {status}")
    
    def log_error_context(self, error_id: str, error_message: str, 
                         context: Dict[str, Any], trace_id: Optional[str] = None):
        """
        Log error with full context for debugging.
        
        Useful for production troubleshooting:
        - Request details
        - System state
        - Full stack trace
        """
        self._send_to_langsmith({
            "type": "error",
            "error_id": error_id,
            "error_message": error_message,
            "context": context,
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.error(f"Error logged: {error_id} ({error_message})")
    
    def create_batch_trace(self, batch_id: str, batch_size: int, 
                          evaluation_ids: List[str]) -> "BatchTrace":
        """
        Create a batch trace for coordinated logging of multiple evaluations.
        
        Useful for:
        - Tracking batch operations
        - Aggregating metrics
        - Understanding batch-level patterns
        """
        return BatchTrace(
            observer=self,
            batch_id=batch_id,
            batch_size=batch_size,
            evaluation_ids=evaluation_ids
        )
    
    def _send_to_langsmith(self, data: Dict[str, Any]):
        """Send data to LangSmith API (would be actual HTTP call in production)."""
        if not self.enabled:
            return
        
        # In production, this would make HTTP POST to LangSmith API
        # For now, just log locally
        logger.debug(f"[LangSmith] {json.dumps(data, default=str)}")
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        import uuid
        return f"trace-{uuid.uuid4().hex[:8]}"
    
    def _sanitize_inputs(self, *args, **kwargs) -> Dict[str, Any]:
        """Sanitize function inputs for logging (remove sensitive data)."""
        sanitized = {
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys())
        }
        return sanitized
    
    def _sanitize_output(self, output: Any) -> Dict[str, Any]:
        """Sanitize function output for logging."""
        if hasattr(output, '__dict__'):
            return {
                "type": type(output).__name__,
                "fields": list(output.__dict__.keys())
            }
        elif isinstance(output, dict):
            return {
                "type": "dict",
                "keys": list(output.keys())
            }
        else:
            return {"type": type(output).__name__}
    
    def _log_trace(self, trace_id: str, operation: str, status: str, duration_ms: float,
                   inputs: Optional[Dict[str, Any]] = None, 
                   outputs: Optional[Dict[str, Any]] = None,
                   error: Optional[str] = None,
                   error_type: Optional[str] = None):
        """Internal method to log trace data."""
        trace_data = {
            "trace_id": trace_id,
            "operation": operation,
            "status": status,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if inputs:
            trace_data["inputs"] = inputs
        if outputs:
            trace_data["outputs"] = outputs
        if error:
            trace_data["error"] = error
            trace_data["error_type"] = error_type
        
        self._send_to_langsmith(trace_data)


class BatchTrace:
    """
    Context manager for batch evaluation tracing.
    
    Usage:
        async with observer.create_batch_trace(batch_id, batch_size, eval_ids) as batch:
            for eval_id in eval_ids:
                batch.log_item(eval_id, {"status": "pass"})
            batch.log_summary({"pass_rate": 0.95})
    """
    
    def __init__(self, observer: LangSmithObserver, batch_id: str, 
                 batch_size: int, evaluation_ids: List[str]):
        self.observer = observer
        self.batch_id = batch_id
        self.batch_size = batch_size
        self.evaluation_ids = evaluation_ids
        self.start_time = datetime.utcnow()
        self.items_logged = 0
    
    async def __aenter__(self):
        self.observer._send_to_langsmith({
            "type": "batch_start",
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "timestamp": datetime.utcnow().isoformat()
        })
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        
        self.observer._send_to_langsmith({
            "type": "batch_end",
            "batch_id": self.batch_id,
            "items_logged": self.items_logged,
            "duration_ms": duration_ms,
            "status": "error" if exc_type else "success",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def log_item(self, evaluation_id: str, item_metrics: Dict[str, Any]):
        """Log individual item in batch."""
        self.observer._send_to_langsmith({
            "type": "batch_item",
            "batch_id": self.batch_id,
            "evaluation_id": evaluation_id,
            "metrics": item_metrics,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.items_logged += 1
    
    def log_summary(self, summary_metrics: Dict[str, Any]):
        """Log batch summary statistics."""
        self.observer._send_to_langsmith({
            "type": "batch_summary",
            "batch_id": self.batch_id,
            "summary": summary_metrics,
            "timestamp": datetime.utcnow().isoformat()
        })
