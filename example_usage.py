"""
Example usage of RAG Evaluator with RAGAS, DeepEval, and LangSmith integration.
Shows complete evaluation pipeline for production RAG systems.
"""

import asyncio
from datetime import datetime

from schemas import RAGEvaluationRequest
from evaluator import RAGEvaluator
from observability import LangSmithObserver


async def main():
    """Complete example of RAG evaluation pipeline."""
    
    print("=" * 80)
    print("RAG Evaluator: RAGAS + DeepEval + LangSmith Integration")
    print("=" * 80)
    print()
    
    # Initialize components
    evaluator = RAGEvaluator()
    observer = LangSmithObserver(
        project_name="production_rag",
        api_key=None  # Set to actual API key for production
    )
    
    # Sample RAG system output
    rag_request = RAGEvaluationRequest(
        question="How do I integrate LangSmith with my RAG system?",
        retrieved_context=[
            "LangSmith is an observability platform for LLM applications. It provides tracing, "
            "debugging, and monitoring capabilities. You can integrate it by installing the langsmith "
            "package and setting up API keys.",
            
            "To trace your RAG pipeline with LangSmith, wrap your chain steps with the @langsmith.trace "
            "decorator or use the client.trace() context manager. This captures inputs, outputs, and "
            "metadata for each step.",
            
            "LangSmith's observability features help you debug failing evaluations by showing the full "
            "execution trace, including retrieval results, LLM calls, and evaluation metrics."
        ],
        generated_answer="To integrate LangSmith with your RAG system, install the langsmith package, "
                         "set up your API key, and use the @trace decorator or client.trace() context "
                         "manager to wrap your chain steps. This captures the full execution trace for "
                         "debugging and monitoring.",
        ground_truth_answer="Integrate LangSmith by installing the package, configuring API credentials, "
                           "and wrapping chain steps with tracing decorators to capture inputs, outputs, "
                           "and metadata for observability and debugging."
    )
    
    print("INPUT:")
    print(f"  Question: {rag_request.question}")
    print(f"  Context docs: {len(rag_request.retrieved_context)}")
    print(f"  Generated answer length: {len(rag_request.generated_answer)} chars")
    print()
    
    # Step 1: RAGAS Evaluation
    print("STEP 1: RAGAS Evaluation")
    print("-" * 80)
    
    response = await evaluator.evaluate_single(rag_request)
    
    print(f"Evaluation ID: {response.evaluation_id}")
    print()
    print("RAGAS Metrics:")
    print(f"  Faithfulness:      {response.metrics.faithfulness.score:.2f} "
          f"(confidence: {response.metrics.faithfulness.confidence:.2f})")
    print(f"  Answer Relevancy:  {response.metrics.answer_relevancy.score:.2f} "
          f"(confidence: {response.metrics.answer_relevancy.confidence:.2f})")
    print(f"  Context Precision: {response.metrics.context_precision.score:.2f} "
          f"(confidence: {response.metrics.context_precision.confidence:.2f})")
    print(f"  Context Recall:    {response.metrics.context_recall.score:.2f} "
          f"(confidence: {response.metrics.context_recall.confidence:.2f})")
    print()
    
    # Step 2: DeepEval Cross-Validation
    print("STEP 2: DeepEval Cross-Validation")
    print("-" * 80)
    
    cross_val = response.metadata.get("cross_validation", {})
    for metric_name, result in cross_val.items():
        print(f"{metric_name.title()}:")
        print(f"  RAGAS score:    {result.ragas_score:.2f}")
        print(f"  DeepEval score: {result.deepeval_score:.2f}")
        print(f"  Agreement:      {result.agreement:.2f}")
        print(f"  Recommendation: {result.recommendation}")
        print()
    
    # Step 3: Overall Quality Assessment
    print("STEP 3: Overall Quality Assessment")
    print("-" * 80)
    
    print(f"Overall Score:  {response.overall_quality.score:.2f}")
    print(f"Status:         {response.overall_quality.status.value}")
    print(f"Threshold:      {response.overall_quality.threshold:.2f}")
    print(f"Passed:         {'Yes' if response.overall_quality.passed else 'No'}")
    print()
    
    # Step 4: Recommendations
    print("STEP 4: Recommendations")
    print("-" * 80)
    
    for i, rec in enumerate(response.recommendations, 1):
        print(f"{i}. {rec}")
    print()
    
    # Step 5: LangSmith Observability Logging
    print("STEP 5: LangSmith Observability Logging")
    print("-" * 80)
    
    # Log individual metrics
    observer.log_metric("faithfulness_score", response.metrics.faithfulness.score)
    observer.log_metric("answer_relevancy_score", response.metrics.answer_relevancy.score)
    observer.log_metric("context_precision_score", response.metrics.context_precision.score)
    observer.log_metric("context_recall_score", response.metrics.context_recall.score)
    
    print("✓ Individual metrics logged to LangSmith")
    
    # Log evaluation result
    observer.log_evaluation_result(
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
    
    print("✓ Evaluation result logged to LangSmith")
    
    # Log chain execution (RAG pipeline step)
    observer.log_chain(
        chain_name="rag_evaluation_pipeline",
        inputs={
            "question": rag_request.question,
            "context_docs": len(rag_request.retrieved_context)
        },
        outputs={
            "evaluation_id": response.evaluation_id,
            "status": response.overall_quality.status.value,
            "score": response.overall_quality.score
        },
        metadata={
            "evaluation_framework": "RAGAS + DeepEval",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    print("✓ Chain execution logged to LangSmith")
    print()
    
    # Batch Evaluation Example
    print("=" * 80)
    print("BONUS: Batch Evaluation with Statistical Analysis")
    print("=" * 80)
    print()
    
    batch_requests = [
        RAGEvaluationRequest(
            question=f"Technical question {i+1}",
            retrieved_context=[f"Context for question {i+1}"],
            generated_answer=f"Answer to technical question {i+1}",
            ground_truth_answer=f"Ground truth answer {i+1}"
        )
        for i in range(5)
    ]
    
    print(f"Evaluating batch of {len(batch_requests)} RAG outputs...")
    batch_response = await evaluator.evaluate_batch(batch_requests)
    print()
    
    print(f"Batch ID: {batch_response.batch_id}")
    print(f"Overall Pass Rate: {batch_response.overall_pass_rate*100:.1f}%")
    print(f"Summary: {batch_response.summary}")
    print()
    
    if batch_response.statistics:
        print("Statistical Analysis:")
        print("-" * 80)
        for metric_name, stats in batch_response.statistics.items():
            print(f"{metric_name}:")
            print(f"  Mean:              {stats.mean:.2f}")
            print(f"  Std Dev:           {stats.std_dev:.2f}")
            print(f"  Range:             [{stats.min:.2f}, {stats.max:.2f}]")
            print(f"  95% CI:            [{stats.confidence_interval_95[0]:.2f}, "
                  f"{stats.confidence_interval_95[1]:.2f}]")
            print()
    
    # Log batch metrics to LangSmith
    observer.log_metric("batch_pass_rate", batch_response.overall_pass_rate,
                       metadata={"batch_id": batch_response.batch_id})
    
    print("✓ Batch metrics logged to LangSmith")
    print()
    
    print("=" * 80)
    print("Evaluation pipeline complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
