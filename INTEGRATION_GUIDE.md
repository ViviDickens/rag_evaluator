# RAGAS + DeepEval + LangSmith Integration Guide

## Overview

This guide explains how the three frameworks work together in the RAG Evaluator:

```
Input RAG Output
    ↓
RAGAS Evaluation
    ├─ Faithfulness (answer adherence to context)
    ├─ Answer Relevancy (relevance to question)
    ├─ Context Precision (retrieval quality)
    └─ Context Recall (coverage completeness)
    ↓
DeepEval Cross-Validation
    ├─ Hallucination detection
    ├─ Bias checking
    ├─ Relevancy validation
    └─ Agreement scoring
    ↓
Confidence Adjustment
    (RAGAS scores weighted by DeepEval agreement)
    ↓
Overall Quality Assessment
    ├─ Score aggregation
    ├─ Status determination (PASS/WARN/FAIL)
    └─ Recommendations generation
    ↓
LangSmith Observability
    ├─ Metric logging
    ├─ Trace recording
    ├─ Error tracking
    └─ Batch analytics
    ↓
Production Dashboard
    (LangSmith provides real-time monitoring)
```

## Why Three Frameworks?

### RAGAS (Faithfulness, Relevancy, Precision, Recall)
**Purpose:** Core RAG quality metrics based on retrieval-generation alignment

**Strengths:**
- Specifically designed for RAG systems
- Four complementary metrics covering different aspects
- Lightweight and fast
- Minimal external dependencies

**Limitations:**
- Primarily token/overlap-based (not semantic)
- No hallucination detection
- No bias analysis

### DeepEval (Hallucination, Bias, Semantics)
**Purpose:** Cross-framework validation and comprehensive quality checks

**Strengths:**
- Detects hallucinations (answers contradicting context)
- Bias and toxicity detection
- Semantic similarity evaluation
- LLM-powered validation available

**Limitations:**
- More expensive (if using LLM-based metrics)
- Slower than RAGAS
- Different metric definitions (can diverge)

### LangSmith (Observability, Debugging, Monitoring)
**Purpose:** Production observability and debugging infrastructure

**Strengths:**
- Full execution tracing
- Performance monitoring
- Error context capture
- Batch analytics and statistics
- Production dashboard

**Limitations:**
- External service dependency
- Requires API key and authentication
- Network latency for logging

## How They Work Together

### 1. RAGAS Core Evaluation
```python
# RAGAS calculates four metrics
faithfulness = measure_answer_adherence_to_context()
answer_relevancy = measure_answer_relevance_to_question()
context_precision = measure_retrieved_doc_quality()
context_recall = measure_coverage_completeness()
```

**Output:** Four metric scores (0-1) with confidence estimates

### 2. DeepEval Cross-Validation
```python
# DeepEval validates RAGAS assumptions
hallucination_score = detect_contradictions(answer, context)
bias_score = check_for_bias(answer)
relevancy_validation = semantic_similarity(question, answer)
agreement = compare_ragas_vs_deepeval_scores()
```

**Output:** Cross-validation results and agreement metrics

### 3. Confidence Adjustment
```python
# Adjust RAGAS confidence based on DeepEval agreement
adjusted_confidence = ragas_confidence * deepeval_agreement
# High agreement → higher confidence
# Low agreement → flag for investigation
```

**Output:** Adjusted metric scores with realistic confidence bounds

### 4. Overall Assessment
```python
# Aggregate into overall quality
overall_score = mean([faithfulness, relevancy, precision, recall])
status = PASS if overall_score > threshold else WARN/FAIL
recommendations = analyze_metric_gaps()
```

**Output:** Single quality score, status, and actionable recommendations

### 5. LangSmith Observability
```python
# Log everything for production monitoring
observer.log_metric("faithfulness", score)
observer.log_evaluation_result(eval_id, metrics, status)
observer.log_chain("rag_pipeline", inputs, outputs)
observer.log_llm_call(model, prompt, response, latency)
```

**Output:** Complete audit trail and production dashboard

## Real-World Example

### Scenario: Low Faithfulness Score

1. **RAGAS detects:** Faithfulness = 0.62 (below 0.85 threshold)
   - Answer uses words not in context
   - Some claims not grounded in retrieved docs

2. **DeepEval validates:** Hallucination = 0.45
   - Confirms answer has ungrounded statements
   - Detects potential hallucination patterns

3. **Confidence adjustment:** 
   - RAGAS confidence: 0.88
   - DeepEval agreement: 0.92
   - Adjusted confidence: 0.88 * 0.92 = 0.81
   - Result: High confidence that there's a real issue

4. **Overall status:** FAIL (overall score 0.71 < 0.75 threshold)

5. **Recommendations:**
   - "Ensure answer strictly adheres to retrieved context"
   - "Implement output guardrails to prevent hallucinations"

6. **LangSmith logs:**
   - Metrics: all scores and confidence levels
   - Error context: full question, answer, context
   - Trace: which evaluation step failed
   - Dashboard: visible in production monitoring

**Result:** Production team can quickly identify and fix the retrieval or generation problem.

## Integration Patterns

### Pattern 1: Simple Evaluation (Default)
```python
response = await evaluator.evaluate_single(request)
# Returns: RAGAS + DeepEval + Overall Assessment
# No LangSmith logging
```

### Pattern 2: Full Observability
```python
response = await evaluator.evaluate_single(request)
observer.log_evaluation_result(
    evaluation_id=response.evaluation_id,
    metrics={
        "faithfulness": response.metrics.faithfulness.score,
        "answer_relevancy": response.metrics.answer_relevancy.score,
        ...
    },
    recommendations=response.recommendations,
    status=response.overall_quality.status.value
)
```

### Pattern 3: Batch Processing with Statistics
```python
batch_response = await evaluator.evaluate_batch(requests)
# Returns: Per-sample evaluations + aggregate statistics
# 95% confidence intervals for each metric
# Overall pass rate and distribution analysis
```

### Pattern 4: Continuous Monitoring
```python
async with observer.create_batch_trace(batch_id, batch_size, eval_ids) as batch:
    for eval_id in eval_ids:
        result = await evaluator.evaluate_single(request)
        batch.log_item(eval_id, {"status": result.overall_quality.status.value})
    batch.log_summary({"pass_rate": 0.95})
```

## Configuration for Different Use Cases

### Development/Testing
```python
# Fast, simple evaluation (RAGAS only)
evaluator = RAGEvaluator()
observer = LangSmithObserver(api_key=None)  # Disabled
```

### Staging/CI
```python
# Full evaluation with local LangSmith simulation
evaluator = RAGEvaluator()
observer = LangSmithObserver(project_name="staging", api_key=None)
```

### Production
```python
# Full observability with remote LangSmith
evaluator = RAGEvaluator()
observer = LangSmithObserver(
    project_name="production_rag",
    api_key=os.environ["LANGSMITH_API_KEY"]
)

# Log every evaluation
@observer.trace_evaluation
async def evaluate_rag_output(request):
    response = await evaluator.evaluate_single(request)
    observer.log_evaluation_result(...)
    return response
```

## Troubleshooting

### Divergent RAGAS/DeepEval Scores
**Symptom:** RAGAS says 0.92 faithfulness, DeepEval says 0.45 hallucination

**Causes:**
- Different metric definitions
- RAGAS uses token overlap, DeepEval uses semantic analysis
- One framework more strict than the other

**Solution:**
- Investigate specific disagreements
- Check which framework aligns better with your goals
- Adjust thresholds accordingly
- Use agreement score to flag uncertain cases

### Low Confidence Scores
**Symptom:** All confidence values below 0.70

**Causes:**
- High disagreement between RAGAS and DeepEval
- Unusual or ambiguous RAG output
- Potentially problematic evaluation data

**Solution:**
- Manually review flagged cases
- Check for data quality issues
- Consider retraining evaluation models
- Log to LangSmith for team review

### LangSmith Logging Gaps
**Symptom:** Metrics not appearing in dashboard

**Causes:**
- API key not configured
- Network connectivity issues
- Batch size too large causing timeouts

**Solution:**
- Verify LANGSMITH_API_KEY is set
- Check network connectivity
- Reduce batch size
- Check LangSmith server status

## Performance Considerations

### Latency
- RAGAS: ~100-200ms per evaluation
- DeepEval: ~200-500ms per evaluation (can be slower with LLM-based metrics)
- LangSmith logging: ~50ms per log (async)
- **Total per evaluation:** ~300-700ms

### Throughput
```python
# Single evaluation
response = await evaluator.evaluate_single(request)  # 300-700ms

# Batch evaluation (5 items)
batch_response = await evaluator.evaluate_batch(requests)  # 1.5-3.5 seconds

# With LangSmith (adds 50ms per item for logging)
```

### Cost Estimation
- RAGAS: Free (local computation)
- DeepEval: Free for rule-based, costs for LLM-based metrics
- LangSmith: $0.001-0.01 per log depending on plan

## Next Steps

1. **Implement in FastAPI:** Use `app.py` as template
2. **Add authentication:** Secure API endpoints
3. **Configure LangSmith:** Get API key and project setup
4. **Deploy to production:** Use observability features
5. **Monitor metrics:** Track trends over time
6. **Iterate:** Refine thresholds based on data

---

**For more details, see:**
- RAGAS docs: https://docs.ragas.io/
- DeepEval docs: https://docs.confident-ai.com/
- LangSmith docs: https://docs.smith.langchain.com/
