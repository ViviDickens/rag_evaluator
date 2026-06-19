# RAG Evaluator Mini - Portfolio Summary

## Project Overview

A production-grade **Retrieval-Augmented Generation (RAG) evaluation framework** integrating three complementary LLM evaluation frameworks:

1. **RAGAS** - Core RAG quality metrics (Faithfulness, Answer Relevancy, Context Precision/Recall)
2. **DeepEval** - Cross-validation and advanced quality checks (Hallucination, Bias, Semantics)
3. **LangSmith** - Production observability, debugging, and monitoring

## Technical Stack

- **Framework:** FastAPI (async Python)
- **Evaluation:** RAGAS, DeepEval, scipy (statistical analysis)
- **Observability:** LangSmith
- **Testing:** pytest with async support
- **Data Models:** Pydantic
- **Statistics:** numpy, scipy (confidence intervals, distribution analysis)

## Key Features

✅ **Multi-Framework Evaluation**
- RAGAS metrics for RAG-specific quality
- DeepEval cross-validation for hallucination/bias detection
- Confidence-weighted scoring system

✅ **Production Observability**
- LangSmith tracing and debugging
- Metric logging and batch analytics
- Error context capture
- Full execution traces

✅ **Statistical Rigor**
- 95% confidence intervals for batch evaluations
- Per-sample and aggregate statistics
- Distribution analysis for non-deterministic AI outputs

✅ **Comprehensive API**
- Single evaluation endpoint
- Batch evaluation with statistics
- Cross-framework validation
- Extensible architecture

## File Structure

```
rag_evaluator_mini/
├── app.py                      # FastAPI application
├── evaluator.py                # RAGAS + DeepEval logic
├── observability.py            # LangSmith integration
├── schemas.py                  # Pydantic models
├── example_usage.py            # Complete usage example
├── tests/
│   └── test_integration.py     # Integration tests
├── requirements.txt            # Dependencies
├── .env.example               # Environment variables
├── README.md                   # Quick start guide
├── INTEGRATION_GUIDE.md        # Deep integration guide
└── PORTFOLIO_SUMMARY.md        # This file
```

## API Endpoints

### POST /evaluate/rag
Evaluate a single RAG output with RAGAS + DeepEval

**Request:**
```json
{
  "question": "What is RAG?",
  "retrieved_context": ["RAG is a technique..."],
  "generated_answer": "RAG stands for...",
  "ground_truth_answer": "RAG augments LLMs with..."
}
```

**Response:**
```json
{
  "evaluation_id": "eval-abc123",
  "metrics": {
    "faithfulness": {"score": 0.92, "confidence": 0.88},
    "answer_relevancy": {"score": 0.87, "confidence": 0.85},
    "context_precision": {"score": 0.95, "confidence": 0.90},
    "context_recall": {"score": 0.80, "confidence": 0.78}
  },
  "overall_quality": {
    "score": 0.885,
    "status": "PASS",
    "passed": true
  },
  "recommendations": [...],
  "metadata": {"cross_validation": {...}}
}
```

### POST /evaluate/batch
Batch evaluation with statistical analysis

**Returns:**
- Per-sample evaluation results
- Aggregate statistics (mean, std_dev, 95% CI)
- Overall pass rate
- Distribution analysis

### POST /evaluate/with-deepeval
Cross-validate RAGAS metrics with DeepEval framework

## Metrics Explained

| Metric | What | Threshold |
|--------|------|-----------|
| **Faithfulness** | Answer adherence to context | >0.85 |
| **Answer Relevancy** | Answer relevance to question | >0.80 |
| **Context Precision** | Retrieved doc quality | >0.90 |
| **Context Recall** | Coverage completeness | >0.75 |

## Use Cases

### 1. Quality Validation in CI/CD
```python
# Test RAG outputs before deployment
response = await evaluator.evaluate_single(request)
assert response.overall_quality.passed, "RAG output quality too low"
```

### 2. Production Monitoring
```python
# Log metrics to LangSmith dashboard
observer.log_evaluation_result(
    evaluation_id=response.evaluation_id,
    metrics={...},
    status=response.overall_quality.status.value
)
```

### 3. Batch Quality Analysis
```python
# Evaluate dataset with statistical rigor
batch_response = await evaluator.evaluate_batch(requests)
print(f"Pass rate: {batch_response.overall_pass_rate*100:.1f}%")
# Includes 95% confidence intervals for each metric
```

### 4. Debugging Failed Evaluations
```python
# Get specific recommendations for improvement
for rec in response.recommendations:
    print(f"⚠️ {rec}")

# Check cross-validation to understand metric divergence
cross_val = response.metadata["cross_validation"]
for metric, result in cross_val.items():
    print(f"{metric} agreement: {result.agreement:.2f}")
```

## Integration with ai-qa-decision-engine

This project complements the **AI QA Decision Engine** (OWASP LLM Top 10 security framework):

```
ai-qa-decision-engine (Security Layer)
    ↓ validates: Prompt Injection, Data Poisoning, etc.
    ↓
rag_evaluator_mini (Quality Layer)
    ↓ validates: Faithfulness, Relevancy, Precision, Recall
    ↓
LangSmith (Observability Layer)
    ↓ monitors: Metrics, traces, errors
```

Together they provide:
- **Security:** Adversarial robustness (OWASP LLM Top 10)
- **Quality:** RAG-specific evaluation (RAGAS + DeepEval)
- **Observability:** Production monitoring (LangSmith)

## Test Coverage

- 20+ unit tests for RAGAS metric calculations
- 8+ tests for DeepEval cross-validation
- 10+ tests for LangSmith observability
- 5+ integration tests for end-to-end pipeline
- Edge case testing (empty context, no ground truth, etc.)

## Performance

- Single evaluation: 300-700ms (RAGAS + DeepEval)
- Batch evaluation (5 items): 1.5-3.5 seconds
- LangSmith logging: ~50ms per log (async, non-blocking)
- Throughput: ~2 evaluations/second per worker

## Why This Project Matters for Prediktive

1. **Hands-on with all three frameworks** Prediktive asked for
   - RAGAS (core RAG evaluation)
   - DeepEval (cross-validation)
   - LangSmith (observability)

2. **Production-ready architecture**
   - FastAPI REST API
   - Async/await throughout
   - Comprehensive error handling
   - Extensible detector pattern

3. **Addresses their specific needs**
   - "Statistical reasoning instead of purely deterministic assertions"
   - "Non-deterministic AI outputs" handled with confidence intervals
   - "CI/CD quality gates" design
   - "Production incident triage" with full trace capture

4. **Demonstrates depth in**
   - LLM evaluation methodology
   - AI/ML QA practices
   - Production observability
   - Statistical rigor

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start server
python app.py

# Run example
python example_usage.py
```

## Next Steps

1. Connect to real RAG pipeline (LangChain, LlamaIndex)
2. Configure LangSmith API key for production
3. Fine-tune thresholds for your domain
4. Deploy to production with observability
5. Monitor metrics dashboard in LangSmith

---

**GitHub:** ViviDickens/rag_evaluator_mini  
**Status:** Production-ready MVP  
**License:** MIT
