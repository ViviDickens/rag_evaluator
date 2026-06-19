# RAG Evaluator lab

Lightweight RAGAS integration for validating Retrieval-Augmented Generation pipelines. Built to complement `ai-qa-decision-engine` with statistical evaluation of LLM-based RAG systems.

## Overview

This project demonstrates hands-on experience with:
- **RAGAS metrics** (Faithfulness, Answer Relevancy, Context Precision/Recall)
- **Statistical assertions** for non-deterministic AI outputs
- **Integration with DeepEval** for cross-framework validation
- **FastAPI endpoints** for RAG evaluation as a service

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:8000/docs` for Swagger UI.

## Project Structure

```
rag_evaluator_mini/
├── app.py                    # FastAPI application
├── evaluator.py              # RAGAS evaluation logic
├── schemas.py                # Pydantic models
├── rag_pipeline.py           # Mock RAG implementation
├── tests/
│   ├── test_evaluator.py     # RAGAS metrics validation
│   └── test_integration.py   # FastAPI integration tests
├── fixtures/
│   └── sample_data.json      # Test QA datasets
├── requirements.txt
├── .env.example
└── README.md
```

## Key Endpoints

### `POST /evaluate/rag`

Evaluate a complete RAG pipeline output.

**Request:**
```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "retrieved_context": ["RAG is a technique that combines...", "It improves LLM accuracy by..."],
  "generated_answer": "RAG is a method where an LLM retrieves relevant documents first...",
  "ground_truth_answer": "RAG (Retrieval-Augmented Generation) augments LLM outputs with retrieved context to improve factuality."
}
```

**Response:**
```json
{
  "evaluation_id": "eval-123",
  "metrics": {
    "faithfulness": {"score": 0.92, "confidence": 0.88},
    "answer_relevancy": {"score": 0.87, "confidence": 0.85},
    "context_precision": {"score": 0.95, "confidence": 0.90},
    "context_recall": {"score": 0.80, "confidence": 0.78}
  },
  "overall_quality": {
    "score": 0.885,
    "status": "PASS",
    "threshold": 0.75
  },
  "recommendations": [
    "Context recall slightly below optimal—consider expanding retrieval window"
  ]
}
```

### `POST /evaluate/batch`

Evaluate multiple RAG outputs with statistical confidence intervals.

**Response includes:**
- Per-sample scores
- Aggregate statistics (mean, std_dev, confidence_interval)
- Distribution analysis
- Statistical significance tests

## RAGAS Metrics Explained

| Metric | What It Measures | Threshold |
|--------|-----------------|-----------|
| **Faithfulness** | Does the answer stick to retrieved context? | >0.85 |
| **Answer Relevancy** | Is the answer relevant to the question? | >0.80 |
| **Context Precision** | Is retrieved context actually useful? | >0.90 |
| **Context Recall** | Is all needed context retrieved? | >0.75 |

## Integration with ai-qa-decision-engine

This evaluator complements your security framework:

```
ai-qa-decision-engine (Security Layer)
    ↓ validates adversarial robustness (LLM01-LLM10)
    ↓
rag_evaluator_mini (Quality Layer)
    ↓ validates RAG metrics (Faithfulness, Relevancy, etc.)
    ↓
LangSmith (Observability Layer) [next phase]
    ↓ logs, traces, debugging for production
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific evaluator tests
pytest tests/test_evaluator.py -v

# With coverage
pytest tests/ --cov=evaluator --cov-report=html
```

## Next Steps

1. Connect to production RAG pipeline
2. Add LangSmith tracing for observability
3. Implement batch evaluation with confidence intervals
4. Cross-validate with DeepEval metrics
5. Build dashboard for metric tracking

## References

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAG Evaluation Best Practices](https://www.anthropic.com/research/retrieval-augmented-generation)
- [DeepEval Integration](https://docs.confident-ai.com/)

---

**Author:** Viviana Perez  
**Portfolio:** [ViviDickens](https://github.com/ViviDickens)
