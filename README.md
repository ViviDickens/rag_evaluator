# RAG Evaluator lab


An orchestrator that coordinates **RAGAS**, **DeepEval**, and **LangSmith** for comprehensive Retrieval-Augmented Generation (RAG) quality assurance.

## Features

- **RAGAS Integration** - Faithfulness, Answer Relevancy, Context Precision/Recall
- **DeepEval Cross-Validation** - Hallucination detection, bias checking
- **LangSmith Observability** - Production tracing, debugging, monitoring
- **Statistical Rigor** - Confidence intervals, distribution analysis
- **FastAPI REST API** - Easy integration with RAG pipelines

## Installation

### 1. Clone and setup

```bash
git clone https://github.com/ViviDickens/rag_evaluator.git
cd rag_evaluator
python -m venv venv
source venv/Scripts/activate  # On Windows
pip install -e .
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Run tests

```bash
pytest tests/ -v
```

### Start server

```bash
python -m rag_evaluator.app
```

Visit `http://localhost:8000/docs` for API documentation.

### Example usage

```python
from rag_evaluator.evaluator import RAGEvaluator
from rag_evaluator.schemas import RAGEvaluationRequest

evaluator = RAGEvaluator()

request = RAGEvaluationRequest(
    question="What is RAG?",
    retrieved_context=["RAG augments LLMs with retrieved context..."],
    generated_answer="RAG is a technique that combines retrieval and generation...",
)

response = await evaluator.evaluate_single(request)
print(f"Score: {response.overall_quality.score}")
print(f"Status: {response.overall_quality.status}")
```

## API Endpoints

- `GET /health` - Health check
- `GET /detectors` - List available metrics
- `POST /evaluate/rag` - Evaluate RAG output
- `GET /docs` - Swagger UI documentation

## Project Structure

```
rag_evaluator/
├── rag_evaluator/           # Package
│   ├── __init__.py
│   ├── app.py              # FastAPI application
│   ├── evaluator.py        # RAGAS + DeepEval logic
│   ├── schemas.py          # Pydantic models
│   └── observability.py    # LangSmith integration
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_integration.py
├── pyproject.toml         # Package configuration
├── requirements.txt       # Dependencies
└── README.md
```

## Metrics Explained

| Metric | Measures | Threshold |
|--------|----------|-----------|
| **Faithfulness** | Answer adherence to context | >0.85 |
| **Answer Relevancy** | Relevance to question | >0.80 |
| **Context Precision** | Retrieved doc quality | >0.90 |
| **Context Recall** | Coverage completeness | >0.75 |

## Requirements

- Python 3.10+
- pip/venv

## License

MIT

---

**Author:** Viviana Perez  
**Portfolio:** [ViviDickens](https://github.com/ViviDickens)
