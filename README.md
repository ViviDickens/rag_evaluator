# RAG Evaluator

An orchestrator that coordinates **RAGAS**, **DeepEval**, and **LangSmith** for Retrieval-Augmented Generation (RAG) quality assurance.

RAGAS runs the core metrics (faithfulness, answer relevancy, context precision/recall) against an OpenAI evaluator model. DeepEval runs an independent cross-check for hallucination and bias on the same test case. LangSmith logs each evaluation as a run for observability — locally if no API key is set, or to your LangSmith project if one is.

## Features

- **RAGAS Integration** — real `ragas.evaluate()` calls: Faithfulness, Answer Relevancy, Context Precision/Recall
- **DeepEval Cross-Validation** — independent hallucination and bias check on the same input, logged alongside the RAGAS scores rather than merged into them
- **LangSmith Observability** — evaluation runs logged via the LangSmith SDK; falls back to local-only logging if no API key is configured
- **FastAPI REST API** — easy integration with RAG pipelines

## Installation

### 1. Clone and setup

```bash
git clone https://github.com/ViviDickens/rag_evaluator.git
cd rag_evaluator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY="sk-..."        # required — RAGAS evaluator LLM + embeddings
export LANGSMITH_API_KEY="ls-..."     # optional — omit to log locally only
```

## Quick Start

### Run tests

```bash
pytest tests/ -v
```

The end-to-end test (`test_evaluate_single`) calls the real OpenAI API and is skipped automatically if `OPENAI_API_KEY` isn't set, so it won't fail CI on a runner without the secret. The other tests don't hit any API.

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

- `GET /health` — Health check
- `GET /detectors` — List available metrics
- `POST /evaluate/rag` — Evaluate RAG output
- `GET /docs` — Swagger UI documentation

## Project Structure

```
rag_evaluator/
├── rag_evaluator/           # Package
│   ├── __init__.py
│   ├── app.py               # FastAPI application
│   ├── evaluator.py         # RAGAS + DeepEval orchestration
│   ├── schemas.py           # Pydantic models
│   └── observability.py     # LangSmith integration
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_integration.py
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
└── README.md
```

## Metrics Explained

| Metric | Measures | Threshold | Source |
|--------|----------|-----------|--------|
| **Faithfulness** | Answer adherence to context | >0.85 | RAGAS |
| **Answer Relevancy** | Relevance to question | >0.80 | RAGAS |
| **Context Precision** | Retrieved doc quality | >0.90 | RAGAS |
| **Context Recall** | Coverage completeness | >0.75 | RAGAS |
| **Hallucination** | Cross-check against context | 0.5 | DeepEval |
| **Bias** | Cross-check on generated answer | 0.5 | DeepEval |

## Why OpenAI as the evaluator model

RAGAS needs an LLM to judge faithfulness/relevancy and an embeddings model for context precision/recall. This project uses OpenAI (`gpt-4o-mini` + `text-embedding-3-small`) because it has first-class, stable support in both RAGAS and LangChain with no version conflicts. A Gemini-based version was tried first and hit repeated dependency clashes between `langchain-google-genai` and RAGAS's pinned LangChain versions — a known pain point with that combination. A free-tier Groq variant (companion repo: `ragas-lab`) is also available for local experimentation without API cost.

## Requirements

- Python 3.10+
- pip/venv
- OpenAI API key (for RAGAS evaluator model)
- LangSmith API key (optional, for remote tracing)

## License

MIT

---

**Author:** Viviana Pérez
**Portfolio:** [ViviDickens](https://github.com/ViviDickens)
