# Architecture, Demos & Code Locations

This guide explains how this application works, where each piece of code is located, and how evaluations are implemented.

---

## 1. What Does This Project Do? (The Big Picture)

This project has two main parts:
1. **The Application:** An AI HR Assistant for a company called **Acme Technologies**. Employees can ask questions about company leave, benefits, remote work, expenses, and IT support.
2. **The Evaluation Framework:** Tools to test and measure whether the AI is answering accurately, not hallucinating, using correct sources, and behaving safely.

---

## 2. Where Is the Code? (Project Map)

Everything is cleanly separated so you can easily find what you are looking for:

```text
rag-agent-evaluation/
├── data/
│   ├── documents/               # The 15 mock company policy files (text files)
│   └── evaluation/              # Test question datasets (JSON files)
├── src/
│   ├── config/                  # App configuration & Gemini model setup
│   ├── ingestion/               # Loads and chunks text documents
│   ├── retrieval/               # ChromaDB vector database search
│   ├── rag/                     # The core RAG prompt and answer logic
│   ├── agent/                   # The LangGraph autonomous agent
│   ├── tools/                   # Tools the agent can use (search policies, escalate)
│   ├── evaluators/              # ALL evaluation logic (Code-based & LLM judges)
│   │   ├── code_based/          # Fast Python checks (regex, exact match, sources)
│   │   ├── llm_judges/          # Gemini judges (relevance, faithfulness, correctness)
│   │   └── agent_evaluators/    # Tests if the agent took the right steps
│   ├── langsmith/               # LangSmith tracing & dataset upload code
│   └── monitoring/              # Real-time guardrails and cost calculators
└── scripts/                     # Runnable scripts for terminal interactions & demos
```

---

## 3. How the Core RAG Pipeline Works

When an employee asks a question:
1. **User asks question:** e.g., *"How many annual leaves do I get?"*
2. **Retrieval (`src/retrieval/retriever.py`):** Looks up the top 3 most relevant text chunks from the local vector database (`chroma_db/`) using embeddings (`gemini-embedding-001`).
3. **Prompt & Generation (`src/rag/pipeline.py`):** Passes the retrieved policy chunks + the user question to Gemini 3.5 Flash.
4. **Output:** Gemini answers the question and attaches the source file name (`leave_policy.txt`).

👉 **Try it yourself in terminal:**
```bash
python scripts/run_rag.py
```

---

## 4. How the LangGraph Agent Works

Sometimes a simple RAG search isn't enough. For example, if an employee asks to report an outage or asks a question not in the documents, the system should escalate to a human.

We use **LangGraph** (`src/agent/graph.py`):
1. **`intent_router_node`**: Decides if the question is a policy question, an escalation request, or chitchat.
2. **`retrieval_node`**: Searches company policies if it's a policy question.
3. **`context_assessment_node`**: Checks if the retrieved text actually contains the answer.
4. **`answer_generation_node`**: Answers the question if context is sufficient.
5. **`escalation_node`**: If information is missing or out-of-scope, it searches the HR directory (`src/tools/policy_tools.py`) and gives the user contact emails (`hr-service@acmetechnologies.com`).

👉 **Try the agent in terminal:**
```bash
python scripts/run_agent.py
```

---

## 5. How LLM Evaluations Work (Where & What)

We evaluate the system using two main techniques:

### A. Code-Based Evaluators (`src/evaluators/code_based/metrics.py`)
These are pure Python functions that run in milliseconds and cost 0 tokens:
- **Numerical Fact Validation (`numerical_fact_evaluator`):** Uses regex to make sure numbers (like `20 days` or `5,00,000 INR`) appear in the answer.
- **Source Attribution (`source_validation_evaluator`):** Checks if the retrieved documents match the expected policy file.
- **Abstention Check (`abstention_evaluator`):** Makes sure the bot says *"I could not find this information"* when asked about something not in company policy (like pet insurance).

### B. LLM-as-a-Judge (`src/evaluators/llm_judges/judges.py`)
Here, we use Gemini as an objective judge. We use **Pydantic** (`src/evaluators/llm_judges/schemas.py`) so the judge always returns structured JSON:
- **Answer Relevance (`evaluate_answer_relevance`):** Scores 1 to 5 on whether the answer directly addresses what the user asked.
- **Faithfulness (`evaluate_faithfulness`):** Breaks the answer into atomic claims. It checks each claim against the retrieved text to ensure the model didn't invent anything.
- **Answer Correctness (`evaluate_answer_correctness`):** Compares the model's answer against a verified human reference answer.
- **Context Precision (`evaluate_context_precision`):** Checks whether the retrieved chunks are actually relevant or just noise.
- **Context Recall (`evaluate_context_recall`):** Checks whether all required facts from the reference answer were found in the retrieved context.

👉 **See code evals and LLM judges in action:**
```bash
python scripts/run_evaluation_deep_dive.py
```

---

## 6. Testing & Observability with LangSmith

**LangSmith** is our dashboard for seeing what the LLM is doing behind the scenes.
- **Tracing:** When you run `python scripts/run_rag.py`, every prompt, context chunk, token count, and latency measurement is automatically sent to LangSmith.
- **Datasets (`src/langsmith/datasets.py`):** We store our 35 test cases in LangSmith so the whole team tests against the same questions.
- **Experiments (`src/experiments/runner.py`):** Lets us test different chunk sizes (e.g. 300 characters vs 500 characters) and see side-by-side which one gives better correctness scores.

👉 **View traces:** Go to **[https://smith.langchain.com](https://smith.langchain.com)** after running any script.

---

## 7. Running Tests Locally

Before submitting code, always run the unit tests:
```bash
python -m unittest discover -s tests/unit
```
All 13 unit tests run locally using mocked objects and take less than a second (no API keys or internet connection required).

---

## 8. Summary of Useful Scripts

| Script Path | Purpose |
|---|---|
| `scripts/run_rag.py` | Interactive terminal chatbot for policy Q&A |
| `scripts/run_agent.py` | Interactive LangGraph agent with tool routing and escalation |
| `scripts/run_evaluation_deep_dive.py` | Demo comparing exact match vs numerical checks and LLM judges |
| `scripts/run_online_monitoring.py` | Simulates live traffic with guardrails and sampled LLM judges |
| `scripts/upload_datasets.py` | Uploads local test cases to LangSmith |
| `scripts/compare_experiments.py` | Compares LangSmith experiment metrics |
| `scripts/generate_evaluation_report.py` | Generates a full markdown summary report in `reports/` |
