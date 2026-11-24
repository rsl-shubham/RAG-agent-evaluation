# RAG Agent & LLM Evaluation Framework

An enterprise implementation of **RAG Evaluation, LLM-as-a-Judge, LangSmith Experiments, and LangGraph Agent Evaluation** built for Acme Technologies employee knowledge and HR support.

---

## ⚡ Quickstart

### 1. Setup Environment
```bash
# Clone and enter workspace
cd rag-agent-evaluation

# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
```
Ensure your `.env` contains your Google and LangSmith keys:
```bash
GOOGLE_API_KEY=your_google_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=acme-rag-development
```

### 2. Ingest Policy Documents
Index the 15 enterprise company documents into ChromaDB:
```bash
python scripts/ingest_documents.py --force-reindex
```

---

## 💬 How to Run the Chatbots for Interactions

You have two interactive chatbots available in your terminal:

### A. Standard RAG Chatbot (Direct Document Q&A)
Fast, direct question-answering with source attribution:
```bash
python scripts/run_rag.py
```
*Example question to type:* `How many days of paid annual leave do employees receive per year?`

### B. Autonomous LangGraph Agent (With Reasoning & Human Escalation)
Multi-step agent that classifies intent, assesses context sufficiency, and routes unanswerable queries to human HR/IT support:
```bash
python scripts/run_agent.py
```
*Example question to type:* `Does Acme Technologies cover pet health insurance?` *(Watch it route to HR escalation).*

---

## 🧪 Quick Overview: How to Test on LangSmith

LangSmith lets you trace your runs, test benchmark datasets, and run offline experiments.

### 1. Upload Datasets to LangSmith
Upload the 6 curated evaluation datasets to your LangSmith project:
```bash
python scripts/upload_datasets.py --category all
```

### 2. Run an Evaluation Experiment
Test the RAG application against a dataset in LangSmith:
```bash
python scripts/run_langsmith_experiments.py --experiment exp-chunk500-k3
```

### 3. Compare Multiple Experiments
Compare latency, correctness, and faithfulness across chunking configurations:
```bash
python scripts/compare_experiments.py exp-chunk300-k3 exp-chunk500-k3 exp-chunk500-k5
```
👉 Open **[https://smith.langchain.com](https://smith.langchain.com)** to view full visual traces, token usage, and side-by-side run comparisons.

---

## 🎯 Evaluation Scripts

| Task | Command | Description |
|---|---|---|
| **Deep Dive Evals** | `python scripts/run_evaluation_deep_dive.py` | Shows exact match vs numerical checks and LLM judge hallucination detection. |
| **Online Monitoring** | `python scripts/run_online_monitoring.py` | Simulates live traffic with 100% code guardrails and sampled LLM judges. |
| **Code-Based Evals** | `python scripts/run_code_evaluations.py` | Tests exact match, string similarity, and numerical fact extraction. |
| **LLM-as-a-Judge Evals**| `python scripts/run_llm_evaluations.py` | Runs Relevance, Faithfulness, Correctness, Precision, and Recall judges. |
| **Agent Evaluations** | `python scripts/run_agent_evaluations.py` | Evaluates agent routing, tool selection, and execution trajectory. |
| **Generate Report** | `python scripts/generate_evaluation_report.py` | Generates `reports/comprehensive_evaluation_report.md`. |
| **Run Unit Tests** | `python -m unittest discover -s tests/unit` | Runs all 13 unit tests locally (no paid API calls needed). |

---

## 📚 Documentation Index (`docs/`)

All project documentation is located in the [`docs/`](file:///Users/shubham/Projects/rag-agent-evaluation/docs/) folder:

- [docs/DEVELOPER_GUIDE.md](file:///Users/shubham/Projects/rag-agent-evaluation/docs/DEVELOPER_GUIDE.md): **Start here for new joinees!** Simple walkthrough of architecture, files, and where code lives.
- [docs/ARCHITECTURE.md](file:///Users/shubham/Projects/rag-agent-evaluation/docs/ARCHITECTURE.md): System architecture diagrams and data flow.
- [docs/EVALUATION_GUIDE.md](file:///Users/shubham/Projects/rag-agent-evaluation/docs/EVALUATION_GUIDE.md): Mathematical formulations, rubrics, and metric limitations.
- [docs/LANGSMITH_GUIDE.md](file:///Users/shubham/Projects/rag-agent-evaluation/docs/LANGSMITH_GUIDE.md): LangSmith setup, experiments, and monitoring.
- [docs/AGENT_EVALUATION_GUIDE.md](file:///Users/shubham/Projects/rag-agent-evaluation/docs/AGENT_EVALUATION_GUIDE.md): Multi-aspect agent evaluation guide.
- [data/documents/MANIFEST.md](file:///Users/shubham/Projects/rag-agent-evaluation/data/documents/MANIFEST.md): Synthetic enterprise policies and design intent.
