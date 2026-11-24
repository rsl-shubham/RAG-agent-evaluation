# LangSmith Guide

This document explains what **LangSmith** is, why we use it, and how to test with it.

---

## 💡 Key Terms Explained Simply

- **Observability / Tracing:** Seeing inside the "black box" of an AI call. A trace records the exact prompt sent to Gemini, the exact context chunks attached, the token count, the latency (speed in milliseconds), and the output.
- **Dataset (in LangSmith):** A collection of test questions and approved reference answers stored in the cloud so developers can test against the exact same test cases.
- **Experiment / Evaluation Run:** Running our RAG application through a dataset, having our evaluators automatically grade each answer, and plotting a summary table of the scores.
- **Sampling:** Only inspecting a small percentage (e.g. 20%) of live production questions with expensive LLM judges to save money while still tracking overall quality trends.

---

## 1. What Does LangSmith Do in This Project?

1. **Automatic Logging:** Whenever you run `python scripts/run_rag.py` or `python scripts/run_agent.py`, LangSmith automatically captures the trace. If an answer looks weird, you can open LangSmith and see the exact text Gemini received.
2. **Offline Experiments:** Compare two configurations. For example:
   - Configuration A: Chunks of 300 characters, retrieve top 3.
   - Configuration B: Chunks of 500 characters, retrieve top 5.
   LangSmith shows you side-by-side: *Did Configuration B improve correctness? Did it make the response slower or more expensive?*
3. **Production Feedback:** Attaches user ratings (thumbs-up / thumbs-down) directly to traces.

---

## 2. Quick How-To: The 3 Main LangSmith Tasks

### Step 1: Upload Datasets to LangSmith
We have 6 curated test datasets in [`data/evaluation/`](file:///Users/shubham/Projects/rag-agent-evaluation/data/evaluation/). To upload them to LangSmith:
```bash
python scripts/upload_datasets.py --category all
```
*Note:* This script is **idempotent** (meaning if the dataset already exists, it skips it instead of making messy duplicates).

### Step 2: Run an Offline Experiment
To test the RAG app against the test dataset:
```bash
python scripts/run_langsmith_experiments.py --experiment exp-chunk500-k3
```

### Step 3: Compare Experiments in Terminal
To see a summary comparison table:
```bash
python scripts/compare_experiments.py exp-chunk300-k3 exp-chunk500-k3 exp-chunk500-k5
```

---

## 3. How to View Your Traces in the Browser

1. Go to **[https://smith.langchain.com](https://smith.langchain.com)**
2. Click on your project: **`acme-rag-development`**
3. Click on any run to inspect:
   - **Inputs:** The question asked.
   - **Context:** The text retrieved from ChromaDB.
   - **Tokens & Latency:** How many tokens were used and how many seconds it took.
   - **Feedback:** Any automatic scores from code checks or LLM judges attached to that run.
