# AI Evaluation Guide

This guide explains how we evaluate our AI application, what each metric means, and how the code calculates them.

---

## 💡 Key Terms Explained Simply

- **Deterministic:** Something that gives the exact same result every single time (like standard Python code `2 + 2 = 4`). It doesn't guess or change its mind.
- **LLM-as-a-Judge:** Asking a separate LLM (like Gemini) to read the user's question, the retrieved documents, and the generated answer, and grade it according to a strict scoring rule.
- **Ground Truth / Reference Answer:** The official, human-verified "correct" answer that we compare the AI's response against.
- **Hallucination:** When the AI invents facts or rules that do not exist in the company documents.
- **Abstention:** When the AI safely says *"I don't know"* or *"I cannot find this information in the documents"* instead of guessing.

---

## 1. Why Two Types of Evaluation?

1. **Code-Based (Deterministic):**
   - **Cost:** $0.00
   - **Speed:** Less than 2 milliseconds.
   - **Good for:** Checking numbers, checking if the right document was found, checking if the bot refused when it was supposed to.
   - **Bad for:** Understanding synonyms or natural human phrasing.

2. **LLM-as-a-Judge (Semantic):**
   - **Cost:** Fractions of a cent.
   - **Speed:** ~1 to 2 seconds.
   - **Good for:** Understanding meaning, catching subtle lies/hallucinations, grading overall relevance.
   - **Bad for:** Running on 100% of live production traffic (too slow and expensive).

---

## 2. Code-Based Evaluators (`src/evaluators/code_based/metrics.py`)

### A. Numerical Fact Validation (`numerical_fact_evaluator`)
- **What it does:** Ensures important numbers (like `20 days` or `5,00,000 INR`) appear in the answer.
- **Why it matters:** In HR and finance, getting a number wrong (e.g. saying 10 days instead of 20) is a critical error.
- **How it works:** Strips commas so Indian formats (`5,00,000`) and standard formats (`500000`) match, and uses regex `\b{number}\b` to check for exact word boundaries.

### B. Source Validation (`source_validation_evaluator`)
- **What it does:** Checks if the documents retrieved by the system match the expected document list.
- **Score:** 1.0 if all expected documents were retrieved; lower if a document was missed.

### C. Abstention Check (`abstention_evaluator`)
- **What it does:** Tests whether the AI correctly refused to answer when asked an unanswerable question (e.g. *"Does Acme pay for pet insurance?"*).
- **How it works:** Checks if refusal phrases like *"could not find this information"* or *"not available in the provided documents"* appear in the answer.

---

## 3. LLM-as-a-Judge Evaluators (`src/evaluators/llm_judges/judges.py`)

We use Pydantic models so Gemini returns structured JSON instead of rambling text.

### A. Answer Relevance (`evaluate_answer_relevance`)
- **Question asked to judge:** *"Did the answer directly address the user's question, without going off-topic?"*
- **Scored:** 1 (irrelevant) to 5 (complete and direct).
- **Note:** Relevance does **not** check if the answer is factually true—it only checks if it answered the topic asked.

### B. Faithfulness & Hallucination (`evaluate_faithfulness`)
- **Question asked to judge:** *"Is every single claim in this answer backed up by the retrieved text?"*
- **How it works:** 
  1. The judge extracts individual claims (e.g., Claim 1: "Employees get 20 days off", Claim 2: "Employees get free coffee").
  2. The judge marks each claim as `supported`, `contradicted`, or `not_enough_information`.
  3. Score = `supported_claims / total_claims`.
- **Why this is powerful:** If the bot answers mostly correctly but slips in one hallucinated sentence, the score drops to `0.5` and points out the exact unverified claim.

### C. Answer Correctness (`evaluate_answer_correctness`)
- **Question asked to judge:** *"How well does the generated answer match the human ground-truth reference answer?"*
- **Scored:** 1 to 5.
- **Why we don't use Exact Match:** If the reference answer says *"20 days of annual leave"* and the AI says *"twenty (20) days of paid vacation"*, an exact match check fails (`0.0`), but the LLM judge correctly awards a full score (`5/5`).

### D. Retrieval Context Precision (`evaluate_context_precision`)
- **What it tests:** *Did our search engine put the most useful chunks at the top of the list?*
- **Metric:** **Mean Average Precision (MAP)**. If chunk 0 is relevant, score is high. If chunk 0 is junk and chunk 2 is relevant, score is penalized.

### E. Retrieval Context Recall (`evaluate_context_recall`)
- **What it tests:** *Did our search engine retrieve all the facts needed to answer the question, or did it miss a critical page?*
- **Score:** `found_reference_facts / total_reference_facts`.
