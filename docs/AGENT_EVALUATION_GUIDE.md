# LangGraph Agent Evaluation Guide

This document explains how our autonomous agent makes decisions, and how we evaluate the **steps it took**, not just its final answer.

---

## 💡 Key Terms Explained Simply

- **Agent State:** A dictionary (`AgentState`) passed between nodes in the graph that keeps track of the conversation, the chosen tool, the search results, and whether escalation is needed.
- **Node:** A single Python function in the graph that does one specific job (e.g. classify intent, search documents, or generate an answer).
- **Conditional Edge:** An `if/else` fork in the graph. For example: *If context is sufficient -> generate answer; If context is missing -> escalate to human.*
- **Trajectory:** The list of steps/nodes the agent visited from start to finish. e.g. `['router:policy_inquiry', 'retrieval', 'assessment:insufficient', 'escalation']`.

---

## 1. Why Do We Evaluate the Agent Differently than Simple RAG?

In simple RAG, there is only one step: *retrieve and answer*.
In an agent, things can go wrong along the path even if the final text looks okay:
- The agent might pick the wrong tool.
- The agent might pass invalid arguments to a tool.
- The agent might make 10 unnecessary tool calls when 1 was enough.
- The agent might hallucinate an answer instead of escalating when documents are missing.

Therefore, we evaluate **4 distinct aspects** of the agent's behavior (`src/evaluators/agent_evaluators/evaluators.py`):

---

## 2. The 4 Agent Evaluation Metrics

| Metric | What it checks | Example |
|---|---|---|
| **1. Intent Routing Accuracy** (`evaluate_routing`) | Did the router classify the user's intent correctly? | Question: *"I want to report harassment"* -> Must route to `support_escalation`, not `general_chitchat`. |
| **2. Tool Selection Accuracy** (`evaluate_tool_selection`) | Did the agent choose the right tool for the job? | Policy question -> Must invoke `search_company_policy`. Outage -> Must invoke `search_hr_escalation_directory`. |
| **3. Tool Execution Validity** (`evaluate_tool_execution`) | Were tool arguments non-empty, and did the tool return an output? | Verifies the agent didn't pass empty strings `{}` or crash. |
| **4. Trajectory Fidelity** (`evaluate_trajectory`) | Did the agent follow the intended sequence of milestones? | Unanswerable question must visit: `router -> retrieval -> assessment -> escalation`. |

---

## 3. How to Run Agent Evaluations

Run the automated agent evaluation script:
```bash
python scripts/run_agent_evaluations.py
```

The output prints each test case, the trajectory taken, and saves the detailed results to `reports/agent_evaluation_results.json`.
