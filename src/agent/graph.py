import json
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.config.models import get_llm
from src.agent.state import AgentState
from src.tools.policy_tools import search_company_policy, search_hr_escalation_directory, create_support_ticket

class IntentClassification(BaseModel):
    intent: Literal["policy_inquiry", "support_escalation", "general_chitchat", "unanswerable_or_out_of_scope"] = Field(
        description="Categorize the user's intent"
    )
    suggested_tool: Literal["search_company_policy", "search_hr_escalation_directory", "create_support_ticket", "none"] = Field(
        description="Optimal tool to invoke"
    )
    confidence: float = Field(description="Confidence between 0.0 and 1.0")
    reason: str = Field(description="Reason for intent classification")

# --- Agent Graph Nodes ---

def intent_router_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Classify intent and route query."""
    question = state["question"]
    llm = get_llm().with_structured_output(IntentClassification)
    
    prompt = f"""You are the initial intent triage router for the Acme Technologies Employee Assistant.
Classify the user's question into one of:
- 'policy_inquiry': Queries about leaves, benefits, travel, attendance, expenses, IT policies, hardware, payroll.
- 'support_escalation': Explicit request to file a ticket, escalate a dispute, report an outage, or contact human HR/IT.
- 'general_chitchat': Greetings or meta conversations.
- 'unanswerable_or_out_of_scope': Inquiries clearly outside company policy (e.g. general trivia, illegal activity).

User Question: {question}"""
    
    classification: IntentClassification = llm.invoke(prompt)
    
    trajectory = list(state.get("trajectory", []))
    trajectory.append(f"router:{classification.intent}")
    
    return {
        "intent": classification.intent,
        "chosen_tool": classification.suggested_tool if classification.suggested_tool != "none" else None,
        "trajectory": trajectory,
    }

def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Retrieve policy evidence using search_company_policy."""
    question = state["question"]
    
    tool_res = search_company_policy.invoke({"query": question, "k": 3})
    
    # Extract sources from output
    sources = []
    for line in tool_res.splitlines():
        if line.startswith("Document: "):
            src = line.replace("Document: ", "").split()[0].strip()
            if src not in sources:
                sources.append(src)
                
    trajectory = list(state.get("trajectory", []))
    trajectory.append("retrieval:search_company_policy")
    
    return {
        "chosen_tool": "search_company_policy",
        "tool_arguments": {"query": question, "k": 3},
        "tool_output": tool_res,
        "retrieved_context": tool_res,
        "retrieved_sources": sources,
        "trajectory": trajectory,
    }

class ContextSufficiency(BaseModel):
    is_sufficient: bool = Field(description="True if context contains information to address the question, False otherwise")
    reason: str = Field(description="Assessment explanation")

def context_assessment_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Assess whether retrieved context is sufficient or requires escalation."""
    question = state["question"]
    context = state.get("retrieved_context", "")
    
    llm = get_llm().with_structured_output(ContextSufficiency)
    prompt = f"""Determine whether the retrieved context contains sufficient factual information to answer the question.
If the context says 'No relevant company policy documents found' or lacks the necessary rules, mark is_sufficient=false.

Question: {question}
Retrieved Context: {context}"""
    
    res: ContextSufficiency = llm.invoke(prompt)
    
    trajectory = list(state.get("trajectory", []))
    trajectory.append(f"assessment:{'sufficient' if res.is_sufficient else 'insufficient'}")
    
    return {
        "is_context_sufficient": res.is_sufficient,
        "requires_escalation": not res.is_sufficient,
        "trajectory": trajectory,
    }

def answer_generation_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Synthesize policy answer from verified context."""
    question = state["question"]
    context = state.get("retrieved_context", "")
    
    llm = get_llm()
    prompt = f"""You are the Acme Technologies HR & Company Knowledge Assistant.
Answer the user's question clearly, politely, and concisely using only the information in the context below.
Cite the document source at the end.

Context:
{context}

Question:
{question}

Answer:"""
    
    res = llm.invoke(prompt)
    answer_text = res.content if hasattr(res, "content") else str(res)
    if isinstance(answer_text, list) and len(answer_text) > 0 and isinstance(answer_text[0], dict) and "text" in answer_text[0]:
        answer_text = answer_text[0]["text"]
        
    trajectory = list(state.get("trajectory", []))
    trajectory.append("generation:policy_answer")
    
    return {
        "final_response": answer_text,
        "trajectory": trajectory,
    }

def escalation_node(state: AgentState) -> Dict[str, Any]:
    """Node 5: Route unanswerable questions to HR/IT escalation contacts."""
    question = state["question"]
    tool_args = {"topic_or_department": question}
    
    # Query escalation directory
    escalation_info = search_hr_escalation_directory.invoke(tool_args)
    
    response = (
        "I could not find definitive guidance on this matter in the approved company documents.\n\n"
        "To ensure you receive accurate assistance, your inquiry can be addressed by human support:\n"
        f"{escalation_info}\n\n"
        "You may reach out directly or email hr-service@acmetechnologies.com (Slack: #hr-helpdesk)."
    )
    
    trajectory = list(state.get("trajectory", []))
    trajectory.append("escalation:hr_directory")
    
    return {
        "final_response": response,
        "chosen_tool": "search_hr_escalation_directory",
        "tool_arguments": tool_args,
        "tool_output": escalation_info,
        "escalation_details": {"directory_guidance": escalation_info},
        "trajectory": trajectory,
    }

def general_response_node(state: AgentState) -> Dict[str, Any]:
    """Node 6: Handle general chitchat or out-of-scope queries politely."""
    question = state["question"]
    response = (
        f"Hello! I am the Acme Technologies Knowledge Assistant. "
        f"I can help you with questions about company leave, benefits, payroll, travel policies, "
        f"remote work, attendance, and IT support."
    )
    trajectory = list(state.get("trajectory", []))
    trajectory.append("general:chitchat")
    
    return {
        "final_response": response,
        "trajectory": trajectory,
    }

# --- Conditional Routing Functions ---

def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "policy_inquiry":
        return "retrieval_node"
    elif intent == "support_escalation":
        return "escalation_node"
    elif intent == "unanswerable_or_out_of_scope":
        return "escalation_node"
    else:
        return "general_response_node"

def route_after_assessment(state: AgentState) -> str:
    if state.get("is_context_sufficient", True):
        return "answer_generation_node"
    else:
        return "escalation_node"

# --- Build Compiled LangGraph Workflow ---

def build_agent_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("intent_router_node", intent_router_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("context_assessment_node", context_assessment_node)
    graph.add_node("answer_generation_node", answer_generation_node)
    graph.add_node("escalation_node", escalation_node)
    graph.add_node("general_response_node", general_response_node)
    
    # Add edges
    graph.add_edge(START, "intent_router_node")
    graph.add_conditional_edges("intent_router_node", route_after_intent)
    graph.add_edge("retrieval_node", "context_assessment_node")
    graph.add_conditional_edges("context_assessment_node", route_after_assessment)
    graph.add_edge("answer_generation_node", END)
    graph.add_edge("escalation_node", END)
    graph.add_edge("general_response_node", END)
    
    return graph.compile()

# Singleton compiled agent
agent_app = build_agent_graph()

def run_agent(question: str) -> AgentState:
    """Convenience execution wrapper for the compiled LangGraph agent."""
    initial_state: AgentState = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "intent": None,
        "chosen_tool": None,
        "tool_arguments": None,
        "tool_output": None,
        "retrieved_context": None,
        "retrieved_sources": [],
        "is_context_sufficient": False,
        "requires_escalation": False,
        "escalation_details": None,
        "final_response": "",
        "trajectory": [],
    }
    
    final_state = agent_app.invoke(initial_state)
    return final_state
