import json
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from src.retrieval.retriever import retrieve_documents

@tool
def search_company_policy(query: str, k: int = 3) -> str:
    """
    Search the official Acme Technologies employee handbook and company policies for answers
    regarding leaves, benefits, travel, payroll, attendance, remote work, IT assets, and security.
    """
    res = retrieve_documents(query=query, k=k)
    if not res.chunks:
        return "No relevant company policy documents found matching the search query."
        
    outputs = []
    for c in res.chunks:
        outputs.append(f"Document: {c.source} (v{c.version})\nContent:\n{c.content}\n")
    return "\n---\n".join(outputs)

@tool
def search_hr_escalation_directory(topic_or_department: str) -> str:
    """
    Lookup contact channels, email addresses, and service level agreements (SLAs)
    in the Acme HR escalation and support directory.
    """
    res = retrieve_documents(
        query=f"HR escalation channel support directory contact email SLA {topic_or_department}",
        k=2,
    )
    if not res.chunks:
        return "Tier 1 HR Helpdesk: hr-service@acmetechnologies.com (Slack: #hr-helpdesk, SLA: 4 hours)."
    return "\n\n".join(c.content for c in res.chunks)

@tool
def create_support_ticket(
    category: str,
    summary: str,
    employee_email: str,
    priority: str = "P3",
) -> str:
    """
    Mock support ticket creation. Generates an internal support ticket when an employee's
    request cannot be answered from documentation or requires formal human escalation.
    """
    import uuid
    ticket_id = f"ACM-{priority}-{uuid.uuid4().hex[:6].upper()}"
    return json.dumps({
        "status": "created",
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "summary": summary,
        "employee_email": employee_email,
        "assigned_queue": "HR-ServiceDesk-Tier1" if "hr" in category.lower() else "IT-ServiceDesk",
        "estimated_sla": "4 business hours",
    })

@tool
def get_policy_details(policy_name: str) -> str:
    """
    Retrieve full section details for a named company policy document (e.g. leave_policy.txt).
    """
    res = retrieve_documents(query=policy_name, k=4)
    if not res.chunks:
        return f"Policy '{policy_name}' not found in active directory."
    return "\n\n".join(c.content for c in res.chunks)

ALL_AGENT_TOOLS = [
    search_company_policy,
    search_hr_escalation_directory,
    create_support_ticket,
    get_policy_details,
]
