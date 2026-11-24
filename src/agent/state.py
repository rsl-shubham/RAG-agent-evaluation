from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Core conversational input & history
    question: str
    messages: List[BaseMessage]
    
    # Execution trajectory tracking
    intent: Optional[str]
    chosen_tool: Optional[str]
    tool_arguments: Optional[Dict[str, Any]]
    tool_output: Optional[str]
    
    # Retrieval assessment
    retrieved_context: Optional[str]
    retrieved_sources: List[str]
    is_context_sufficient: bool
    
    # Escalation & Final Output
    requires_escalation: bool
    escalation_details: Optional[Dict[str, Any]]
    final_response: str
    trajectory: List[str]
