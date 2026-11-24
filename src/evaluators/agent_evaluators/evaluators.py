from typing import Dict, Any, List, Optional
from src.evaluators.base import EvaluationResult
from src.evaluators.llm_judges.judges import evaluate_answer_correctness, evaluate_faithfulness

def evaluate_routing(
    actual_intent: str,
    expected_intent: str,
) -> EvaluationResult:
    """Evaluate whether the agent classified and routed the query appropriately."""
    passed = (actual_intent == expected_intent)
    score = 1.0 if passed else 0.0
    return EvaluationResult(
        evaluator_name="agent_routing_evaluation",
        score=score,
        passed=passed,
        explanation=f"Routed to '{actual_intent}' (expected: '{expected_intent}').",
        details={"actual_intent": actual_intent, "expected_intent": expected_intent},
    )

def evaluate_tool_selection(
    chosen_tool: Optional[str],
    expected_tool: Optional[str],
) -> EvaluationResult:
    """Evaluate whether the agent invoked the expected tool without unnecessary calls."""
    passed = (chosen_tool == expected_tool)
    score = 1.0 if passed else 0.0
    return EvaluationResult(
        evaluator_name="agent_tool_selection_evaluation",
        score=score,
        passed=passed,
        explanation=f"Selected tool '{chosen_tool}' (expected: '{expected_tool}').",
        details={"chosen_tool": chosen_tool, "expected_tool": expected_tool},
    )

def evaluate_tool_execution(
    tool_name: Optional[str],
    tool_arguments: Optional[Dict[str, Any]],
    tool_output: Optional[str],
) -> EvaluationResult:
    """Validate that the tool received valid non-empty arguments and returned an output."""
    if not tool_name:
        return EvaluationResult(
            evaluator_name="agent_tool_execution_evaluation",
            score=None,
            passed=None,
            explanation="No tool was executed for this request.",
            details={},
        )
        
    has_args = bool(tool_arguments)
    has_output = bool(tool_output and len(tool_output.strip()) > 0)
    passed = has_args and has_output
    score = 1.0 if passed else 0.0
    
    return EvaluationResult(
        evaluator_name="agent_tool_execution_evaluation",
        score=score,
        passed=passed,
        explanation="Tool executed with valid arguments and returned response." if passed else "Tool arguments or output were missing or failed.",
        details={"tool_name": tool_name, "has_arguments": has_args, "has_output": has_output},
    )

def evaluate_trajectory(
    actual_trajectory: List[str],
    expected_nodes: List[str],
) -> EvaluationResult:
    """Evaluate observable graph state transitions and nodes visited."""
    if not expected_nodes:
        return EvaluationResult(
            evaluator_name="agent_trajectory_evaluation",
            score=1.0,
            passed=True,
            explanation="No specific trajectory sequence required.",
            details={"actual_trajectory": actual_trajectory},
        )
        
    # Check if all expected nodes appear in the trajectory
    present_nodes = [node for node in expected_nodes if any(node in step for step in actual_trajectory)]
    score = len(present_nodes) / len(expected_nodes)
    passed = (len(present_nodes) == len(expected_nodes))
    
    return EvaluationResult(
        evaluator_name="agent_trajectory_evaluation",
        score=round(score, 4),
        passed=passed,
        explanation=f"Agent visited {len(present_nodes)}/{len(expected_nodes)} expected milestone nodes.",
        details={
            "actual_trajectory": actual_trajectory,
            "expected_milestones": expected_nodes,
            "missing_milestones": [n for n in expected_nodes if n not in present_nodes],
        },
    )
