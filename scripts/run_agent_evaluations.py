import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.agent.graph import run_agent
from src.evaluators.agent_evaluators.evaluators import (
    evaluate_routing,
    evaluate_tool_selection,
    evaluate_tool_execution,
    evaluate_trajectory,
)

def main():
    agent_dataset_path = Path("data/evaluation/agent/agent_dataset.json")
    if not agent_dataset_path.exists():
        print("Agent evaluation dataset not found.")
        return
        
    with agent_dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print("=" * 70)
    print(" Running LangGraph Agent Evaluation Suite ")
    print("=" * 70)
    
    results = []
    
    for case in dataset:
        cid = case["id"]
        q = case["inputs"]["question"]
        print(f"\n[Evaluating {cid}]: '{q}'")
        
        # Execute agent
        agent_state = run_agent(q)
        
        # Evaluate agent dimensions
        r_eval = evaluate_routing(agent_state.get("intent", ""), case.get("expected_intent", ""))
        t_eval = evaluate_tool_selection(agent_state.get("chosen_tool"), case.get("expected_tool"))
        exec_eval = evaluate_tool_execution(
            agent_state.get("chosen_tool"),
            agent_state.get("tool_arguments"),
            agent_state.get("tool_output"),
        )
        traj_eval = evaluate_trajectory(
            agent_state.get("trajectory", []),
            case.get("expected_trajectory_nodes", []),
        )
        
        print(f"  * Routing:        score={r_eval.score} ({r_eval.explanation})")
        print(f"  * Tool Selection: score={t_eval.score} ({t_eval.explanation})")
        print(f"  * Tool Execution: score={exec_eval.score} ({exec_eval.explanation})")
        print(f"  * Trajectory:     score={traj_eval.score} ({traj_eval.explanation})")
        
        results.append({
            "id": cid,
            "question": q,
            "state": {
                "intent": agent_state.get("intent"),
                "chosen_tool": agent_state.get("chosen_tool"),
                "trajectory": agent_state.get("trajectory"),
                "final_response": agent_state.get("final_response"),
            },
            "evaluations": {
                "routing": r_eval.model_dump(),
                "tool_selection": t_eval.model_dump(),
                "tool_execution": exec_eval.model_dump(),
                "trajectory": traj_eval.model_dump(),
            }
        })
        
    out_file = Path("reports/agent_evaluation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"Agent evaluations completed! Results saved to {out_file}")

if __name__ == "__main__":
    main()
