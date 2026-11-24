import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.graph import run_agent

def main():
    print("=" * 60)
    print(" Acme Technologies - Autonomous LangGraph HR Agent ")
    print("=" * 60)
    print("Type your request or escalation below. Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("Employee: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Agent session ended. Goodbye!")
                break
                
            state = run_agent(query)
            
            print("\n[Agent Decision Trajectory]")
            print(" -> ".join(state.get("trajectory", [])))
            
            if state.get("chosen_tool"):
                print(f"[Tool Selected]: {state['chosen_tool']}")
            if state.get("retrieved_sources"):
                print(f"[Sources]: {state['retrieved_sources']}")
                
            print("\nAgent Response:")
            print(state["final_response"])
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error executing agent: {e}")

if __name__ == "__main__":
    main()
