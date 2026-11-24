import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.pipeline import answer_question

def main():
    print("=" * 60)
    print(" Acme Technologies - Employee Knowledge & HR Assistant ")
    print("=" * 60)
    print("Type your policy question below. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Employee: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Thank you for using Acme Knowledge Assistant. Goodbye!")
                break
                
            res = answer_question(query)
            print("\nAssistant:")
            print(res["answer"])
            
            if res.get("sources"):
                print("\nSources Referenced:")
                for s in res["sources"]:
                    ver = res.get("document_versions", {}).get(s, "1.0")
                    print(f"  - {s} (v{ver})")
            print(f"[Latency: {res.get('latency_ms', 0):.1f}ms]\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()
