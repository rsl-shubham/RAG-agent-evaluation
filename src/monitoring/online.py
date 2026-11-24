import random
import uuid
from typing import Dict, Any, Optional
from src.langsmith.datasets import get_langsmith_client

class OnlineMonitor:
    def __init__(self, sample_rate: float = 0.2):
        """
        sample_rate: fraction of production queries sampled for detailed LLM judge inspection (e.g. 20%).
        """
        self.sample_rate = sample_rate
        self.client = None

    def should_sample(self) -> bool:
        """Probabilistically decide if trace should undergo expensive LLM judge evaluation."""
        return random.random() < self.sample_rate

    def record_user_feedback(
        self,
        run_id: str,
        score: int,  # 1 for thumbs up, 0 for thumbs down
        comment: Optional[str] = None,
    ):
        """Log user thumbs-up/down feedback directly to LangSmith trace."""
        try:
            if not self.client:
                self.client = get_langsmith_client()
            self.client.create_feedback(
                run_id=run_id,
                key="user_feedback",
                score=score,
                comment=comment,
            )
            print(f"Logged user feedback (score={score}) for run {run_id}")
        except Exception as e:
            print(f"Failed to record feedback to LangSmith: {e}")
