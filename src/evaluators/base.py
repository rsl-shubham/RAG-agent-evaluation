from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class EvaluationResult(BaseModel):
    evaluator_name: str
    score: Optional[float] = Field(default=None, description="Score between 0.0 and 1.0 or custom scale; None if not applicable")
    passed: Optional[bool] = Field(default=None, description="Pass/fail status where applicable")
    explanation: str = Field(description="Explanation of the assigned score and verdict")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic data, breakdown of claims, or intermediate values")
