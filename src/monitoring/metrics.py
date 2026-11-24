from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class TokenCostConfig(BaseModel):
    # Gemini 3.5 Flash / Flash Lite typical pricing per million tokens
    input_cost_per_million: float = 0.075   # $0.075 per 1M prompt tokens
    output_cost_per_million: float = 0.30   # $0.30 per 1M completion tokens
    embedding_cost_per_million: float = 0.02 # $0.02 per 1M embedding tokens

def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    embedding_tokens: int = 0,
    config: Optional[TokenCostConfig] = None,
) -> Dict[str, float]:
    """Calculate cost breakdown based on token usage."""
    cfg = config or TokenCostConfig()
    prompt_cost = (prompt_tokens / 1_000_000) * cfg.input_cost_per_million
    completion_cost = (completion_tokens / 1_000_000) * cfg.output_cost_per_million
    embedding_cost = (embedding_tokens / 1_000_000) * cfg.embedding_cost_per_million
    total_cost = prompt_cost + completion_cost + embedding_cost
    
    return {
        "prompt_cost_usd": round(prompt_cost, 6),
        "completion_cost_usd": round(completion_cost, 6),
        "embedding_cost_usd": round(embedding_cost, 6),
        "total_cost_usd": round(total_cost, 6),
    }

def aggregate_evaluations(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metric scores, pass rates, and failure categories."""
    if not evaluations:
        return {}
        
    metric_scores: Dict[str, List[float]] = {}
    pass_counts: Dict[str, int] = {}
    total_counts: Dict[str, int] = {}
    failures: List[Dict[str, Any]] = []
    
    for item in evaluations:
        test_id = item.get("id", "unknown")
        for key, res in item.get("evaluations", {}).items():
            score = res.get("score")
            passed = res.get("passed")
            
            if score is not None:
                metric_scores.setdefault(key, []).append(score)
            if passed is not None:
                total_counts[key] = total_counts.get(key, 0) + 1
                if passed:
                    pass_counts[key] = pass_counts.get(key, 0) + 1
                else:
                    failures.append({
                        "test_id": test_id,
                        "metric": key,
                        "explanation": res.get("explanation", ""),
                    })
                    
    summary = {}
    for key, scores in metric_scores.items():
        summary[key] = {
            "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
            "count": len(scores),
            "pass_rate": round(pass_counts.get(key, 0) / total_counts[key], 4) if total_counts.get(key, 0) > 0 else None,
        }
        
    return {
        "metrics_summary": summary,
        "total_failures": len(failures),
        "failures": failures,
    }
