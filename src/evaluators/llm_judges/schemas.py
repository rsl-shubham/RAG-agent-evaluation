from typing import Literal, List, Optional
from pydantic import BaseModel, Field

# 1. Answer Relevance Schema
class AnswerRelevanceResult(BaseModel):
    score: int = Field(ge=1, le=5, description="Relevance score between 1 and 5 according to rubric")
    verdict: Literal["relevant", "partially_relevant", "irrelevant"] = Field(description="Categorical judgment")
    reason: str = Field(description="Explanation justifying the score")

# 2. Faithfulness Schema (Claim-level)
class ClaimAssessment(BaseModel):
    claim: str = Field(description="Atomic factual claim extracted from the generated answer")
    verdict: Literal["supported", "contradicted", "not_enough_information"] = Field(
        description="Verification against the provided context"
    )
    evidence: str = Field(default="", description="Exact supporting or contradicting quote from the context, or empty")
    reason: str = Field(description="Short rationale for the verdict")

class FaithfulnessResult(BaseModel):
    claims: List[ClaimAssessment] = Field(description="Extracted claims and their verdicts")
    overall_reason: str = Field(description="Summary explanation of the faithfulness assessment")

# 3. Answer Correctness Schema (Reference-based)
class AnswerCorrectnessResult(BaseModel):
    score: int = Field(ge=1, le=5, description="Correctness score between 1 and 5")
    verdict: Literal["correct", "partially_correct", "incorrect"] = Field(description="Correctness label")
    missing_facts: List[str] = Field(default_factory=list, description="Reference facts missing from generated answer")
    incorrect_claims: List[str] = Field(default_factory=list, description="Claims in generated answer that contradict the reference")
    extra_unverified_claims: List[str] = Field(default_factory=list, description="Claims in answer not mentionable in reference")
    reason: str = Field(description="Detailed reason for the score")

# 4. Context Precision Schema (Chunk-level)
class ChunkRelevanceAssessment(BaseModel):
    chunk_index: int = Field(description="0-indexed position of the retrieved chunk")
    is_relevant: bool = Field(description="True if this chunk contains facts useful for answering the user's question, False if irrelevant noise")
    evidence: str = Field(default="", description="Snippet from the chunk demonstrating relevance")
    reason: str = Field(description="Brief explanation of why the chunk is relevant or irrelevant")

class ContextPrecisionResult(BaseModel):
    assessments: List[ChunkRelevanceAssessment] = Field(description="Relevance classification for every retrieved chunk")
    summary_reason: str = Field(description="Overall context precision summary")

# 5. Context Recall Schema (Atomic-fact level)
class FactSupportAssessment(BaseModel):
    atomic_fact: str = Field(description="One required atomic fact from the reference answer")
    is_supported: bool = Field(description="True if the retrieved context contains evidence for this fact, False otherwise")
    evidence: str = Field(default="", description="Snippet from retrieved context supporting this fact, if found")
    reason: str = Field(description="Explanation of why this fact is or is not supported by context")

class ContextRecallResult(BaseModel):
    atomic_facts: List[FactSupportAssessment] = Field(description="Assessments for all reference atomic facts")
    summary_reason: str = Field(description="Summary of context recall findings")

# 6. Hallucination / Unsupported Claim Schema
class HallucinationResult(BaseModel):
    has_hallucinations: bool = Field(description="True if any unverified or fabricated claims were made, False otherwise")
    hallucinated_claims: List[str] = Field(default_factory=list, description="List of unsupported or invented statements")
    severity: Literal["none", "minor", "severe"] = Field(description="Impact of hallucinated content")
    reason: str = Field(description="Analysis of potential hallucinations")
