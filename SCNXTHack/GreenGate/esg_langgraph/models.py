from typing import List, Optional
from pydantic import BaseModel


class ExtractedFact(BaseModel):
    id: str
    category: str
    topic: str
    evidence: str
    document: str
    page: int
    confidence: float
    keywords: List[str] = []


class QuestionAnswer(BaseModel):
    answer: str
    confidence: float
    source: Optional[str] = None


class CategoryScore(BaseModel):
    category: str
    score: float
    max_score: float


class ESGResult(BaseModel):
    company_name: str
    scores: dict
    recommendation: str
    verdict: str
    reasoning: str
