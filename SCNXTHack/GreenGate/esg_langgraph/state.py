from typing import TypedDict, List, Dict, Any


class ESGState(TypedDict):
    company_name: str

    documents: List[str]

    document_texts: List[Dict[str, Any]]

    extracted_context: Dict[str, Any]

    answers: Dict[str, Any]

    scores: Dict[str, Any]

    recommendation: Dict[str, Any]

    verdict: str

    reasoning: str

    metadata: Dict[str, Any]
