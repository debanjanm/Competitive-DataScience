"""
Pointwise LLM reranker for RAG, based on Fin AI's approach:
https://fin.ai/research/using-llms-as-a-reranker-for-rag-a-practical-guide/

Pipeline:
  1. Vector search returns top-K passages.
  2. Passages are split into N batches via round-robin (spreads positional
     bias evenly across batches instead of clumping it in one).
  3. Each batch is scored by an LLM in one call -> compact JSON {id: score}.
  4. Scores <5 are dropped (relevance floor) to cut output tokens/latency.
  5. Batches are merged and sorted; ties broken with a cheap fallback score.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


@dataclass
class Passage:
    id: str
    text: str


LLMCallFn = Callable[[str], str]  # prompt -> raw model output

CUSTOMER_SUPPORT_PROMPT = """You are a customer support answer service. Your task is to evaluate help center passages and score their relevance to a given customer query for a retrieval augmented generation (RAG) system.

Evaluation Process:
1. Analyze the customer's query to identify both explicit needs and implicit context including underlying user goals
2. Assess each passage's ability to directly resolve the query or provide substantive supporting information with actionable guidance
3. Score based on how effectively the passage addresses the query's core intent while considering potential interpretations

Grading Criteria:
<grading_scale>
10: EXCEPTIONAL match - Contains exact step-by-step instructions that perfectly match the query's specific scenario. Must include all required parameters/context and resolve the issue completely without any ambiguity. Reserved for definitive solutions that exactly mirror the user's described situation and require no interpretation.

9: NEAR-PERFECT solution - Contains all critical steps for resolution but may lack one minor non-essential detail. Addresses the precise query parameters with specialized information. Solution must be directly applicable without requiring adaptation or assumptions.

8: STRONG MATCH - Provides complete technical resolution through specific instructions, but may require simple logical inferences for full application. Covers all essential components but might need minor contextualization.

7: GOOD MATCH - Contains substantial relevant details that address core aspects of the query, but lacks one important element for complete resolution. Provides concrete guidance requiring some user interpretation.


6: PARTIAL match – General guidance on the right topic but lacks the specifics for direct application. May only resolve a subset of the request.


5: LIMITED relevance – Related context or approach, but indirect. Requires substantial effort to adapt to the user's exact need.


4: TANGENTIAL – Mentions related concepts/keywords with little practical connection to the request. Minimal actionable value.


3: VAGUE domain info – Talks about the general area but not the query's specifics. No concrete, actionable steps.


2: TOKEN overlap – Shares isolated terms without context or intent aligned to the request. Similarity is coincidental.


1: IRRELEVANT – Uses query terms in a completely unrelated way. No meaningful link to the user's goal.


0: UNRELATED – No thematic or contextual connection to the query at all.
</grading_scale>

Input Format:
<input_format>
<query>
// The customer's question or request
</query>
<passages>
<passage id='id0'>...</passage>
<passage id='id1'>...</passage>
...
</passages>
</input_format>

Output Format:
<output_format>
Return your response in a valid JSON (skip spaces):
{{"id0":score0,"id1":score1,...}}

Strict guidelines:
- Return ONLY a well-formed valid JSON with passage IDs as keys
- Each key must be a passage id in the format "idN"
- Each score must be an integer between 5 to 10. EXCLUDE passages that score below 5 (i.e. 0, 1, 2, 3 or 4)
- Integer values only, no decimals
- Skip spaces in the JSON
- No additional text or formatting
- Maintain original passage ID order
- Note: If NO passages score 5+, return empty JSON object
</output_format>

<query>
{query}
</query>
<passages>
{passages_block}
</passages>"""


def _round_robin_batches(passages: list[Passage], n_batches: int) -> list[list[Passage]]:
    batches: list[list[Passage]] = [[] for _ in range(n_batches)]
    for i, p in enumerate(passages):
        batches[i % n_batches].append(p)
    return batches


def _build_prompt(query: str, batch: list[Passage]) -> str:
    passages_block = "\n".join(f"<passage id='{p.id}'>{p.text}</passage>" for p in batch)
    return CUSTOMER_SUPPORT_PROMPT.format(query=query, passages_block=passages_block)


def _parse_scores(raw_output: str) -> dict[str, int]:
    # models sometimes wrap JSON in code fences or add stray text - extract the {...}
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {str(k): int(v) for k, v in parsed.items()}


def _tie_break_score(passage: Passage, query: str) -> float:
    """Cheap fallback for equal LLM scores: lexical overlap. Swap in a
    cross-encoder (e.g. BGE) here for a stronger tie-break, as Fin AI does."""
    q_terms = set(query.lower().split())
    p_terms = set(passage.text.lower().split())
    if not q_terms:
        return 0.0
    return len(q_terms & p_terms) / len(q_terms)


def rerank(
    query: str,
    passages: list[Passage],
    llm_call: LLMCallFn,
    n_batches: int = 4,
    score_floor: int = 5,
) -> list[tuple[Passage, int]]:
    """Returns passages sorted by relevance score, descending.

    Dropped (sub-floor) passages are excluded from the result entirely.
    """
    batches = [b for b in _round_robin_batches(passages, n_batches) if b]
    by_id = {p.id: p for p in passages}

    def score_batch(batch: list[Passage]) -> dict[str, int]:
        return _parse_scores(llm_call(_build_prompt(query, batch)))

    scored: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        for result in pool.map(score_batch, batches):
            scored.update(result)

    results = [
        (by_id[pid], score)
        for pid, score in scored.items()
        if pid in by_id and score >= score_floor
    ]
    results.sort(key=lambda r: (r[1], _tie_break_score(r[0], query)), reverse=True)
    return results


if __name__ == "__main__":
    # mock LLM: pretend-scores passages containing "refund" highly
    def mock_llm(prompt: str) -> str:
        scores = {}
        for pid, text in re.findall(r"<passage id='(\w+)'>(.*?)</passage>", prompt):
            scores[pid] = 9 if "refund" in text.lower() else 4
        return json.dumps(scores)

    demo_passages = [
        Passage("p0", "To request a refund, go to Settings > Billing and click Refund."),
        Passage("p1", "Our office hours are 9am to 5pm EST, Monday through Friday."),
        Passage("p2", "Refunds are processed within 5-7 business days."),
        Passage("p3", "You can change your password from the account page."),
    ]

    ranked = rerank("how do I get a refund", demo_passages, mock_llm, n_batches=2)
    for passage, score in ranked:
        print(f"{score:>2}  {passage.id}  {passage.text}")
