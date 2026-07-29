from langgraph.graph import StateGraph
from langgraph.graph import END

from state import ESGState

from nodes import (
    input_request_node,
    document_analyzer_node,
    esg_question_answering_node,
    esg_score_calculation_node,
    recommendation_node,
    verdict_node,
)

builder = StateGraph(ESGState)

builder.add_node("input_request", input_request_node)

builder.add_node("document_analyzer", document_analyzer_node)

builder.add_node("question_answering", esg_question_answering_node)

builder.add_node("score_calculation", esg_score_calculation_node)

builder.add_node("recommendation", recommendation_node)

builder.add_node("verdict", verdict_node)

builder.set_entry_point("input_request")

builder.add_edge("input_request", "document_analyzer")

builder.add_edge("document_analyzer", "question_answering")

builder.add_edge("question_answering", "score_calculation")

builder.add_edge("score_calculation", "recommendation")

builder.add_edge("recommendation", "verdict")

builder.add_edge("verdict", END)

graph = builder.compile()
