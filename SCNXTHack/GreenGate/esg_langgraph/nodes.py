import json
import logging

from tools import (
    load_prompts,
    load_rules,
    load_questionnaire,
    load_company_documents,
    parse_json_response,
    load_llm,
)


logger = logging.getLogger(__name__)

PROMPTS = load_prompts()
RULES = load_rules()
QUESTIONNAIRE = load_questionnaire()
llm = load_llm()


def input_request_node(state):

    logger.info("[input_request] company=%s", state["company_name"])

    prompt = PROMPTS["input_request"]["system"]

    result = llm.invoke(prompt)

    company, documents = load_company_documents(state["company_name"])

    state["metadata"] = {
        "intake": result["response"],
        "industry": company["industry"],
        "country": company["country"],
        "reporting_year": company["reporting_year"],
    }

    state["documents"] = [doc["filename"] for doc in documents]
    state["document_texts"] = documents

    logger.info("[input_request] loaded %d document(s):\n%s", len(documents), json.dumps(state["documents"], indent=2))

    return state


def document_analyzer_node(state):

    documents = state["document_texts"]

    logger.info("[document_analyzer] analyzing %d document(s)", len(documents))

    doc_text = "\n\n".join(
        f"### {doc['filename']} ({doc['type']})\n{doc['text']}" for doc in documents
    )

    prompt = f"{PROMPTS['document_analyzer']['system']}\n\nDocuments:\n{doc_text}"

    result = llm.invoke(prompt)

    facts = parse_json_response(result["response"])

    state["extracted_context"] = facts

    logger.info(
        "[document_analyzer] extracted %d fact(s):\n%s",
        len(facts.get("facts", [])),
        json.dumps(facts, indent=2),
    )

    return state


def esg_question_answering_node(state):

    logger.info("[question_answering] answering %d question(s)", len(QUESTIONNAIRE))

    prompt = (
        f"{PROMPTS['esg_question_answering']['system']}\n\n"
        f"Question Bank:\n{json.dumps(QUESTIONNAIRE, indent=2)}\n\n"
        f"Extracted Evidence:\n{json.dumps(state['extracted_context'], indent=2)}"
    )

    result = llm.invoke(prompt)

    assessment = parse_json_response(result["response"])

    state["answers"] = {item["question_id"]: item for item in assessment}

    logger.info("[question_answering] answers:\n%s", json.dumps(state["answers"], indent=2))

    return state


def esg_score_calculation_node(state):

    logger.info("[score_calculation] computing weighted score")

    prompt = PROMPTS["score_calculation"]["system"]

    llm.invoke(prompt)

    status_points = RULES["status_points"]

    category_contribution = {}
    category_weight = {}
    total_contribution = 0
    total_weight = 0

    for question in QUESTIONNAIRE:

        answer = state["answers"].get(question["id"])

        if not answer:
            continue

        points = status_points.get(answer["status"])

        if points is None:
            continue

        weight = question["weight"]
        category = question["category"]

        contribution = weight * (points / 100)

        category_contribution[category] = category_contribution.get(category, 0) + contribution
        category_weight[category] = category_weight.get(category, 0) + weight

        total_contribution += contribution
        total_weight += weight

    categories = {
        category: round((category_contribution[category] / category_weight[category]) * 100, 1)
        for category in category_contribution
    }

    overall = round((total_contribution / total_weight) * 100, 1) if total_weight else 0

    state["scores"] = {
        "categories": categories,
        "overall": overall,
        "coverage": round((total_weight / sum(q["weight"] for q in QUESTIONNAIRE)) * 100, 1),
    }

    logger.info("[score_calculation] scores:\n%s", json.dumps(state["scores"], indent=2))

    return state


def recommendation_node(state):

    logger.info("[recommendation] generating recommendation")

    prompt = (
        f"{PROMPTS['recommendation']['system']}\n\n"
        f"ESG Assessment:\n{json.dumps(state['answers'], indent=2)}\n\n"
        f"ESG Score:\n{json.dumps(state['scores'], indent=2)}\n\n"
        f"Company Information:\n{json.dumps(state['metadata'], indent=2)}"
    )

    result = llm.invoke(prompt)

    state["recommendation"] = parse_json_response(result["response"])

    logger.info("[recommendation] result:\n%s", json.dumps(state["recommendation"], indent=2))

    return state


def verdict_node(state):

    overall = state["scores"]["overall"]

    band = next(
        b["label"] for b in RULES["recommendation_bands"] if b["min"] <= overall <= b["max"]
    )

    logger.info("[verdict] overall=%s band=%s", overall, band)

    prompt = (
        f"{PROMPTS['verdict']['system']}\n\n"
        f"Overall Score: {overall}\n"
        f"Band: {band}\n"
        f"Recommendation:\n{json.dumps(state['recommendation'], indent=2)}"
    )

    result = llm.invoke(prompt)

    state["verdict"] = band
    state["reasoning"] = result["response"]

    logger.info("[verdict] reasoning:\n%s", state["reasoning"])

    return state
