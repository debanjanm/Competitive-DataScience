import json
import re
import zipfile
from pathlib import Path

from llm import OpenRouterLLM

CONFIG = Path(__file__).parent / "configs"
DATA = Path(__file__).parent.parent / "data"


def load_llm():

    return OpenRouterLLM()


def load_prompts():

    with open(CONFIG / "prompts.json") as f:
        return json.load(f)


def load_contacts():

    with open(CONFIG / "contacts.json") as f:
        return json.load(f)


def load_rules():

    with open(CONFIG / "scoring_rules.json") as f:
        return json.load(f)


def load_questionnaire():

    with open(CONFIG / "esg_questionnaire.json") as f:
        return json.load(f)["questions"]


def load_companies():

    with open(CONFIG / "companies.json") as f:
        return json.load(f)


def extract_docx_text(path: Path) -> str:

    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")

    xml = xml.replace("</w:p>", "\n")
    text = re.sub("<[^>]+>", "", xml)

    return text.strip()


def load_company_documents(company_name: str):

    companies = load_companies()
    company = companies[company_name]

    docs = []

    for doc in company["documents"]:
        path = DATA / doc["filename"]
        docs.append({
            "type": doc["type"],
            "filename": doc["filename"],
            "text": extract_docx_text(path),
        })

    return company, docs


def parse_json_response(text: str):

    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    return json.loads(cleaned)
