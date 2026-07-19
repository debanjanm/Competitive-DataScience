import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from graph import graph


state = {

    "company_name": "ABC Green Infrastructure Ltd.",

    "documents": [],

    "document_texts": [],

    "extracted_context": {},

    "answers": {},

    "scores": {},

    "recommendation": {},

    "verdict": "",

    "reasoning": "",

    "metadata": {}
}


result = graph.invoke(state)

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(json.dumps(result, indent=2, default=str))
