import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)


class DummyLLM:

    def invoke(self, prompt: str):

        print("=" * 60)
        print(prompt)
        print("=" * 60)

        return {
            "response": f"Dummy response for:\n{prompt[:120]}"
        }


class OpenRouterLLM:

    def __init__(self):
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENROUTER_API_KEY. Add it to your environment or .env file.")

        model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.1")
        logger.info("Building OpenRouter LLM with model=%s", model_name)

        self.model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )

    def invoke(self, prompt: str):

        logger.info(
            "LLM request →\n%s\n%s\n%s",
            "-" * 60,
            prompt.strip(),
            "-" * 60,
        )

        result = self.model.invoke(prompt)

        logger.info(
            "LLM response ←\n%s\n%s\n%s",
            "-" * 60,
            result.content.strip(),
            "-" * 60,
        )

        return {
            "response": result.content
        }
