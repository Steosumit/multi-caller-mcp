"""ChatOpenAI client pointed at a local OpenAI-compatible server (LM Studio)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"


def get_llm(
    *,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI client from environment variables.

    LLM_BASE_URL   OpenAI-compatible endpoint (default: LM Studio)
    LLM_MODEL      model id served by the endpoint
    LLM_API_KEY    API key; LM Studio ignores it but the SDK requires one
    LLM_TEMPERATURE
    """
    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", DEFAULT_MODEL),
        base_url=base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("LLM_API_KEY", "lm-studio"),
        temperature=(
            float(os.getenv("LLM_TEMPERATURE", "0"))
            if temperature is None
            else temperature
        ),
        timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
    )


def list_models(base_url: str | None = None) -> list[str]:
    """Return the model ids advertised by the server."""
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("LLM_API_KEY", "lm-studio"),
        timeout=10,
    )
    return [model.id for model in client.models.list().data]


if __name__ == "__main__":
    url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    print(f"Server: {url}")
    try:
        for model_id in list_models():
            print(f"  - {model_id}")
    except Exception as exc:  # noqa: BLE001 - CLI reporting
        raise SystemExit(
            f"Could not reach {url}: {exc}\n"
            "Start LM Studio and enable the local server (Developer tab)."
        ) from exc
