from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from spy_game.config import LLMConfig


def load_llm(overrides: LLMConfig | None = None) -> tuple[ChatOpenAI, StrOutputParser]:
    _pkg = Path(__file__).resolve().parents[1]
    load_dotenv(_pkg / ".env")
    load_dotenv(_pkg.parent / ".env")
    o = overrides or LLMConfig()
    api_key = o.api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = o.base_url or os.getenv("OPENAI_API_BASE", "") or None
    model = o.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    kwargs: dict = {
        "api_key": api_key,
        "model": model,
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", o.temperature)),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", o.max_tokens)),
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs), StrOutputParser()
