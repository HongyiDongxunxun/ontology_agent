"""
pipeline.llm — LLM 客户端 (兼容 DeepSeek-V4 / OpenAI API)
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from openai import OpenAI


class LLMClient:
    """
    LLM 调用客户端，兼容 OpenAI API 协议。
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 60,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    def _call(self, prompt: str) -> str:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""

    def call_json(self, prompt: str, default: object = None) -> object:
        raw = self._call(prompt)
        return self._safe_json_parse(raw, default)

    @staticmethod
    def _safe_json_parse(raw: str, default: object = None) -> object:
        if default is None:
            default = []
        text = raw.strip()
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if code_block:
            text = code_block.group(1).strip()
        text = re.sub(r",\s*(\]|\})", r"\1", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            json_match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            return default
        return default


def build_llm_client() -> LLMClient:
    return LLMClient(
        model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
