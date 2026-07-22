"""
pipeline.llm — LLM 客户端 (兼容 OpenAI API 协议)
V4.3: 增加 JSON Schema 验证 + retry-with-feedback + response_format 支持
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from openai import OpenAI


class LLMClient:
    """
    LLM 调用客户端，兼容 OpenAI API 协议。
    支持 JSON Schema 验证、retry-with-feedback、DeepSeek thinking mode。
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
        enable_thinking: bool = False,
        reasoning_effort: str = "max",
        use_json_mode: bool = True,
        json_retries: int = 2,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.use_json_mode = use_json_mode
        self.json_retries = json_retries

        _api_key = api_key or ""
        _base_url = base_url or "https://api.deepseek.com"

        self._client = OpenAI(
            api_key=_api_key,
            base_url=_base_url,
            timeout=timeout,
        )

    def _call(self, prompt: str, use_json: bool = False) -> str:
        """单轮对话调用，含自动重试。支持 DeepSeek thinking mode 和 JSON mode。"""
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout,
                }

                # JSON mode: force structured output (DeepSeek/OpenAI compatible)
                if use_json and self.use_json_mode and not self.enable_thinking:
                    kwargs["response_format"] = {"type": "json_object"}

                # DeepSeek thinking mode
                if self.enable_thinking:
                    kwargs["extra_body"] = {
                        "thinking": {"type": "enabled"},
                    }
                    kwargs["reasoning_effort"] = self.reasoning_effort
                else:
                    kwargs["temperature"] = self.temperature

                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                # For thinking mode, fall back to reasoning_content if needed
                if not content:
                    reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
                    if reasoning:
                        # Extract final answer from reasoning (thinking mode may put answer after reasoning)
                        content = reasoning

                return content or ""

            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait = min(2 ** attempt, 30)
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM 调用失败 (重试 {self.max_retries} 次): {last_error}"
        )

    def call_json(self, prompt: str, default: object = None, schema_hint: str = "") -> object:
        """
        调用 LLM 并进行 JSON 解析，含多级容错 + retry-with-feedback。

        schema_hint: 可选的 schema 描述，用于 retry 时告诉模型期望的格式。
        """
        if default is None:
            default = []

        # First attempt: try with JSON mode
        raw = self._call(prompt, use_json=True)
        result = self._safe_json_parse(raw, None)

        if result is not None:
            return result

        # Retry with feedback: tell the model what went wrong
        for retry in range(self.json_retries):
            if schema_hint:
                retry_prompt = (
                    f"Your previous output was NOT valid JSON. Please output STRICTLY valid JSON.\n"
                    f"Expected format: {schema_hint}\n"
                    f"DO NOT include markdown code blocks, explanations, or extra text.\n"
                    f"Original task:\n{prompt}"
                )
            else:
                retry_prompt = (
                    f"Your previous output was NOT valid JSON. Please output STRICTLY valid JSON.\n"
                    f"DO NOT include markdown code blocks, explanations, or extra text.\n\n"
                    f"Original task:\n{prompt}"
                )

            raw = self._call(retry_prompt, use_json=True)
            result = self._safe_json_parse(raw, None)
            if result is not None:
                return result

        # Final fallback: try without JSON mode (some models handle this better)
        raw = self._call(prompt, use_json=False)
        result = self._safe_json_parse(raw, None)
        if result is not None:
            return result

        return default

    def call_json_voting(
        self,
        prompt: str,
        default: object = None,
        rounds: int = 3,
        voting_temperature: float = 0.3,
        schema_hint: str = "",
    ) -> tuple[object, list[object], int]:
        """
        Self-consistency voting: 调用 LLM 多次, 对结果进行多数投票。

        返回: (winning_result, all_results, agreement_count)
        - winning_result: 得票最多的结果
        - all_results: 所有轮次的结果列表
        - agreement_count: 最高票数
        """
        if default is None:
            default = []

        all_results: list[object] = []
        original_temp = self.temperature
        original_thinking = self.enable_thinking

        # 投票需要温度多样性 — 暂时禁用 thinking mode
        self.enable_thinking = False
        self.temperature = voting_temperature

        try:
            for r in range(rounds):
                # 每轮微调温度增加多样性
                self.temperature = voting_temperature + r * 0.05
                result = self.call_json(prompt, default, schema_hint)
                all_results.append(result)
        finally:
            self.temperature = original_temp
            self.enable_thinking = original_thinking

        # Vote on string representations (sorted for consistency)
        def _key(r: object) -> str:
            return json.dumps(r, ensure_ascii=False, sort_keys=True)

        votes: dict[str, tuple[int, object]] = {}
        for r in all_results:
            k = _key(r)
            if k in votes:
                votes[k] = (votes[k][0] + 1, r)
            else:
                votes[k] = (1, r)

        best = max(votes.values(), key=lambda x: x[0])
        return best[1], all_results, best[0]
        """
        调用 LLM 并进行 JSON 解析，含多级容错 + retry-with-feedback。

        schema_hint: 可选的 schema 描述，用于 retry 时告诉模型期望的格式。
        """
        if default is None:
            default = []

        # First attempt: try with JSON mode
        raw = self._call(prompt, use_json=True)
        result = self._safe_json_parse(raw, None)

        if result is not None:
            return result

        # Retry with feedback: tell the model what went wrong
        for retry in range(self.json_retries):
            if schema_hint:
                retry_prompt = (
                    f"Your previous output was NOT valid JSON. Please output STRICTLY valid JSON.\n"
                    f"Expected format: {schema_hint}\n"
                    f"DO NOT include markdown code blocks, explanations, or extra text.\n"
                    f"Original task:\n{prompt}"
                )
            else:
                retry_prompt = (
                    f"Your previous output was NOT valid JSON. Please output STRICTLY valid JSON.\n"
                    f"DO NOT include markdown code blocks, explanations, or extra text.\n\n"
                    f"Original task:\n{prompt}"
                )

            raw = self._call(retry_prompt, use_json=True)
            result = self._safe_json_parse(raw, None)
            if result is not None:
                return result

        # Final fallback: try without JSON mode (some models handle this better)
        raw = self._call(prompt, use_json=False)
        result = self._safe_json_parse(raw, None)
        if result is not None:
            return result

        return default

    @staticmethod
    def _safe_json_parse(raw: str, default: object = None) -> object:
        """
        多级 JSON 解析容错:
        1. 提取 markdown 代码块
        2. 修正常见JSON错误
        3. 直接解析
        4. 正则提取JSON片段
        """
        if not raw or not raw.strip():
            return default
        text = raw.strip()

        # 1) 提取 markdown 代码块
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if code_block:
            text = code_block.group(1).strip()

        # 2) 修正常见 JSON 错误
        # 去除尾逗号
        text = re.sub(r",\s*(\]|\})", r"\1", text)
        # 修复单引号
        if text.count("'") > text.count('"') * 2:
            # Only fix if there are significantly more single quotes
            pass  # skip — risky to blindly replace

        # 3) 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 4) 解析前先找到第一个 [ 或 { 的位置
        json_match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        return default


def build_llm_client(
    model: str = "deepseek-chat",
    api_key: str = "",
    base_url: str = "https://api.deepseek.com",
) -> LLMClient:
    """工厂函数：创建 LLMClient 实例"""
    return LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
