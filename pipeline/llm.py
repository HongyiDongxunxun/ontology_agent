"""
pipeline.llm — LLM 客户端 (兼容 DeepSeek / OpenAI API)
V4.3: + thinking mode + call_json_voting + schema_hint retry
"""
from __future__ import annotations
import json, os, re, time
from typing import Optional
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

class LLMClient:
    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, temperature: float = 0.0,
                 max_tokens: int = 2048, timeout: int = 60,
                 enable_thinking: bool = False, reasoning_effort: str = "max",
                 json_retries: int = 2):
        self.model = model; self.temperature = temperature
        self.max_tokens = max_tokens; self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.json_retries = json_retries
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    def _call(self, prompt: str, use_json: bool = False) -> str:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)
        retryable = (APIConnectionError, APITimeoutError, RateLimitError)
        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": self.max_tokens, "timeout": self.timeout}
                if use_json and not self.enable_thinking:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.enable_thinking:
                    kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                    kwargs["reasoning_effort"] = self.reasoning_effort
                else:
                    kwargs["temperature"] = self.temperature
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content and self.enable_thinking:
                    reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
                    if reasoning: content = reasoning
                return content or ""
            except retryable as exc:
                last_error = exc
                if attempt == 4: break
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"LLM failed after retries: {type(last_error).__name__}")

    def call_json(self, prompt: str, default: object = None, schema_hint: str = "") -> object:
        if default is None: default = []
        raw = self._call(prompt, use_json=True)
        result = self._safe_json_parse(raw, None)
        if result is not None: return result
        for _ in range(self.json_retries):
            retry_prompt = (f"Previous output was NOT valid JSON. Output STRICTLY valid JSON.\n"
                            f"Expected: {schema_hint}\nOriginal task:\n{prompt}" if schema_hint else
                            f"Previous output was NOT valid JSON. Output STRICTLY valid JSON.\n\nOriginal task:\n{prompt}")
            raw = self._call(retry_prompt, use_json=True)
            result = self._safe_json_parse(raw, None)
            if result is not None: return result
        raw = self._call(prompt, use_json=False)
        result = self._safe_json_parse(raw, None)
        return result if result is not None else default

    def call_json_voting(self, prompt: str, default: object = None, rounds: int = 3,
                         voting_temperature: float = 0.3, schema_hint: str = ""
                         ) -> tuple[object, list[object], int]:
        if default is None: default = []
        all_results, orig_temp, orig_thinking = [], self.temperature, self.enable_thinking
        self.enable_thinking = False; self.temperature = voting_temperature
        try:
            for r in range(rounds):
                self.temperature = voting_temperature + r * 0.05
                result = self.call_json(prompt, default, schema_hint)
                all_results.append(result)
        finally:
            self.temperature = orig_temp; self.enable_thinking = orig_thinking
        def _key(r): return json.dumps(r, ensure_ascii=False, sort_keys=True)
        votes: dict[str, tuple[int, object]] = {}
        for r in all_results:
            k = _key(r)
            votes[k] = (votes[k][0] + 1, r) if k in votes else (1, r)
        best = max(votes.values(), key=lambda x: x[0])
        return best[1], all_results, best[0]

    @staticmethod
    def _safe_json_parse(raw: str, default: object = None) -> object:
        if not raw or not raw.strip(): return default
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m: text = m.group(1).strip()
        text = re.sub(r",\s*(\]|\})", r"\1", text)
        try: return json.loads(text)
        except json.JSONDecodeError: pass
        m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except json.JSONDecodeError: pass
        return default

def build_llm_client() -> "LLMClient":
    return LLMClient(model=os.environ.get("LLM_MODEL", "deepseek-chat"),
                     api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                     base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
