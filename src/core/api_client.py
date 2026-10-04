"""
اتصال به سرویس‌های API (سازگار با OpenAI)
==========================================
- ‎LLMClient‎        → سرویس مدل زبانی (Qwen3 روی llama.cpp — پورت 8080)
- ‎EmbeddingClient‎  → سرویس امبدینگ (bge-m3 روی llama.cpp — پورت 8081)

نکتهٔ معماری: کدِ امبدینگ کاملاً از کدِ مدل زبانی جدا است؛ هرکدام
کلاینت مستقل خود را دارند و فقط یک کلاس پایهٔ مشترک برای منطق
HTTP/retry استفاده می‌کنند. هر دو سرویس با llama.cpp server بالا
آمده‌اند و endpoint های سازگار با OpenAI ارائه می‌دهند:

    POST /v1/chat/completions   (LLM)
    POST /v1/embeddings         (Embedding)
    GET  /health                (بررسی سلامت)
"""

from __future__ import annotations

import json
import time
from typing import Iterator

import requests

from ..config import Settings
from .exceptions import APIConnectionError, APIResponseError


class BaseAPIClient:
    """کلاینت HTTP پایه با تلاش مجدد (retry با backoff نمایی) و خطاهای سفارشی."""

    name: str = "api"

    def __init__(self, base_url: str, timeout: int, max_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self._session = requests.Session()

    # ── لایهٔ ارتباطی ─────────────────────────────────────────
    def _request(self, method: str, endpoint: str, payload: dict | None = None,
                 stream: bool = False) -> requests.Response:
        """ارسال درخواست؛ خطاهای 5xx/اتصال با retry، خطاهای 4xx فوری."""
        url = f"{self.base_url}{endpoint}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.request(
                    method, url, json=payload, timeout=self.timeout, stream=stream,
                )
                if response.status_code >= 500:
                    last_error = APIResponseError(
                        f"[{self.name}] خطای سرور {response.status_code}: {response.text[:300]}"
                    )
                elif response.status_code >= 400:
                    raise APIResponseError(
                        f"[{self.name}] درخواست رد شد ({response.status_code}): "
                        f"{response.text[:300]}"
                    )
                else:
                    return response
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = APIConnectionError(
                    f"[{self.name}] اتصال به {url} برقرار نشد: {exc}"
                )
            if attempt < self.max_retries:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s, ...
        raise last_error if last_error else APIConnectionError(f"[{self.name}] درخواست ناموفق بود.")

    def health_check(self) -> bool:
        """بررسی سریع در دسترس بودن سرویس."""
        try:
            response = self._session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def close(self) -> None:
        self._session.close()


class LLMClient(BaseAPIClient):
    """کلاینت سرویس مدل زبانی (Qwen3) — ارسال پیام‌ها و دریافت پاسخ/جریان."""

    name = "LLM"

    def __init__(self, settings: Settings):
        super().__init__(settings.llm_base_url, settings.llm_timeout,
                         settings.llm_max_retries)
        self.model = settings.llm_model

    def chat(self, messages: list[dict], temperature: float = 0.3,
             max_tokens: int = 1500) -> str:
        """ارسال غیرجریانی؛ متن کامل پاسخ را برمی‌گرداند."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = self._request("POST", "/chat/completions", payload)
        try:
            content = response.json()["choices"][0]["message"]["content"]
            return content or ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise APIResponseError(f"[LLM] ساختار پاسخ نامعتبر است: {exc}") from exc

    def chat_stream(self, messages: list[dict], temperature: float = 0.3,
                    max_tokens: int = 1500) -> Iterator[str]:
        """ارسال جریانی (SSE)؛ تکه‌های متن پاسخ را به‌تدریج برمی‌گرداند."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        response = self._request("POST", "/chat/completions", payload, stream=True)
        yield from self._iter_sse_tokens(response)

    @staticmethod
    def _iter_sse_tokens(response: requests.Response) -> Iterator[str]:
        """پیمایش جریان SSE و استخراج محتوای هر تکه."""
        for raw_line in response.iter_lines():
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="ignore")
            else:
                line = raw_line or ""
            line = line.strip()
            if not line or line.startswith(":"):
                continue        # heartbeat / کامنت
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            token = delta.get("content")
            if token:
                yield token


class EmbeddingClient(BaseAPIClient):
    """کلاینت سرویس امبدینگ (bge-m3) — جدا و مستقل از کلاینت مدل زبانی."""

    name = "Embedding"

    def __init__(self, settings: Settings):
        super().__init__(settings.embedding_base_url, settings.embedding_timeout,
                         settings.embedding_max_retries)
        self.model = settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """دریافت بردار امبدینگ برای یک لیست متن در یک درخواست."""
        if not texts:
            return []
        payload = {"model": self.model, "input": list(texts)}
        response = self._request("POST", "/embeddings", payload)
        try:
            data = response.json()["data"]
            return [
                item["embedding"]
                for item in sorted(data, key=lambda x: x.get("index", 0))
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise APIResponseError(f"[Embedding] ساختار پاسخ نامعتبر است: {exc}") from exc
