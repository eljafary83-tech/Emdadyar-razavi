"""Production chat-completions transport with safe, typed failure mapping."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

class LLMError(Exception): pass
class LLMNotConfigured(LLMError): pass
class LLMAuthenticationError(LLMError): pass
class LLMRateLimitError(LLMError): pass
class LLMConnectionError(LLMError): pass
class LLMResponseError(LLMError): pass

@dataclass(frozen=True)
class APIConfig:
    url: str
    api_key: str
    model: str
    timeout_seconds: float = 30.0

class ChatCompletionsClient:
    """Generic OpenAI-compatible client; credentials never appear in logs or UI."""
    def __init__(self, config: APIConfig) -> None: self.config = config

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not (self.config.url and self.config.api_key and self.config.model):
            raise LLMNotConfigured("سرویس مدل زبانی پیکربندی نشده است.")
        body = json.dumps({"model": self.config.model, "messages": [
            {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0, "response_format": {"type": "json_object"}}).encode("utf-8")
        request = Request(self.config.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            logger.warning("LLM HTTP error: status=%s", error.code)
            if error.code in (401, 403): raise LLMAuthenticationError("احراز هویت سرویس مدل ناموفق بود.") from error
            if error.code == 429: raise LLMRateLimitError("سقف درخواست سرویس مدل موقتاً پر شده است.") from error
            if error.code in (400, 404): raise LLMResponseError("درخواست ارسالی به سرویس مدل معتبر نیست.") from error
            raise LLMConnectionError("سرویس مدل موقتاً در دسترس نیست.") from error
        except (URLError, TimeoutError, SocketTimeout) as error:
            logger.warning("LLM connection failure: %s", type(error).__name__)
            raise LLMConnectionError("اتصال به سرویس مدل برقرار نشد یا زمان آن تمام شد.") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            logger.warning("LLM invalid transport response: %s", type(error).__name__)
            raise LLMResponseError("پاسخ سرویس مدل معتبر نیست.") from error
        try: answer = payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as error:
            raise LLMResponseError("ساختار پاسخ سرویس مدل معتبر نیست.") from error
        if not answer: raise LLMResponseError("سرویس مدل پاسخ خالی برگرداند.")
        return answer
