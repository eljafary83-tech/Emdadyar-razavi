from __future__ import annotations
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
import pytest
from llm.client import APIConfig, ChatCompletionsClient, LLMAuthenticationError, LLMConnectionError, LLMNotConfigured, LLMRateLimitError, LLMResponseError
def configured(): return ChatCompletionsClient(APIConfig("https://example.invalid", "key", "model"))
def call(client): return client.complete("system", "user")
def test_unconfigured_api_is_controlled():
    with pytest.raises(LLMNotConfigured): call(ChatCompletionsClient(APIConfig("", "", "")))
@pytest.mark.parametrize("status,error", [(400, LLMResponseError), (401, LLMAuthenticationError), (403, LLMAuthenticationError), (404, LLMResponseError), (429, LLMRateLimitError), (500, LLMConnectionError), (502, LLMConnectionError), (503, LLMConnectionError)])
def test_http_errors_are_mapped(monkeypatch, status, error):
    monkeypatch.setattr("llm.client.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(HTTPError("u", status, "x", {}, None)))
    with pytest.raises(error): call(configured())
@pytest.mark.parametrize("failure", [URLError("offline"), SocketTimeout()])
def test_connection_and_timeout_are_mapped(monkeypatch, failure):
    monkeypatch.setattr("llm.client.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(failure))
    with pytest.raises(LLMConnectionError): call(configured())
@pytest.mark.parametrize("body", [b'{', b'{}', b'{"choices":[{"message":{"content":" "}}]}'])
def test_invalid_or_empty_response_is_mapped(monkeypatch, body):
    class Response:
        def read(self): return body
        def __enter__(self): return self
        def __exit__(self, *_): return False
    monkeypatch.setattr("llm.client.urlopen", lambda *_a, **_k: Response())
    with pytest.raises(LLMResponseError): call(configured())
