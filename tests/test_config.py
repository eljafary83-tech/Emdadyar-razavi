import pytest
from config import Settings

def test_invalid_threshold_is_rejected(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "2")
    with pytest.raises(ValueError): Settings.from_environment()

def test_invalid_timeout_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError): Settings.from_environment()
