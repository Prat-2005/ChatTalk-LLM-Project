"""Tests for llm.py — config, placeholder fallback, Groq transport, and tone helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm  # noqa: E402
from llm import ProviderError, generate_reply, get_config, get_last_tone  # noqa: E402


class TestConfig:
    def test_primary_and_fallback_returned(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "llama3.2")
        monkeypatch.setenv("FALLBACK_PROVIDER", "groq")
        monkeypatch.setenv("FALLBACK_API_KEY", "gsk-test")
        llm.reload_config()

        cfg = get_config()
        assert cfg["primary"]["provider"] == "ollama"
        assert cfg["fallback"]["provider"] == "groq"
        assert cfg["fallback"]["api_key"] == "gsk-test"

    def test_fallback_empty_when_unset(self, monkeypatch):
        monkeypatch.setattr(llm, "_load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.delenv("FALLBACK_API_KEY", raising=False)
        llm.reload_config()

        cfg = get_config()
        assert cfg["fallback"]["api_key"] == ""

    def test_reload_picks_up_env_changes(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "m1")
        llm.reload_config()
        assert llm.CONFIG["model"] == "m1"

        monkeypatch.setenv("LLM_MODEL", "m2")
        llm.reload_config()
        assert llm.CONFIG["model"] == "m2"


class TestPlaceholderFallback:
    def test_empty_message(self):
        reply = llm._placeholder_reply("")
        assert "didn't catch" in reply.lower() or "again" in reply.lower()

    def test_long_message_truncated(self):
        reply = llm._placeholder_reply("x" * 200)
        assert "..." in reply
        assert len(reply) < 300

    def test_unknown_provider_returns_placeholder(self, monkeypatch):
        monkeypatch.setattr(llm, "CONFIG", {
            "provider": "mystery",
            "model": "x",
            "base_url": "http://x",
            "temperature": 0.5,
            "max_tokens": 100,
        })
        monkeypatch.setattr(llm, "FALLBACK_CONFIG", {
            "provider": "",
            "model": "",
            "base_url": "",
            "api_key": "",
            "temperature": 0.5,
            "max_tokens": 100,
        })

        reply, label = generate_reply("hi", [])
        assert "placeholder" in reply.lower()
        assert label == "placeholder"


class TestGenerateReplyOllama:
    def _patched_primary(self):
        return {
            "provider": "ollama",
            "model": "llama3.2",
            "base_url": "http://localhost:11434",
            "temperature": 0.8,
            "max_tokens": 512,
        }

    def _patched_fallback_blank(self):
        return {
            "provider": "",
            "model": "",
            "base_url": "",
            "api_key": "",
            "temperature": 0.8,
            "max_tokens": 512,
        }

    def test_uses_primary_when_available(self, monkeypatch):
        monkeypatch.setattr(llm, "CONFIG", self._patched_primary())
        monkeypatch.setattr(llm, "FALLBACK_CONFIG", self._patched_fallback_blank())

        def fake_chat(messages, cfg):
            return "hello from primary"

        monkeypatch.setattr(llm, "_ollama_chat", fake_chat)
        reply, label = generate_reply("hi", [])
        assert reply == "hello from primary"
        assert label == "primary"

    def test_falls_back_when_primary_raises(self, monkeypatch):
        monkeypatch.setattr(llm, "CONFIG", self._patched_primary())
        monkeypatch.setattr(llm, "FALLBACK_CONFIG", {
            "provider": "groq",
            "model": "llama3-8b-8192",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": "gsk-x",
            "temperature": 0.7,
            "max_tokens": 200,
        })

        def boom(messages, cfg):
            raise ProviderError("connection refused")

        def ok(messages, cfg):
            return "hello from fallback"

        monkeypatch.setattr(llm, "_ollama_chat", boom)
        monkeypatch.setattr(llm, "_groq_chat", ok)
        reply, label = generate_reply("hi", [])
        assert reply == "hello from fallback"
        assert label == "fallback"

    def test_returns_placeholder_when_both_fail(self, monkeypatch):
        monkeypatch.setattr(llm, "CONFIG", self._patched_primary())
        monkeypatch.setattr(llm, "FALLBACK_CONFIG", {
            "provider": "groq",
            "model": "llama3-8b-8192",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": "gsk-x",
            "temperature": 0.7,
            "max_tokens": 200,
        })

        def boom(messages, cfg):
            raise ProviderError("nope")

        monkeypatch.setattr(llm, "_ollama_chat", boom)
        monkeypatch.setattr(llm, "_groq_chat", boom)
        reply, label = generate_reply("hi", [])
        assert "placeholder" in reply.lower()
        assert label == "placeholder"
        assert "tried" not in reply.lower()

    def test_history_is_passed_through(self, monkeypatch):
        monkeypatch.setattr(llm, "CONFIG", self._patched_primary())
        monkeypatch.setattr(llm, "FALLBACK_CONFIG", self._patched_fallback_blank())
        seen = {}

        def fake_chat(messages, cfg):
            seen["messages"] = messages
            return "ok"

        monkeypatch.setattr(llm, "_ollama_chat", fake_chat)
        history = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        generate_reply("new question", history)
        contents = [m["content"] for m in seen["messages"]]
        assert "earlier question" in contents
        assert "earlier answer" in contents
        assert "new question" in contents


class TestOllamaTransport:
    def test_parses_response(self, monkeypatch):
        class FakeResp:
            def __init__(self, body):
                self._body = body.encode("utf-8")

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        body = json.dumps({"message": {"content": "  hi from ollama  "}})
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["data"] = json.loads(req.data.decode("utf-8"))
            return FakeResp(body)

        monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
        cfg = {
            "provider": "ollama",
            "model": "llama3.2",
            "base_url": "http://localhost:11434",
            "temperature": 0.5,
            "max_tokens": 100,
        }
        out = llm._ollama_chat([{"role": "user", "content": "hi"}], cfg)
        assert out == "hi from ollama"
        assert captured["url"].endswith("/api/chat")
        assert captured["data"]["model"] == "llama3.2"
        assert captured["data"]["options"]["temperature"] == 0.5

    def test_wraps_url_errors_as_provider_error(self, monkeypatch):
        def fake_urlopen(req, timeout):
            raise llm.urllib.error.URLError("nope")

        monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
        cfg = {
            "provider": "ollama",
            "model": "llama3.2",
            "base_url": "http://localhost:11434",
            "temperature": 0.5,
            "max_tokens": 100,
        }
        with pytest.raises(ProviderError):
            llm._ollama_chat([{"role": "user", "content": "hi"}], cfg)

    def test_wraps_http_errors_as_provider_error(self, monkeypatch):
        def fake_urlopen(req, timeout):
            raise llm.urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, None)

        monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
        cfg = {
            "provider": "ollama",
            "model": "llama3.2",
            "base_url": "http://localhost:11434",
            "temperature": 0.5,
            "max_tokens": 100,
        }
        with pytest.raises(ProviderError):
            llm._ollama_chat([{"role": "user", "content": "hi"}], cfg)

    def test_empty_response_is_provider_error(self, monkeypatch):
        class FakeResp:
            def __init__(self, body):
                self._body = body.encode("utf-8")

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        body = json.dumps({"message": {"content": ""}})
        monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: FakeResp(body))
        cfg = {
            "provider": "ollama",
            "model": "llama3.2",
            "base_url": "http://localhost:11434",
            "temperature": 0.5,
            "max_tokens": 100,
        }
        with pytest.raises(ProviderError):
            llm._ollama_chat([{"role": "user", "content": "hi"}], cfg)


class TestGroqTransport:
    def _cfg(self, **overrides):
        cfg = {
            "provider": "groq",
            "model": "llama3-8b-8192",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": "gsk-test",
            "temperature": 0.7,
            "max_tokens": 200,
        }
        cfg.update(overrides)
        return cfg

    def test_requires_api_key(self):
        with pytest.raises(ProviderError):
            llm._groq_chat([{"role": "user", "content": "hi"}], self._cfg(api_key=""))

    def test_calls_chat_completions_with_bearer(self, monkeypatch):
        body = json.dumps({"choices": [{"message": {"content": "  hi from groq  "}}]})
        captured = {}

        class FakeResp:
            def __init__(self, body):
                self._body = body.encode("utf-8")

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.headers)
            captured["data"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResp(body)

        monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
        out = llm._groq_chat(
            [
                {"role": "system", "content": "you are chatty"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi back"},
                {"role": "user", "content": "how are you?"},
            ],
            self._cfg(),
        )
        assert out == "hi from groq"
        assert captured["url"].endswith("/openai/v1/chat/completions")
        assert captured["headers"]["Authorization"] == "Bearer gsk-test"
        roles = [m["role"] for m in captured["data"]["messages"]]
        assert roles == ["system", "user", "assistant", "user"]
        assert captured["data"]["model"] == "llama3-8b-8192"
        assert captured["data"]["max_tokens"] == 200
        assert captured["data"]["temperature"] == 0.7

    def test_wraps_url_errors_as_provider_error(self, monkeypatch):
        def fake_urlopen(req, timeout):
            raise llm.urllib.error.URLError("nope")

        monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(ProviderError):
            llm._groq_chat([{"role": "user", "content": "hi"}], self._cfg())

    def test_wraps_http_errors_as_provider_error(self, monkeypatch):
        body = b"upstream said no"

        def fake_urlopen(req, timeout):
            raise llm.urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, _FakeBody(body))

        monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(ProviderError) as ei:
            llm._groq_chat([{"role": "user", "content": "hi"}], self._cfg())
        assert "401" in str(ei.value)
        assert "upstream said no" in str(ei.value)

    def test_non_json_response_is_provider_error(self, monkeypatch):
        class FakeResp:
            def __init__(self, body):
                self._body = body.encode("utf-8")

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: FakeResp("not json"))
        with pytest.raises(ProviderError):
            llm._groq_chat([{"role": "user", "content": "hi"}], self._cfg())

    def test_empty_choices_is_provider_error(self, monkeypatch):
        class FakeResp:
            def __init__(self, body):
                self._body = body.encode("utf-8")

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        body = json.dumps({"choices": []})
        monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: FakeResp(body))
        with pytest.raises(ProviderError):
            llm._groq_chat([{"role": "user", "content": "hi"}], self._cfg())

    def test_empty_content_is_provider_error(self, monkeypatch):
        class FakeResp:
            def __init__(self, body):
                self._body = body.encode("utf-8")

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        body = json.dumps({"choices": [{"message": {"content": ""}}]})
        monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: FakeResp(body))
        with pytest.raises(ProviderError):
            llm._groq_chat([{"role": "user", "content": "hi"}], self._cfg())


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def close(self):
        pass


class TestGetLastTone:
    def test_returns_tone_signal(self):
        tone = get_last_tone([], current="OMG this is amazing!!")
        assert tone.label == "excited"

    def test_considers_history(self):
        history = [
            {"role": "user", "content": "I'm feeling really down today"},
            {"role": "assistant", "content": "I'm sorry to hear that"},
        ]
        tone = get_last_tone(history, current="yeah, just sad")
        assert tone.label == "sad"
