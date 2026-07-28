"""LLM interface for ChatTalk.

Provider chain:
    1) Primary provider (LLM_PROVIDER, default `ollama`).
    2) If it raises (timeout, connection error, HTTP error), try the
         fallback provider (FALLBACK_PROVIDER, default `groq`).
    3) If everything fails, return a placeholder reply so the UI stays usable.

The fallback uses Groq's OpenAI-compatible `/chat/completions` HTTP shape, so
the transport stays simple and does not depend on the `groq` SDK.
Authentication uses `FALLBACK_API_KEY`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from prompts import (
    DEFAULT_TONE,
    ToneSignal,
    build_system_prompt,
    detect_tone,
    trim_history,
    _slang_mirror_instruction,
)


# ---------------------------------------------------------------------------
# .env and Secrets loader
# ---------------------------------------------------------------------------

def _load_dotenv(path: str | os.PathLike = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    """Retrieve environment variable from os.environ or streamlit.secrets."""
    value = os.environ.get(name)
    if value is not None and value.strip() != "":
        return value.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            sec_val = str(st.secrets[name])
            if sec_val and sec_val.strip() != "":
                return sec_val.strip()
    except Exception:
        pass
    return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _build_config() -> dict[str, Any]:
    return {
        "provider": _env("LLM_PROVIDER").lower(),
        "model": _env("LLM_MODEL"),
        "base_url": _env("LLM_BASE_URL").rstrip("/"),
        "temperature": _env_float("LLM_TEMPERATURE", 0.8),
        "max_tokens": _env_int("LLM_MAX_TOKENS", 512),
    }


def _build_fallback_config() -> dict[str, Any]:
    return {
        "provider": _env("FALLBACK_PROVIDER").lower(),
        "model": _env("FALLBACK_MODEL"),
        "base_url": _env("FALLBACK_BASE_URL").rstrip("/"),
        "api_key": _env("FALLBACK_API_KEY"),
        "temperature": _env_float("LLM_TEMPERATURE", 0.8),
        "max_tokens": _env_int("LLM_MAX_TOKENS", 512),
    }


CONFIG: dict[str, Any] = _build_config()
FALLBACK_CONFIG: dict[str, Any] = _build_fallback_config()


# ---------------------------------------------------------------------------
# Placeholder fallback
# ---------------------------------------------------------------------------

def _placeholder_reply(user_message: str) -> str:
    if not user_message or not user_message.strip():
        return "I didn't catch that — say it again?"
    preview = user_message.strip()
    if len(preview) > 60:
        preview = preview[:57] + "..."
    return (
        f"(placeholder) You said: \"{preview}\". "
        f"No LLM provider is reachable right now — set LLM_PROVIDER / "
        f"FALLBACK_API_KEY in your environment or Streamlit Secrets."
    )


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------

class ProviderError(RuntimeError):
    """Raised when a provider fails. Triggers fallback."""


_USER_AGENT = "ChatTalk/1.0"


def _ollama_chat(messages: list[dict], cfg: dict[str, Any]) -> str:
    url = f"{cfg['base_url']}/api/chat"
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": cfg["temperature"],
            "num_predict": cfg["max_tokens"],
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderError(f"Ollama unreachable: {exc}") from exc
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"Ollama HTTP {exc.code}: {exc.reason}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Ollama returned non-JSON: {exc}") from exc

    content = (parsed.get("message") or {}).get("content", "").strip()
    if not content:
        raise ProviderError("Ollama returned an empty response")
    return content


def _chat_completions_request(
    url: str,
    payload: dict,
    api_key: str,
    timeout: float,
) -> dict:
    """POST `payload` to `url` with bearer auth, return parsed JSON."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise ProviderError(
            f"Fallback HTTP {exc.code}: {exc.reason}"
            + (f" — {detail}" if detail else "")
        ) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderError(f"Fallback unreachable: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Fallback returned non-JSON: {exc}") from exc


def _groq_chat(messages: list[dict], cfg: dict[str, Any]) -> str:
    """Call Groq's OpenAI-compatible `/chat/completions` endpoint."""
    api_key = cfg.get("api_key") or ""
    if not api_key:
        raise ProviderError("FALLBACK_API_KEY is not set.")

    base_url = cfg["base_url"].rstrip("/")
    if not base_url.endswith("/openai/v1"):
        base_url = f"{base_url}/openai/v1"
    url = f"{base_url}/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
    }
    parsed = _chat_completions_request(url, payload, api_key, timeout=20)

    choices = parsed.get("choices") or []
    if not choices:
        raise ProviderError("Fallback returned no choices")
    content = (choices[0].get("message") or {}).get("content", "").strip()
    if not content:
        raise ProviderError("Fallback returned an empty response")
    return content


def _ollama_chat_stream(messages: list[dict], cfg: dict[str, Any]):
    url = f"{cfg['base_url']}/api/chat"
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": cfg["temperature"],
            "num_predict": cfg["max_tokens"],
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderError(f"Ollama stream unreachable: {exc}") from exc
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"Ollama stream HTTP {exc.code}: {exc.reason}") from exc

    try:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                content = (chunk.get("message") or {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                pass
    finally:
        resp.close()


def _groq_chat_stream(messages: list[dict], cfg: dict[str, Any]):
    api_key = cfg.get("api_key") or ""
    if not api_key:
        raise ProviderError("FALLBACK_API_KEY is not set for stream.")

    base_url = cfg["base_url"].rstrip("/")
    if not base_url.endswith("/openai/v1"):
        base_url = f"{base_url}/openai/v1"
    url = f"{base_url}/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise ProviderError(
            f"Fallback stream HTTP {exc.code}: {exc.reason}"
            + (f" — {detail}" if detail else "")
        ) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderError(f"Fallback stream unreachable: {exc}") from exc

    try:
        for line in resp:
            line = line.decode("utf-8").strip() if isinstance(line, bytes) else line.strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices") or []
                if choices:
                    content = (choices[0].get("delta") or {}).get("content", "")
                    if content:
                        yield content
            except json.JSONDecodeError:
                pass
    finally:
        resp.close()


def _resolve_handler(provider: str):
    if provider == "ollama":
        return _ollama_chat
    if provider == "groq":
        return _groq_chat
    return None


def _resolve_stream_handler(provider: str):
    if provider == "ollama":
        return _ollama_chat_stream
    if provider == "groq":
        return _groq_chat_stream
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _detect_tone_for(history: list[dict], current: str) -> ToneSignal:
    user_texts = [m["content"] for m in history if m.get("role") == "user"]
    user_texts.append(current)
    return detect_tone(user_texts)


def _build_messages_for_llm(
    user_message: str,
    history: list[dict],
    tone: ToneSignal,
) -> list[dict]:
    user_messages_so_far = [m["content"] for m in history if m.get("role") == "user"]
    system_prompt = (
        build_system_prompt(tone)
        + _slang_mirror_instruction(user_messages_so_far + [user_message])
    )
    trimmed_history = trim_history(history)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_message})
    return messages


def _is_configured(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("provider")) and bool(cfg.get("model"))


def _try_provider(
    cfg: dict[str, Any],
    messages: list[dict],
) -> str:
    handler = _resolve_handler(cfg["provider"])
    if handler is None:
        raise ProviderError(f"Unknown provider: {cfg['provider']!r}")

    return handler(messages, dict(cfg))


def generate_reply(
    user_message: str,
    history: list[dict],
    primary: dict[str, Any] | None = None,
    fallback: dict[str, Any] | None = None,
) -> tuple[str, str]:
    primary = primary or CONFIG
    fallback = fallback or FALLBACK_CONFIG

    if not user_message or not user_message.strip():
        return "I didn't catch that — say it again?", "placeholder"

    tone = _detect_tone_for(history, user_message)
    messages = _build_messages_for_llm(user_message, history, tone=tone)

    if _is_configured(primary):
        try:
            return _try_provider(primary, messages), "primary"
        except ProviderError as exc:
            print(f"[ChatTalk] Primary ({primary['provider']}) failed: {exc}")

    if _is_configured(fallback):
        try:
            return _try_provider(fallback, messages), "fallback"
        except ProviderError as exc:
            print(f"[ChatTalk] Fallback ({fallback['provider']}) failed: {exc}")

    return _placeholder_reply(user_message), "placeholder"


def generate_reply_stream(
    user_message: str,
    history: list[dict],
    result_info: dict[str, str],
    primary: dict[str, Any] | None = None,
    fallback: dict[str, Any] | None = None,
):
    primary = primary or CONFIG
    fallback = fallback or FALLBACK_CONFIG

    if not user_message or not user_message.strip():
        result_info["provider"] = "placeholder"
        yield "I didn't catch that — say it again?"
        return

    tone = _detect_tone_for(history, user_message)
    messages = _build_messages_for_llm(user_message, history, tone=tone)

    if _is_configured(primary):
        handler = _resolve_stream_handler(primary["provider"])
        if handler:
            try:
                result_info["provider"] = "primary"
                yield from handler(messages, primary)
                return
            except ProviderError as exc:
                print(f"[ChatTalk] Primary stream ({primary['provider']}) failed: {exc}")

    if _is_configured(fallback):
        handler = _resolve_stream_handler(fallback["provider"])
        if handler:
            try:
                result_info["provider"] = "fallback"
                yield from handler(messages, fallback)
                return
            except ProviderError as exc:
                print(f"[ChatTalk] Fallback stream ({fallback['provider']}) failed: {exc}")

    result_info["provider"] = "placeholder"
    yield _placeholder_reply(user_message)


def get_config() -> dict[str, Any]:
    return {
        "primary": dict(CONFIG),
        "fallback": dict(FALLBACK_CONFIG),
    }


def get_last_tone(history: list[dict], current: str = "") -> ToneSignal:
    return _detect_tone_for(history, current)


def reload_config() -> None:
    global CONFIG, FALLBACK_CONFIG
    _load_dotenv()
    CONFIG = _build_config()
    FALLBACK_CONFIG = _build_fallback_config()


__all__ = [
    "CONFIG",
    "FALLBACK_CONFIG",
    "ProviderError",
    "generate_reply",
    "generate_reply_stream",
    "get_config",
    "get_last_tone",
    "reload_config",
    "trim_history",
    "DEFAULT_TONE",
]
