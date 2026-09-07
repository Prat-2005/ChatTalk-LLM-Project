from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Generator
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from backend.prompts import (
    DEFAULT_TONE,
    ToneSignal,
    build_system_prompt,
    detect_tone,
    trim_history,
    _slang_mirror_instruction,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# .env loader
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
    """Retrieve environment variable from os.environ only."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
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

def _normalize_base_url(url: str, provider: str) -> str:
    """Ensure the base URL ends with '/v1' for OpenAI-compatible endpoints."""
    url = url.rstrip("/")
    if provider.lower() == "ollama" and not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _build_config(provider_env: str = "LLM_PROVIDER") -> dict[str, Any]:
    provider = _env(provider_env, "ollama").lower()
    model = _env("LLM_MODEL", "llama3.2")
    base_url = _env("LLM_BASE_URL", "http://localhost:11434")
    base_url = _normalize_base_url(base_url, provider)
    api_key = _env("LLM_API_KEY", "ollama")  # Ollama ignores the key
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": _env_float("LLM_TEMPERATURE", 0.8),
        "max_tokens": _env_int("LLM_MAX_TOKENS", 512),
    }


def _build_fallback_config() -> dict[str, Any]:
    provider = _env("FALLBACK_PROVIDER", "").lower()
    model = _env("FALLBACK_MODEL", "")
    base_url = _env("FALLBACK_BASE_URL", "")
    base_url = _normalize_base_url(base_url, provider) if base_url else ""
    api_key = _env("FALLBACK_API_KEY", "")
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": _env_float("LLM_TEMPERATURE", 0.8),
        "max_tokens": _env_int("LLM_MAX_TOKENS", 512),
    }


CONFIG: dict[str, Any] = _build_config()
FALLBACK_CONFIG: dict[str, Any] = _build_fallback_config()


# ---------------------------------------------------------------------------
# OpenAI Client & Streaming
# ---------------------------------------------------------------------------

class ProviderError(RuntimeError):
    """Raised when an LLM provider fails."""


def _create_openai_client(cfg: dict[str, Any]) -> OpenAI:
    """Build an OpenAI client from a provider configuration."""
    base_url = cfg.get("base_url", "")
    api_key = cfg.get("api_key", "dummy")  # some endpoints require a non-empty key
    if not base_url:
        raise ProviderError("Missing base_url for provider")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)


def _openai_stream(messages: list[dict], cfg: dict[str, Any]) -> Generator[str, None, None]:
    """Stream completions using the OpenAI client."""
    client = _create_openai_client(cfg)
    try:
        stream = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except (APIError, APIConnectionError, RateLimitError) as exc:
        raise ProviderError(f"{cfg['provider'].title()} error: {exc}") from exc


def _placeholder_reply(user_message: str, errors: list[str] | None = None) -> str:
    if not user_message or not user_message.strip():
        return "I didn't catch that — say it again?"
    preview = user_message.strip()
    if len(preview) > 60:
        preview = preview[:57] + "..."

    err_details = ""
    if errors:
        err_details = "\n\n**Diagnostic Details:**\n" + "\n".join(f"- {e}" for e in errors)

    return (
        f"(placeholder) You said: \"{preview}\".\n"
        f"No LLM provider generated a response. Check your environment settings."
        f"{err_details}"
    )


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
    return bool(cfg.get("provider")) and bool(cfg.get("model")) and bool(cfg.get("base_url"))


def generate_reply_stream(
    user_message: str,
    history: list[dict],
    result_info: dict[str, str],
    primary: dict[str, Any] | None = None,
    fallback: dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """
    Stream a reply from the primary or fallback provider.

    Yields:
        Chunks of the assistant's reply.
        result_info is updated with 'provider': 'primary' | 'fallback' | 'placeholder'.
    """
    primary = primary or CONFIG
    fallback = fallback or FALLBACK_CONFIG

    if not user_message or not user_message.strip():
        result_info["provider"] = "placeholder"
        yield "I didn't catch that — say it again?"
        return

    tone = _detect_tone_for(history, user_message)
    messages = _build_messages_for_llm(user_message, history, tone)

    error_log: list[str] = []

    # Try primary first
    if _is_configured(primary):
        try:
            stream_gen = _openai_stream(messages, primary)
            first_chunk = next(stream_gen)
            result_info["provider"] = "primary"
            yield first_chunk
            yield from stream_gen
            return
        except StopIteration:
            error_log.append(f"Primary ({primary['provider']}): Empty stream response")
        except Exception as exc:
            err_msg = f"Primary ({primary['provider']}): {exc}"
            logger.error(err_msg)
            error_log.append(err_msg)

    # Then fallback
    if _is_configured(fallback):
        try:
            stream_gen = _openai_stream(messages, fallback)
            first_chunk = next(stream_gen)
            result_info["provider"] = "fallback"
            yield first_chunk
            yield from stream_gen
            return
        except StopIteration:
            error_log.append(f"Fallback ({fallback['provider']}): Empty stream response")
        except Exception as exc:
            err_msg = f"Fallback ({fallback['provider']}): {exc}"
            logger.error(err_msg)
            error_log.append(err_msg)

    # Ultimate fallback
    result_info["provider"] = "placeholder"
    yield _placeholder_reply(user_message, error_log)


def generate_reply(
    user_message: str,
    history: list[dict],
    primary: dict[str, Any] | None = None,
    fallback: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Non-streaming wrapper around generate_reply_stream."""
    result_info: dict[str, str] = {"provider": "placeholder"}
    chunks = list(generate_reply_stream(user_message, history, result_info, primary, fallback))
    return "".join(chunks), result_info["provider"]


def generate_title(history: list[dict]) -> str:
    if not history:
        return "New Chat"

    summary_prompt = (
        "Based on the following conversation, generate a suitable concise title "
        "(maximum 5 words). Return ONLY the title text, with no quotes or extra formatting.\n\n"
    )
    for msg in history[-4:]:
        summary_prompt += f"{msg['role']}: {msg['content']}\n"

    reply, _ = generate_reply(summary_prompt, [])
    title = reply.strip(' "\'\n\r*')
    if len(title) > 50:
        title = title[:47] + "..."
    return title or "New Chat"


def get_config() -> dict[str, Any]:
    return {
        "primary": CONFIG,
        "fallback": FALLBACK_CONFIG,
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