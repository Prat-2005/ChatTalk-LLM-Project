"""LLM interface for ChatTalk.

Provider chain:
    1) Primary provider (LLM_PROVIDER, default `ollama`).
    2) If primary fails, try fallback provider (FALLBACK_PROVIDER, default `groq`).
    3) If everything fails, return an informative error placeholder reply.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Generator

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
    """
    Safely retrieve environment variable from os.environ or streamlit.secrets.
    Dynamically scans root secrets and all nested sections for the requested key.
    """
    # 1. Check system environment variables first
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()

    try:
        import streamlit as st
        # Safeguard if secrets are empty or not initialized
        if not hasattr(st, "secrets") or st.secrets is None:
            return default

        # 2. Check root-level secrets (exact, lower, upper)
        for key in (name, name.lower(), name.upper()):
            if key in st.secrets:
                val = str(st.secrets[key]).strip()
                if val:
                    return val

        # 3. Dynamically scan ALL nested TOML sections for the requested key
        for section in st.secrets.keys():
            try:
                sec_dict = st.secrets[section]
                # Verify the section is actually a dictionary or nested Secrets object
                if isinstance(sec_dict, (dict, type(st.secrets))):
                    for key in (name, name.lower(), name.upper()):
                        if key in sec_dict:
                            val = str(sec_dict[key]).strip()
                            if val:
                                return val
            except Exception:
                pass  # Skip malformed or inaccessible sections safely

    except Exception:
        pass  # Fallback if streamlit module is missing or completely errors out

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
    provider = _env("LLM_PROVIDER", "ollama").lower()
    model = _env("LLM_MODEL", "llama3.2")
    base_url = _env("LLM_BASE_URL", "http://localhost:11434").rstrip("/")
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "temperature": _env_float("LLM_TEMPERATURE", 0.8),
        "max_tokens": _env_int("LLM_MAX_TOKENS", 512),
    }


def _build_fallback_config() -> dict[str, Any]:
    api_key = _env("FALLBACK_API_KEY")
    provider = _env("FALLBACK_PROVIDER").lower()
    model = _env("FALLBACK_MODEL")
    base_url = _env("FALLBACK_BASE_URL").rstrip("/")
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
# Transports & Exceptions
# ---------------------------------------------------------------------------

class ProviderError(RuntimeError):
    """Raised when a provider fails. Triggers fallback."""


_USER_AGENT = "ChatTalk/1.0"


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
        f"No LLM provider generated a response. Check your environment settings or Streamlit Secrets."
        f"{err_details}"
    )


def _local_stream(messages: list[dict], cfg: dict[str, Any]) -> Generator[str, None, None]:
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
        resp = urllib.request.urlopen(req, timeout=60)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderError(f"{cfg['provider'].title()} unreachable: {exc}") from exc
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"{cfg['provider'].title()} HTTP {exc.code}: {exc.reason}") from exc

    try:
        has_content = False
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                content = (chunk.get("message") or {}).get("content", "")
                if content:
                    has_content = True
                    yield content
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                pass
        if not has_content:
            raise ProviderError(f"{cfg['provider'].title()} returned an empty response")
    finally:
        resp.close()


def _fallback_stream(messages: list[dict], cfg: dict[str, Any]) -> Generator[str, None, None]:
    api_key = cfg.get("api_key", "")
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
            f"Fallback HTTP {exc.code}: {exc.reason}"
            + (f" — {detail}" if detail else "")
        ) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderError(f"Fallback unreachable: {exc}") from exc

    try:
        has_content = False
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
                        has_content = True
                        yield content
            except json.JSONDecodeError:
                pass
        if not has_content:
            raise ProviderError("Fallback returned an empty response")
    finally:
        resp.close()


def _resolve_provider_stream(provider: str):
    if provider == _build_config().get("provider"):
        return _local_stream
    if provider == _build_fallback_config().get("provider"):
        return _fallback_stream
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


def generate_reply_stream(
    user_message: str,
    history: list[dict],
    result_info: dict[str, str],
    primary: dict[str, Any] | None = None,
    fallback: dict[str, Any] | None = None,
):
    primary = primary or (CONFIG if CONFIG.get("provider") != "ollama" or os.environ.get("LLM_PROVIDER") else _build_config())
    fallback = fallback or (FALLBACK_CONFIG if FALLBACK_CONFIG.get("api_key") or os.environ.get("FALLBACK_API_KEY") else _build_fallback_config())

    if not user_message or not user_message.strip():
        result_info["provider"] = "placeholder"
        yield "I didn't catch that — say it again?"
        return

    tone = _detect_tone_for(history, user_message)
    messages = _build_messages_for_llm(user_message, history, tone=tone)

    error_log: list[str] = []

    # 1. Primary provider
    if _is_configured(primary):
        handler = _resolve_provider_stream(primary["provider"])
        if handler:
            try:
                stream_gen = handler(messages, primary)
                first_chunk = next(stream_gen)
                result_info["provider"] = "primary"
                yield first_chunk
                yield from stream_gen
                return
            except StopIteration:
                error_log.append(f"Primary ({primary['provider']}): Empty stream response")
            except Exception as exc:
                err_msg = f"Primary ({primary['provider']}): {exc}"
                print(f"[ChatTalk] {err_msg}")
                error_log.append(err_msg)

    # 2. Fallback provider
    if _is_configured(fallback):
        handler = _resolve_provider_stream(fallback["provider"])
        if handler:
            try:
                stream_gen = handler(messages, fallback)
                first_chunk = next(stream_gen)
                result_info["provider"] = "fallback"
                yield first_chunk
                yield from stream_gen
                return
            except StopIteration:
                error_log.append(f"Fallback ({fallback['provider']}): Empty stream response")
            except Exception as exc:
                err_msg = f"Fallback ({fallback['provider']}): {exc}"
                print(f"[ChatTalk] {err_msg}")
                error_log.append(err_msg)

    # 3. Placeholder fallback with diagnostic details
    result_info["provider"] = "placeholder"
    yield _placeholder_reply(user_message, error_log)


def generate_reply(
    user_message: str,
    history: list[dict],
    primary: dict[str, Any] | None = None,
    fallback: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Non-streaming entry point — wraps generate_reply_stream cleanly."""
    result_info: dict[str, str] = {"provider": "placeholder"}
    chunks = list(generate_reply_stream(user_message, history, result_info, primary, fallback))
    return "".join(chunks), result_info["provider"]


def get_config() -> dict[str, Any]:
    return {
        "primary": _build_config(),
        "fallback": _build_fallback_config(),
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
