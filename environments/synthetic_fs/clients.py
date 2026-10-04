from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from environments.synthetic_fs.common import (
    JudgeBackend,
    MODEL_HTTP_MAX_ATTEMPTS,
    MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS,
    MODEL_HTTP_TIMEOUT_SECONDS,
)


CONTEXT_COMPLETION_SAFETY_TOKENS = 16


def context_limited_completion_tokens(
    error: Exception,
    requested: int,
    *,
    safety_tokens: int = CONTEXT_COMPLETION_SAFETY_TOKENS,
) -> int | None:
    """Return a conservative retry cap parsed from an OpenAI context error."""
    text = str(error)
    context_match = re.search(r"maximum context length is (\d+) tokens", text)
    input_match = re.search(r"prompt contains at least (\d+) input tokens", text)
    if context_match is None or input_match is None:
        return None
    available = int(context_match.group(1)) - int(input_match.group(1)) - safety_tokens
    if available < 1:
        return 0
    return available if available < requested else None


def context_limited_prompt_tokens(
    error: Exception,
    requested_output: int,
    *,
    safety_tokens: int = CONTEXT_COMPLETION_SAFETY_TOKENS,
) -> int | None:
    """Return a safe vLLM input cap parsed from a context-boundary error."""
    text = str(error)
    context_match = re.search(
        r"maximum context length is\s+(\d+)\s+tokens",
        text,
        flags=re.IGNORECASE,
    )
    prompt_length_match = re.search(
        r"(?:your\s+)?prompt\s+contains\b",
        text,
        flags=re.IGNORECASE,
    )
    if context_match is None or prompt_length_match is None:
        return None
    available = int(context_match.group(1)) - requested_output - safety_tokens
    return max(available, 0)


def _normalize_model_http_url(url: str) -> str:
    url = url.strip()
    if url.startswith("//"):
        return f"https:{url}"
    return url


def _openai_compatible_chat_completions_url(base_url: str) -> str:
    url = _normalize_model_http_url(base_url).rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _api_key_from_env(api_key_env: str, *, required: bool) -> str:
    env_name = api_key_env.strip()
    if not env_name:
        if required:
            raise RuntimeError("Missing API key env var name")
        return ""
    api_key = os.getenv(env_name, "").strip()
    if not api_key and required:
        raise RuntimeError(f"Missing API key in env var {env_name}")
    return api_key


def _json_headers_with_optional_bearer(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _openai_compatible_model_candidates(backend: JudgeBackend, model: str) -> list[str]:
    model = model.strip()
    candidates = [model or "llm"]
    if backend == "vllm" and model.lower() == "qwen/qwen3.5-35b-a3b":
        candidates.append("llm")
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _should_retry_vllm_model_alias(error: Exception) -> bool:
    text = str(error).lower()
    missing_model_markers = (
        "does not exist",
        "not exist",
        "not found",
        "not served",
        "not available",
        "available models",
    )
    return "model" in text and any(marker in text for marker in missing_model_markers)


def _openai_compatible_chat_payload(
    *,
    model: str,
    messages: Sequence[dict[str, str]],
    response_json: bool,
    max_output_tokens: int | None,
    temperature: float = 0,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    enable_thinking: bool | None = None,
    truncate_prompt_tokens: int | None = None,
    truncation_side: str | None = None,
) -> dict[str, Any]:
    system_messages = [m for m in messages if m.get("role") == "system"]
    non_system_messages = [m for m in messages if m.get("role") != "system"]
    payload: dict[str, Any] = {
        "model": model,
        "messages": [],
        "temperature": temperature,
    }
    if max_output_tokens is not None and max_output_tokens > 0:
        payload["max_tokens"] = max_output_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if min_p is not None:
        payload["min_p"] = min_p
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if truncate_prompt_tokens is not None and truncate_prompt_tokens > 0:
        payload["truncate_prompt_tokens"] = truncate_prompt_tokens
    if truncation_side is not None:
        payload["truncation_side"] = truncation_side
    if response_json and not system_messages:
        payload["messages"].append({"role": "system", "content": "Return only compact JSON."})
    payload["messages"].extend(system_messages)
    payload["messages"].extend(non_system_messages)
    return payload


def _openai_compatible_response_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"Response missing choices: {body}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(parts).strip()
    return str(content).strip()


def _openai_compatible_usage(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", prompt + completion) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "requests": 1,
    }


def _gemini_usage(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usageMetadata")
    usage = usage if isinstance(usage, dict) else {}
    prompt = int(usage.get("promptTokenCount", 0) or 0)
    completion = int(usage.get("candidatesTokenCount", 0) or 0)
    total = int(usage.get("totalTokenCount", prompt + completion) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "requests": 1,
    }


def _call_openai_compatible_chat_model_messages(
    *,
    backend: JudgeBackend,
    model: str,
    base_url: str,
    api_key_env: str,
    messages: Sequence[dict[str, str]],
    response_json: bool,
    max_output_tokens: int | None,
    open_json: Callable[[urllib.request.Request], dict[str, Any]],
    temperature: float = 0,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    enable_thinking: bool | None = None,
    truncate_prompt_tokens: int | None = None,
    truncation_side: str | None = None,
    usage_callback: Callable[[dict[str, int]], None] | None = None,
) -> str:
    url = _openai_compatible_chat_completions_url(base_url)
    api_key = _api_key_from_env(
        api_key_env,
        required=backend == "openrouter" and bool(api_key_env.strip()),
    )
    model_candidates = _openai_compatible_model_candidates(backend, model)
    last_error: Exception | None = None

    for i, candidate_model in enumerate(model_candidates):
        payload = _openai_compatible_chat_payload(
            model=candidate_model,
            messages=messages,
            response_json=response_json,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
            truncate_prompt_tokens=(truncate_prompt_tokens if backend == "vllm" else None),
            truncation_side=(truncation_side if backend == "vllm" else None),
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=_json_headers_with_optional_bearer(api_key),
            method="POST",
        )
        try:
            body = open_json(req)
            text = _openai_compatible_response_text(body)
            if usage_callback is not None:
                usage_callback(_openai_compatible_usage(body))
            return text
        except RuntimeError as e:
            last_error = e
            has_fallback = i + 1 < len(model_candidates)
            if backend == "vllm" and has_fallback and _should_retry_vllm_model_alias(e):
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("No OpenAI-compatible model candidates were available")


def call_chat_model_messages(
    *,
    backend: JudgeBackend,
    model: str,
    base_url: str,
    api_key_env: str,
    messages: Sequence[dict[str, str]],
    response_json: bool,
    max_output_tokens: int | None = None,
    temperature: float = 0,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    enable_thinking: bool | None = None,
    truncate_prompt_tokens: int | None = None,
    truncation_side: str | None = None,
    http_max_attempts: int | None = None,
    usage_callback: Callable[[dict[str, int]], None] | None = None,
) -> str:
    effective_http_attempts = (
        MODEL_HTTP_MAX_ATTEMPTS if http_max_attempts is None else http_max_attempts
    )
    if effective_http_attempts < 1:
        raise ValueError("http_max_attempts must be positive")
    if backend in {"openrouter", "vllm"}:
        return _call_openai_compatible_chat_model_messages(
            backend=backend,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            messages=messages,
            response_json=response_json,
            max_output_tokens=max_output_tokens,
            open_json=lambda req: urlopen_json_with_retry(
                req, max_attempts=effective_http_attempts
            ),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
            truncate_prompt_tokens=truncate_prompt_tokens,
            truncation_side=truncation_side,
            usage_callback=usage_callback,
        )

    if backend == "gemini":
        api_key = _api_key_from_env(api_key_env, required=True)
        model_path = model if model.startswith("models/") else f"models/{model}"
        url = f"{_normalize_model_http_url(base_url).rstrip('/')}/{model_path}:generateContent"
        system_texts = [
            m["content"] for m in messages if m.get("role") == "system" and m.get("content")
        ]
        contents = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": message.get("content", "")}]})
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if max_output_tokens is not None and max_output_tokens > 0:
            payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
        if top_p is not None:
            payload["generationConfig"]["topP"] = top_p
        if top_k is not None:
            payload["generationConfig"]["topK"] = top_k
        if presence_penalty is not None:
            payload["generationConfig"]["presencePenalty"] = presence_penalty
        if enable_thinking is False:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
        if response_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        body = urlopen_json_with_retry(req, max_attempts=effective_http_attempts)

        candidates = body.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Response missing candidates: {body}")
        content = (candidates[0].get("content") or {}).get("parts") or []
        text_parts = [
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("text")
        ]
        text = "".join(text_parts).strip()
        if usage_callback is not None:
            usage_callback(_gemini_usage(body))
        return text

    raise ValueError(f"Unknown backend: {backend}")


def urlopen_json_with_retry(
    req: urllib.request.Request, *, max_attempts: int = MODEL_HTTP_MAX_ATTEMPTS
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    last_error: Exception | None = None
    retryable_http_codes = {429, 500, 502, 503, 504}

    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=MODEL_HTTP_TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code in retryable_http_codes and attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS))
                last_error = RuntimeError(f"HTTP error {e.code}: {body[:500]}")
                continue
            raise RuntimeError(f"HTTP error {e.code}: {body[:500]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS))
                last_error = RuntimeError(f"URL/timeout error: {e}")
                continue
            raise RuntimeError(f"URL/timeout error: {e}") from e

    if last_error is not None:
        raise last_error
    raise RuntimeError("Request failed without a captured error")
