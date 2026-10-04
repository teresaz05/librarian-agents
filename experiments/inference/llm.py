from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)
from tqdm.asyncio import tqdm_asyncio


ReasoningMode = Literal["none", "request_param", "openrouter_extra_body"]
MaxTokensParam = Literal["max_completion_tokens", "max_tokens"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "outputs" / "cache"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    default_model: str
    api_key_env: str
    base_url: str | None = None
    reasoning_mode: ReasoningMode = "none"
    default_reasoning_effort: str | None = None
    max_tokens_param: MaxTokensParam = "max_completion_tokens"
    cache_dir: Path | None = None

    @property
    def resolved_cache_dir(self) -> Path:
        return self.cache_dir or CACHE_ROOT / self.name


PROVIDERS: dict[str, ProviderConfig] = {
    "local": ProviderConfig(
        name="local",
        default_model="openai/gpt-oss-120b",
        api_key_env="VLLM_API_KEY",
        base_url="http://127.0.0.1:8000/v1",
        reasoning_mode="request_param",
        default_reasoning_effort="low",
        max_tokens_param="max_tokens",
    ),
    "gemini": ProviderConfig(
        name="gemini",
        default_model="gemini-3.1-flash-lite-preview",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        reasoning_mode="request_param",
        default_reasoning_effort="medium",
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        default_model="claude-4-5-sonnet-20251022",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        reasoning_mode="request_param",
        default_reasoning_effort="medium",
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        default_model="qwen/qwen3.5-35b-a3b",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        reasoning_mode="openrouter_extra_body",
    ),
    "openai": ProviderConfig(
        name="openai",
        default_model="gpt-4.1",
        api_key_env="OPENAI_API_KEY",
        reasoning_mode="request_param",
    ),
    "together": ProviderConfig(
        name="together",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        max_tokens_param="max_tokens",
    ),
}

ProviderLike = str | ProviderConfig


class _ProviderDefault:
    pass


PROVIDER_DEFAULT = _ProviderDefault()
ReasoningEffort = str | None | _ProviderDefault


def get_provider_config(provider: ProviderLike) -> ProviderConfig:
    if isinstance(provider, ProviderConfig):
        return provider
    try:
        return PROVIDERS[provider]
    except KeyError:
        available = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown provider {provider!r}. Available providers: {available}") from None


def get_cache_dir(provider: ProviderLike) -> Path:
    cache_dir = get_provider_config(provider).resolved_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def create_async_client(
    provider: ProviderLike,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **client_kwargs: Any,
) -> AsyncOpenAI:
    config = get_provider_config(provider)
    resolved_api_key = api_key if api_key is not None else os.environ[config.api_key_env]
    resolved_base_url = base_url if base_url is not None else config.base_url

    kwargs = {"api_key": resolved_api_key, **client_kwargs}
    if resolved_base_url is not None:
        kwargs["base_url"] = resolved_base_url
    return AsyncOpenAI(**kwargs)


def create_client(
    provider: ProviderLike,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **client_kwargs: Any,
) -> OpenAI:
    config = get_provider_config(provider)
    resolved_api_key = api_key if api_key is not None else os.environ[config.api_key_env]
    resolved_base_url = base_url if base_url is not None else config.base_url

    kwargs = {"api_key": resolved_api_key, **client_kwargs}
    if resolved_base_url is not None:
        kwargs["base_url"] = resolved_base_url
    return OpenAI(**kwargs)


def get_cache_key(
    prompt: str,
    model: str,
    N: int,
    max_completion_tokens: int,
    reasoning_effort: str | None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Generate a unique cache key based on prompt and request parameters."""
    cache_data = {
        "prompt": prompt,
        "model": model,
        "N": N,
        "max_completion_tokens": max_completion_tokens,
        "reasoning_effort": "none" if reasoning_effort is None else reasoning_effort,
        "temperature": temperature,
        "top_p": top_p,
    }
    cache_string = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_string.encode()).hexdigest()


def _get_cached_response_sync(cache_dir: Path, cache_key: str) -> list[str] | None:
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _save_to_cache_sync(cache_dir: Path, cache_key: str, responses: list[str]) -> None:
    cache_file = cache_dir / f"{cache_key}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(responses, f)
    except IOError as e:
        print(f"Warning: Failed to save to cache: {e}")


async def get_cached_response(cache_key: str, provider: ProviderLike) -> list[str] | None:
    """Retrieve cached response if it exists."""
    cache_dir = get_cache_dir(provider)
    return await asyncio.to_thread(_get_cached_response_sync, cache_dir, cache_key)


async def save_to_cache(cache_key: str, responses: list[str], provider: ProviderLike) -> None:
    """Save responses to the provider cache."""
    cache_dir = get_cache_dir(provider)
    await asyncio.to_thread(_save_to_cache_sync, cache_dir, cache_key, responses)


def _resolve_reasoning_effort(config: ProviderConfig, reasoning_effort: ReasoningEffort) -> str | None:
    if isinstance(reasoning_effort, _ProviderDefault):
        return config.default_reasoning_effort
    return reasoning_effort


def _build_request_kwargs(
    config: ProviderConfig,
    prompt: str,
    max_completion_tokens: int,
    reasoning_effort: str | None,
    model_name: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model_name or config.default_model,
        "messages": [{"role": "user", "content": prompt}],
        config.max_tokens_param: max_completion_tokens,
    }

    if reasoning_effort is None:
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        return request_kwargs

    if config.reasoning_mode == "request_param":
        request_kwargs["reasoning_effort"] = reasoning_effort
    elif config.reasoning_mode == "openrouter_extra_body":
        request_kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}

    if temperature is not None:
        request_kwargs["temperature"] = temperature
    if top_p is not None:
        request_kwargs["top_p"] = top_p

    return request_kwargs


async def _sleep_before_retry(
    message: str,
    attempt: int,
    max_retries: int,
    *,
    cap_seconds: int,
    base_seconds: int = 1,
) -> bool:
    if attempt == max_retries - 1:
        return False
    jitter = random.uniform(0.5, 1.5)
    wait_time = min(base_seconds * (2**attempt), cap_seconds) * jitter
    print(f"{message}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
    await asyncio.sleep(wait_time)
    return True


async def generate_single_response(
    client: AsyncOpenAI,
    prompt: str,
    max_completion_tokens: int = 8192,
    reasoning_effort: ReasoningEffort = PROVIDER_DEFAULT,
    model_name: str = "",
    max_retries: int = 10,
    *,
    provider: ProviderLike = "openrouter",
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Generate a single response with retry logic for transient API failures."""
    config = get_provider_config(provider)
    resolved_reasoning_effort = _resolve_reasoning_effort(config, reasoning_effort)

    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                **_build_request_kwargs(
                    config,
                    prompt,
                    max_completion_tokens,
                    resolved_reasoning_effort,
                    model_name,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
            return completion.choices[0].message.content or ""
        except RateLimitError:
            should_retry = await _sleep_before_retry(
                "Rate limited",
                attempt,
                max_retries,
                cap_seconds=60,
                base_seconds=2,
            )
            if not should_retry:
                raise
        except (APIConnectionError, APITimeoutError) as e:
            should_retry = await _sleep_before_retry(
                f"Connection error: {e}",
                attempt,
                max_retries,
                cap_seconds=30,
            )
            if not should_retry:
                raise
        except APIStatusError as e:
            if e.status_code < 500:
                print(f"Non-retryable error {e.status_code}: {e}")
                raise
            should_retry = await _sleep_before_retry(
                f"Server error {e.status_code}",
                attempt,
                max_retries,
                cap_seconds=30,
            )
            if not should_retry:
                raise

    return ""


async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    N: int = 1,
    max_completion_tokens: int = 16384,
    reasoning_effort: ReasoningEffort = PROVIDER_DEFAULT,
    use_cache: bool = True,
    semaphore: asyncio.Semaphore | None = None,
    model_name: str = "",
    *,
    provider: ProviderLike = "openrouter",
    temperature: float | None = None,
    top_p: float | None = None,
) -> list[str]:
    config = get_provider_config(provider)
    resolved_model = model_name or config.default_model
    resolved_reasoning_effort = _resolve_reasoning_effort(config, reasoning_effort)
    cache_key = get_cache_key(
        prompt,
        resolved_model,
        N,
        max_completion_tokens,
        resolved_reasoning_effort,
        temperature,
        top_p,
    )

    if use_cache:
        cached = await get_cached_response(cache_key, config)
        if cached is not None:
            return cached

    try:
        # Semaphore controls prompt-level concurrency. Once acquired, all N
        # samples for that prompt are still issued in parallel.
        async def make_all_requests() -> list[str]:
            tasks = [
                generate_single_response(
                    client,
                    prompt,
                    max_completion_tokens,
                    resolved_reasoning_effort,
                    resolved_model,
                    provider=config,
                    temperature=temperature,
                    top_p=top_p,
                )
                for _ in range(N)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            responses = []
            for result in results:
                if isinstance(result, Exception):
                    print(f"Request failed: {result}")
                    responses.append("")
                else:
                    responses.append(result)
            return responses

        if semaphore:
            async with semaphore:
                responses = await make_all_requests()
        else:
            responses = await make_all_requests()

        if use_cache:
            await save_to_cache(cache_key, responses, config)

        return responses
    except Exception as e:
        print(f"Error generating response: {e}")
        return [""] * N


async def generate_responses(
    client: AsyncOpenAI,
    prompts: Sequence[str],
    N: int = 1,
    max_completion_tokens: int = 16384,
    reasoning_effort: ReasoningEffort = PROVIDER_DEFAULT,
    use_cache: bool = True,
    max_concurrent: int = 50,
    model_name: str = "",
    *,
    provider: ProviderLike = "openrouter",
    temperature: float | None = None,
    top_p: float | None = None,
) -> list[list[str]]:
    """Generate responses for multiple prompts concurrently.

    max_concurrent controls prompt-level concurrency. Total in-flight API
    requests can be as high as max_concurrent * N.
    """
    config = get_provider_config(provider)
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        generate_response(
            client,
            prompt,
            N,
            max_completion_tokens,
            reasoning_effort,
            use_cache,
            semaphore,
            model_name,
            provider=config,
            temperature=temperature,
            top_p=top_p,
        )
        for prompt in prompts
    ]
    return await tqdm_asyncio.gather(*tasks, desc=f"Processing {config.name} prompts")


def clear_cache(provider: ProviderLike) -> None:
    """Clear all cached responses for a provider."""
    cache_dir = get_cache_dir(provider)
    for cache_file in cache_dir.glob("*.json"):
        cache_file.unlink()
    print(f"Cache cleared: {cache_dir}")
