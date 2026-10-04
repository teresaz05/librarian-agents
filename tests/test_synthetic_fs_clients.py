from __future__ import annotations

from environments.synthetic_fs import clients


def test_openai_compatible_usage_callback_receives_successful_request() -> None:
    observed: list[dict[str, int]] = []

    text = clients._call_openai_compatible_chat_model_messages(
        backend="vllm",
        model="test-model",
        base_url="http://localhost:8000/v1",
        api_key_env="",
        messages=[{"role": "user", "content": "hello"}],
        response_json=False,
        max_output_tokens=8,
        open_json=lambda _request: {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
        usage_callback=observed.append,
    )

    assert text == "done"
    assert observed == [
        {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6, "requests": 1}
    ]


def test_gemini_usage_callback_receives_successful_request(monkeypatch) -> None:
    monkeypatch.setenv("TEST_GEMINI_KEY", "secret")
    monkeypatch.setattr(
        clients,
        "urlopen_json_with_retry",
        lambda _request, *, max_attempts: {
            "candidates": [{"content": {"parts": [{"text": "done"}]}}],
            "usageMetadata": {
                "promptTokenCount": 7,
                "candidatesTokenCount": 3,
                "totalTokenCount": 10,
            },
        },
    )
    observed: list[dict[str, int]] = []

    text = clients.call_chat_model_messages(
        backend="gemini",
        model="test-model",
        base_url="https://example.invalid/v1beta",
        api_key_env="TEST_GEMINI_KEY",
        messages=[{"role": "user", "content": "hello"}],
        response_json=False,
        max_output_tokens=8,
        http_max_attempts=1,
        usage_callback=observed.append,
    )

    assert text == "done"
    assert observed == [
        {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10, "requests": 1}
    ]
