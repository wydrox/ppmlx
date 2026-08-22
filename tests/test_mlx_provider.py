from __future__ import annotations

from collections.abc import Mapping

import pytest

from ppmlx.agent_ir import (
    ContentDeltaEvent,
    ResponseCompletedEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallCompletedEvent,
)
from ppmlx.local_runtime.backend import LocalGeneration
from ppmlx.local_runtime.normalization import NormalizationProfile
from ppmlx.protocols import DecodeContext, openai_chat_adapter
from ppmlx.providers import (
    MLXProvider,
    Provider,
    ProviderCredentialType,
    ProviderDataPath,
    ProviderError,
    ProviderHealthStatus,
    ProviderInvocation,
    ProviderStreamingMode,
    ProviderToolSupportStatus,
)


def _request(
    *,
    model: str = "mlx-community/Qwen3",
    tools: bool = False,
):
    native: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "List files."}],
        "stream": True,
    }
    if tools:
        native["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ]
        native["tool_choice"] = "auto"
    return openai_chat_adapter.decode_request(
        native,
        context=DecodeContext(request_id="req_provider", kind="initial"),
    ).request


def _rows() -> list[Mapping[str, object]]:
    return [
        {
            "repo_id": "mlx-community/Qwen3",
            "alias": "qwen3:local",
            "path": "/private/local/model",
        },
        {
            "repo_id": "meta-llama/Llama-3",
            "alias": "meta-llama/Llama-3",
            "path": "/private/local/llama",
        },
    ]


def test_mlx_provider_satisfies_interface_and_lists_stable_provider_models() -> None:
    provider = MLXProvider(
        generate=lambda request: LocalGeneration("unused", 0, 0),
        model_lister=_rows,
        model_resolver=lambda model: model,
        platform_probe=lambda: ("darwin", "arm64"),
    )

    assert isinstance(provider, Provider)
    models = provider.list_models()
    assert [model.model_id for model in models] == [
        "meta-llama/Llama-3",
        "mlx-community/Qwen3",
    ]
    assert models[1].aliases == ("qwen3:local",)
    qwen = models[1].capabilities
    assert qwen.data_path is ProviderDataPath.LOCAL
    assert qwen.credential_types == (ProviderCredentialType.NONE,)
    assert qwen.streaming is ProviderStreamingMode.BUFFERED
    # No reviewed evidence covers mlx-community/Qwen3, so a family-name match
    # does not create a tool capability claim.
    assert qwen.tools is False
    assert qwen.parallel_tool_calls is False
    assert qwen.tool_support_status is ProviderToolSupportStatus.DISABLED
    llama = models[0].capabilities
    assert llama.tools is False
    assert llama.tool_support_status is ProviderToolSupportStatus.DISABLED


def test_mlx_provider_health_is_local_platform_and_model_aware() -> None:
    healthy = MLXProvider(
        generate=lambda request: LocalGeneration("unused", 0, 0),
        model_lister=_rows,
        platform_probe=lambda: ("darwin", "arm64"),
    ).health()
    empty = MLXProvider(
        generate=lambda request: LocalGeneration("unused", 0, 0),
        model_lister=lambda: (),
        platform_probe=lambda: ("darwin", "arm64"),
    ).health()
    unsupported = MLXProvider(
        generate=lambda request: LocalGeneration("unused", 0, 0),
        model_lister=_rows,
        platform_probe=lambda: ("linux", "x86_64"),
    ).health()

    assert (healthy.status, healthy.code, healthy.model_count) == (
        ProviderHealthStatus.HEALTHY,
        "ready",
        2,
    )
    assert (empty.status, empty.code) == (
        ProviderHealthStatus.DEGRADED,
        "no_models",
    )
    assert (unsupported.status, unsupported.code) == (
        ProviderHealthStatus.UNAVAILABLE,
        "unsupported_platform",
    )


def test_mlx_provider_invokes_text_through_agent_ir_without_protocol_objects() -> None:
    provider = MLXProvider(
        generate=lambda request: LocalGeneration("Visible answer", 7, 3),
        model_lister=_rows,
        model_resolver=lambda model: f"/models/{model}",
        profile_selector=lambda model: None,
        output_id_factory=lambda: "output_generated",
    )
    invocation = ProviderInvocation(
        request=_request(model="meta-llama/Llama-3"),
        model_id="meta-llama/Llama-3",
        output_id="output_provider",
    )

    result = provider.invoke(invocation)

    assert result.provider_id == "mlx"
    assert result.model_id == "meta-llama/Llama-3"
    assert result.streaming is ProviderStreamingMode.BUFFERED
    assert result.calls == ()
    assert isinstance(result.events[1], ContentDeltaEvent)
    assert result.events[1].delta == "Visible answer"
    terminal = result.events[-1]
    assert isinstance(terminal, ResponseCompletedEvent)
    assert terminal.finish_reason == "stop"
    assert terminal.usage.total_tokens == 10


def test_mlx_provider_preserves_tool_arguments_and_source_call_identity() -> None:
    raw_arguments = '{"cmd":"printf  a  b"}'
    output = (
        '{"role":"assistant","content":null,"tool_calls":['
        '{"id":"source_call","type":"function","function":'
        '{"name":"bash","arguments":'
        + repr(raw_arguments).replace("'", '"')
        + "}}]}"
    )
    # Build the exact JSON safely so the embedded arguments remain a JSON string.
    output = (
        '{"role":"assistant","content":null,"tool_calls":['
        '{"id":"source_call","type":"function","function":'
        '{"name":"bash","arguments":"{\\"cmd\\":\\"printf  a  b\\"}"}}]}'
    )
    provider = MLXProvider(
        generate=lambda request: LocalGeneration(output, 11, 5),
        model_lister=_rows,
        model_resolver=lambda model: model,
        profile_selector=lambda _model: NormalizationProfile.GROK_OPENAI_CHAT_V1,
        call_id_factory=lambda: "public_call",
    )
    invocation = ProviderInvocation(
        request=_request(model="xai/grok-4", tools=True),
        model_id="xai/grok-4",
        output_id="output_provider",
    )

    result = provider.invoke(invocation)

    delta = next(
        event
        for event in result.events
        if isinstance(event, ToolCallArgumentsDeltaEvent)
    )
    completed = next(
        event for event in result.events if isinstance(event, ToolCallCompletedEvent)
    )
    assert delta.delta == raw_arguments
    assert completed.arguments_raw == raw_arguments
    assert completed.arguments_json == {"cmd": "printf  a  b"}
    assert result.calls[0].call_id == "public_call"
    assert result.source_call_ids == {"public_call": "source_call"}


def test_mlx_provider_buffered_stream_runs_generation_once() -> None:
    calls = 0

    def generate(request):
        nonlocal calls
        calls += 1
        return LocalGeneration("one response", 1, 1)

    provider = MLXProvider(
        generate=generate,
        model_lister=_rows,
        model_resolver=lambda model: model,
        profile_selector=lambda model: None,
    )
    invocation = ProviderInvocation(
        request=_request(model="meta-llama/Llama-3"),
        model_id="meta-llama/Llama-3",
    )

    events = tuple(provider.stream(invocation))

    assert calls == 1
    assert events
    assert isinstance(events[-1], ResponseCompletedEvent)


def test_mlx_provider_rejects_unsupported_capabilities_before_generation() -> None:
    generated = False

    def generate(request):
        nonlocal generated
        generated = True
        return LocalGeneration("not reached", 1, 1)

    provider = MLXProvider(
        generate=generate,
        model_lister=_rows,
        model_resolver=lambda model: model,
        profile_selector=lambda model: None,
    )

    with pytest.raises(ProviderError) as tools_error:
        provider.invoke(
            ProviderInvocation(
                request=_request(model="meta-llama/Llama-3", tools=True),
                model_id="meta-llama/Llama-3",
            )
        )
    with pytest.raises(ProviderError) as reasoning_error:
        provider.invoke(
            ProviderInvocation(
                request=_request(model="meta-llama/Llama-3"),
                model_id="meta-llama/Llama-3",
                enable_reasoning=True,
            )
        )

    assert tools_error.value.code == "tools_unsupported"
    assert reasoning_error.value.code == "reasoning_unsupported"
    assert generated is False


def test_mlx_provider_sanitizes_registry_and_generation_failures() -> None:
    def fail_registry():
        raise RuntimeError("registry secret")

    def fail_generation(request):
        raise RuntimeError("model output secret")

    with pytest.raises(ProviderError) as registry_error:
        MLXProvider(
            generate=lambda request: LocalGeneration("unused", 0, 0),
            model_lister=fail_registry,
        ).list_models()

    provider = MLXProvider(
        generate=fail_generation,
        model_lister=_rows,
        model_resolver=lambda model: model,
        profile_selector=lambda model: None,
    )
    with pytest.raises(ProviderError) as generation_error:
        provider.invoke(
            ProviderInvocation(
                request=_request(model="meta-llama/Llama-3"),
                model_id="meta-llama/Llama-3",
            )
        )

    assert registry_error.value.code == "model_registry_unavailable"
    assert "registry secret" not in str(registry_error.value)
    assert generation_error.value.code == "generation_failed"
    assert "model output secret" not in str(generation_error.value)
