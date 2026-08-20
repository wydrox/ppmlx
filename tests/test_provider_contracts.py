from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType

import pytest

from ppmlx.providers import (
    ProviderCapabilities,
    ProviderCredentialType,
    ProviderDataPath,
    ProviderError,
    ProviderHealth,
    ProviderHealthStatus,
    ProviderModel,
    ProviderResult,
    ProviderStreamingMode,
    ProviderToolSupportStatus,
)


ROOT = Path(__file__).resolve().parents[1]


def _capabilities(*, tools: bool = True) -> ProviderCapabilities:
    return ProviderCapabilities(
        text=True,
        images=False,
        tools=tools,
        parallel_tool_calls=tools,
        reasoning=False,
        streaming=ProviderStreamingMode.BUFFERED,
        context_window=None,
        data_path=ProviderDataPath.LOCAL,
        credential_types=(ProviderCredentialType.NONE,),
        tool_support_status=(
            ProviderToolSupportStatus.NOT_EVALUATED
            if tools
            else ProviderToolSupportStatus.DISABLED
        ),
    )


def test_provider_capability_vocabulary_is_exact_and_route_relevant() -> None:
    assert {value.value for value in ProviderDataPath} == {"local", "remote"}
    assert {value.value for value in ProviderCredentialType} == {
        "none",
        "api_key",
        "oauth_session",
    }
    assert {value.value for value in ProviderStreamingMode} == {
        "none",
        "buffered",
        "native",
    }
    assert {value.value for value in ProviderHealthStatus} == {
        "healthy",
        "degraded",
        "unavailable",
    }
    assert {value.value for value in ProviderToolSupportStatus} == {
        "not_evaluated",
        "stable",
        "preview",
        "experimental",
        "disabled",
    }


def test_provider_capabilities_reject_inconsistent_parallel_tools() -> None:
    with pytest.raises(ValueError):
        ProviderCapabilities(
            text=True,
            images=False,
            tools=False,
            parallel_tool_calls=True,
            reasoning=False,
            streaming=ProviderStreamingMode.BUFFERED,
            context_window=None,
            data_path=ProviderDataPath.LOCAL,
            credential_types=(ProviderCredentialType.NONE,),
        )


def test_provider_model_and_health_are_sanitized_immutable_values() -> None:
    model = ProviderModel(
        provider_id="mlx",
        model_id="mlx-community/Qwen3",
        aliases=("qwen3:local",),
        capabilities=_capabilities(),
    )
    health = ProviderHealth(
        provider_id="mlx",
        status=ProviderHealthStatus.HEALTHY,
        code="ready",
        model_count=1,
    )

    assert model.provider_id == "mlx"
    assert model.aliases == ("qwen3:local",)
    assert health.code == "ready"

    with pytest.raises(ValueError):
        ProviderModel(
            provider_id="MLX!",
            model_id="mlx-community/Qwen3",
            capabilities=_capabilities(),
        )


def test_provider_result_copies_source_identifier_mapping() -> None:
    source = {"call_public": "call_source"}
    result = ProviderResult(
        provider_id="mlx",
        model_id="mlx-community/Qwen3",
        events=(),
        source_call_ids=source,
    )
    source["call_other"] = "call_secret"

    assert result.source_call_ids == MappingProxyType(
        {"call_public": "call_source"}
    )
    with pytest.raises(TypeError):
        result.source_call_ids["call_other"] = "value"  # type: ignore[index]


def test_provider_error_never_keeps_untrusted_code_or_identifier() -> None:
    error = ProviderError(
        provider_id="bad provider secret",
        code="token=secret-value",
    )

    assert error.provider_id == "provider"
    assert error.code == "provider_failed"
    assert "secret-value" not in str(error)


def test_provider_base_imports_only_agent_ir_from_ppmlx() -> None:
    path = ROOT / "ppmlx" / "providers" / "base.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert {name for name in imports if name.startswith("ppmlx.")} == {
        "ppmlx.agent_ir"
    }
