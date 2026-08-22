"""Local MLX implementation of the protocol-neutral provider interface."""
from __future__ import annotations

import platform
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence

from ppmlx.agent_ir import (
    AgentEvent,
    new_call_id,
    new_output_id,
    new_parallel_group_id,
)
from ppmlx.local_runtime.backend import (
    LocalEngineRequest,
    LocalExecution,
    LocalGeneration,
    LocalGenerator,
    LocalRuntimeError,
    TerminalReasons,
    execute_local_request,
)
from ppmlx.local_runtime.normalization import (
    NormalizationProfile,
    ToolNormalizationError,
    select_normalization_profile,
)
from ppmlx.models import (
    ModelNotFoundError,
    list_local_models,
    resolve_model_path,
)

from .base import (
    ProviderCallReference,
    ProviderCapabilities,
    ProviderCancelledError,
    ProviderCredentialType,
    ProviderDataPath,
    ProviderError,
    ProviderHealth,
    ProviderHealthStatus,
    ProviderInvocation,
    ProviderModel,
    ProviderResult,
    ProviderStreamingMode,
    ProviderToolSupportStatus,
)


ModelLister = Callable[[], Sequence[Mapping[str, object]]]
ModelResolver = Callable[[str], str]
ProfileSelector = Callable[[str], NormalizationProfile | None]
PlatformProbe = Callable[[], tuple[str, str]]

_CANONICAL_TERMINAL_REASONS = TerminalReasons(
    text="stop",
    tool_calls="tool_calls",
)


def _default_generate(request: LocalEngineRequest) -> LocalGeneration:
    from ppmlx.local_runtime.runtime import default_local_generator

    return default_local_generator(request)


def _default_platform_probe() -> tuple[str, str]:
    return sys.platform, platform.machine().lower()


def _uses_image(invocation: ProviderInvocation) -> bool:
    request = invocation.request.request
    for instruction in request.instructions:
        if any(block.type == "image" for block in instruction.content):
            return True
    for message in request.messages:
        if any(block.type == "image" for block in message.content):
            return True
    return False


class MLXProvider:
    """Wrap the existing strict local backend behind the provider contract."""

    def __init__(
        self,
        *,
        generate: LocalGenerator = _default_generate,
        model_lister: ModelLister = list_local_models,
        model_resolver: ModelResolver | None = None,
        profile_selector: ProfileSelector = select_normalization_profile,
        platform_probe: PlatformProbe = _default_platform_probe,
        call_id_factory: Callable[[], str] = new_call_id,
        output_id_factory: Callable[[], str] = new_output_id,
        parallel_group_factory: Callable[[], str] = new_parallel_group_id,
        allow_download: bool = False,
    ) -> None:
        self._allow_download = bool(allow_download)
        if model_resolver is None:
            model_resolver = self._fail_closed_model_resolver
        for value in (
            generate,
            model_lister,
            model_resolver,
            profile_selector,
            platform_probe,
            call_id_factory,
            output_id_factory,
            parallel_group_factory,
        ):
            if not callable(value):
                raise ValueError("MLX provider dependency is not callable")
        self._generate = generate
        self._model_lister = model_lister
        self._model_resolver = model_resolver
        self._profile_selector = profile_selector
        self._platform_probe = platform_probe
        self._call_id_factory = call_id_factory
        self._output_id_factory = output_id_factory
        self._parallel_group_factory = parallel_group_factory

    @property
    def provider_id(self) -> str:
        return "mlx"

    def _fail_closed_model_resolver(self, model_id: str) -> str:
        """Resolve only models already downloaded locally.

        The provider declares ``ProviderDataPath.LOCAL``: silently letting the
        MLX loader fetch a HuggingFace repo derived from caller input would
        break that claim (and the SSRF/trusted-config rules in
        docs/security/threat-model.md). Downloads require the explicit
        ``allow_download=True`` opt-in on this provider.
        """
        return resolve_model_path(model_id, allow_download=self._allow_download)

    def capabilities(self, model_id: str) -> ProviderCapabilities:
        if type(model_id) is not str or not model_id or any(
            character.isspace() for character in model_id
        ):
            raise ProviderError(provider_id=self.provider_id, code="invalid_model_id")
        try:
            profile = self._profile_selector(model_id)
        except Exception:
            raise ProviderError(
                provider_id=self.provider_id,
                code="capability_resolution_failed",
            ) from None
        supports_tools = profile is not None
        return ProviderCapabilities(
            text=True,
            images=False,
            tools=supports_tools,
            parallel_tool_calls=supports_tools,
            reasoning=False,
            streaming=ProviderStreamingMode.BUFFERED,
            context_window=None,
            data_path=ProviderDataPath.LOCAL,
            credential_types=(ProviderCredentialType.NONE,),
            tool_support_status=(
                ProviderToolSupportStatus.NOT_EVALUATED
                if supports_tools
                else ProviderToolSupportStatus.DISABLED
            ),
        )

    def list_models(self) -> tuple[ProviderModel, ...]:
        try:
            rows = self._model_lister()
        except Exception:
            raise ProviderError(
                provider_id=self.provider_id,
                code="model_registry_unavailable",
            ) from None
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            raise ProviderError(
                provider_id=self.provider_id,
                code="invalid_model_registry",
            )
        models: list[ProviderModel] = []
        seen_models: set[str] = set()
        seen_aliases: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ProviderError(
                    provider_id=self.provider_id,
                    code="invalid_model_registry",
                )
            model_id = row.get("repo_id")
            alias = row.get("alias")
            if type(model_id) is not str or not model_id or any(
                character.isspace() for character in model_id
            ):
                raise ProviderError(
                    provider_id=self.provider_id,
                    code="invalid_model_registry",
                )
            if model_id in seen_models:
                raise ProviderError(
                    provider_id=self.provider_id,
                    code="duplicate_model_id",
                )
            aliases: tuple[str, ...] = ()
            if alias is not None and alias != model_id:
                if type(alias) is not str or not alias or any(
                    character.isspace() for character in alias
                ):
                    raise ProviderError(
                        provider_id=self.provider_id,
                        code="invalid_model_registry",
                    )
                if alias in seen_aliases:
                    raise ProviderError(
                        provider_id=self.provider_id,
                        code="duplicate_model_alias",
                    )
                aliases = (alias,)
                seen_aliases.add(alias)
            seen_models.add(model_id)
            models.append(
                ProviderModel(
                    provider_id=self.provider_id,
                    model_id=model_id,
                    aliases=aliases,
                    capabilities=self.capabilities(model_id),
                )
            )
        return tuple(sorted(models, key=lambda model: model.model_id))

    def health(self) -> ProviderHealth:
        try:
            system, architecture = self._platform_probe()
        except Exception:
            return ProviderHealth(
                provider_id=self.provider_id,
                status=ProviderHealthStatus.UNAVAILABLE,
                code="platform_probe_failed",
            )
        if system != "darwin" or architecture != "arm64":
            return ProviderHealth(
                provider_id=self.provider_id,
                status=ProviderHealthStatus.UNAVAILABLE,
                code="unsupported_platform",
            )
        try:
            models = self.list_models()
        except ProviderError:
            return ProviderHealth(
                provider_id=self.provider_id,
                status=ProviderHealthStatus.UNAVAILABLE,
                code="model_registry_unavailable",
            )
        if not models:
            return ProviderHealth(
                provider_id=self.provider_id,
                status=ProviderHealthStatus.DEGRADED,
                code="no_models",
            )
        return ProviderHealth(
            provider_id=self.provider_id,
            status=ProviderHealthStatus.HEALTHY,
            code="ready",
            model_count=len(models),
        )

    def _execute(self, invocation: ProviderInvocation) -> LocalExecution:
        if not isinstance(invocation, ProviderInvocation):
            raise ProviderError(
                provider_id=self.provider_id,
                code="invalid_invocation",
            )
        capabilities = self.capabilities(invocation.model_id)
        if invocation.request.request.tools and not capabilities.tools:
            raise ProviderError(
                provider_id=self.provider_id,
                code="tools_unsupported",
            )
        if invocation.enable_reasoning and not capabilities.reasoning:
            raise ProviderError(
                provider_id=self.provider_id,
                code="reasoning_unsupported",
            )
        if _uses_image(invocation) and not capabilities.images:
            raise ProviderError(
                provider_id=self.provider_id,
                code="images_unsupported",
            )
        try:
            model = self._model_resolver(invocation.model_id)
            if type(model) is not str or not model:
                raise ProviderError(
                    provider_id=self.provider_id,
                    code="model_unavailable",
                )
            profile = self._profile_selector(model)
            return execute_local_request(
                invocation.request,
                model=model,
                generate=self._generate,
                profile=profile,
                terminal_reasons=_CANONICAL_TERMINAL_REASONS,
                output_id=invocation.output_id,
                call_id_factory=self._call_id_factory,
                output_id_factory=self._output_id_factory,
                parallel_group_factory=self._parallel_group_factory,
                sequence_start=invocation.sequence_start,
                max_tokens_cap=invocation.max_tokens_cap,
                enable_thinking=invocation.enable_reasoning,
                parallel_tool_calls=invocation.parallel_tool_calls,
            )
        except ProviderError:
            raise
        except ModelNotFoundError:
            raise ProviderError(
                provider_id=self.provider_id,
                code="model_unavailable",
            ) from None
        except (LocalRuntimeError, ToolNormalizationError) as error:
            raise ProviderError(
                provider_id=self.provider_id,
                code=error.code,
            ) from None
        except Exception:
            raise ProviderError(
                provider_id=self.provider_id,
                code="provider_invoke_failed",
            ) from None

    def invoke(self, invocation: ProviderInvocation) -> ProviderResult:
        handle = invocation.cancel_handle
        if handle is not None and handle.cancelled:
            return ProviderResult(
                provider_id=self.provider_id,
                model_id=invocation.model_id,
                events=(),
                streaming=ProviderStreamingMode.BUFFERED,
                cancelled=True,
            )
        execution = self._execute(invocation)
        calls = tuple(
            ProviderCallReference(
                call_id=reference.call_id,
                name=reference.name,
                choice_index=reference.choice_index,
                output_id=reference.output_id,
                tool_call_index=reference.tool_call_index,
                parallel_group_id=reference.parallel_group_id,
            )
            for reference in execution.calls
        )
        return ProviderResult(
            provider_id=self.provider_id,
            model_id=invocation.model_id,
            events=execution.events,
            calls=calls,
            source_call_ids=execution.source_call_ids,
            streaming=ProviderStreamingMode.BUFFERED,
            cancelled=handle is not None and handle.cancelled,
        )

    def stream(self, invocation: ProviderInvocation) -> Iterator[AgentEvent]:
        handle = invocation.cancel_handle
        if handle is not None and handle.cancelled:
            raise ProviderCancelledError(provider_id=self.provider_id)
        events = self.invoke(invocation).events

        def _generate() -> Iterator[AgentEvent]:
            for event in events:
                if handle is not None and handle.cancelled:
                    # Typed, observable cancellation: never a silent stop.
                    raise ProviderCancelledError(provider_id=self.provider_id)
                yield event

        return _generate()


__all__ = ["MLXProvider"]
