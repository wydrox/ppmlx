"""Explicit capability metadata for local tool-output profiles."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from .normalization import NormalizationProfile
from .tool_argument_repair import ToolArgumentRepairPolicy


class ToolCapabilityLevel(str, Enum):
    """How a model or provider emits tool calls."""

    NATIVE_STRUCTURED = "native_structured"
    TEMPLATE_STRUCTURED = "template_structured"
    PROMPT_EMULATED = "prompt_emulated"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ToolProfileContract:
    """Versioned normalization and repair settings for one local format."""

    normalization_profile: NormalizationProfile
    capability_level: ToolCapabilityLevel
    repair_policy: ToolArgumentRepairPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.normalization_profile, NormalizationProfile):
            raise ValueError("Invalid normalization profile")
        if not isinstance(self.capability_level, ToolCapabilityLevel):
            raise ValueError("Invalid tool capability level")
        if self.repair_policy is not None and not isinstance(
            self.repair_policy,
            ToolArgumentRepairPolicy,
        ):
            raise ValueError("Invalid tool argument repair policy")
        if (
            self.repair_policy is not None
            and self.capability_level
            not in {
                ToolCapabilityLevel.TEMPLATE_STRUCTURED,
                ToolCapabilityLevel.PROMPT_EMULATED,
            }
        ):
            raise ValueError("Repair is unavailable for this capability level")


_PROFILE_CONTRACTS: Mapping[NormalizationProfile, ToolProfileContract] = MappingProxyType(
    {
        profile: ToolProfileContract(
            normalization_profile=profile,
            capability_level=ToolCapabilityLevel.TEMPLATE_STRUCTURED,
        )
        for profile in NormalizationProfile
    }
)


def get_tool_profile_contract(
    profile: NormalizationProfile | str,
) -> ToolProfileContract | None:
    """Return explicit metadata without inferring an unknown profile."""

    try:
        selected = profile if isinstance(profile, NormalizationProfile) else NormalizationProfile(profile)
    except (TypeError, ValueError):
        return None
    return _PROFILE_CONTRACTS.get(selected)


def list_tool_profile_contracts() -> tuple[ToolProfileContract, ...]:
    """Return contracts in normalization-profile declaration order."""

    return tuple(_PROFILE_CONTRACTS[profile] for profile in NormalizationProfile)


__all__ = [
    "ToolCapabilityLevel",
    "ToolProfileContract",
    "get_tool_profile_contract",
    "list_tool_profile_contracts",
]
