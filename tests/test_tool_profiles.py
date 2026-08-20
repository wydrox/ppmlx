"""Tests for explicit local tool-profile capability metadata."""
from __future__ import annotations

import pytest

from ppmlx.local_runtime.normalization import NormalizationProfile
from ppmlx.local_runtime.tool_argument_repair import ToolArgumentRepairPolicy
from ppmlx.local_runtime.tool_profiles import (
    ToolCapabilityLevel,
    ToolProfileContract,
    get_tool_profile_contract,
    list_tool_profile_contracts,
)


def test_all_existing_local_profiles_are_explicit_and_strict() -> None:
    contracts = list_tool_profile_contracts()

    assert [contract.normalization_profile for contract in contracts] == list(
        NormalizationProfile
    )
    assert all(
        contract.capability_level is ToolCapabilityLevel.TEMPLATE_STRUCTURED
        for contract in contracts
    )
    assert all(contract.repair_policy is None for contract in contracts)


@pytest.mark.parametrize("profile", list(NormalizationProfile))
def test_contract_lookup_accepts_enum_and_exact_string(
    profile: NormalizationProfile,
) -> None:
    by_enum = get_tool_profile_contract(profile)
    by_string = get_tool_profile_contract(profile.value)

    assert by_enum is not None
    assert by_string == by_enum
    assert by_enum.normalization_profile is profile


def test_unknown_profile_does_not_infer_capabilities() -> None:
    assert get_tool_profile_contract("unknown-qwen-like-name") is None
    assert get_tool_profile_contract(123) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "level",
    [
        ToolCapabilityLevel.NATIVE_STRUCTURED,
        ToolCapabilityLevel.NONE,
    ],
)
def test_repair_policy_is_rejected_for_ineligible_capability_levels(
    level: ToolCapabilityLevel,
) -> None:
    with pytest.raises(ValueError, match="Repair is unavailable"):
        ToolProfileContract(
            normalization_profile=NormalizationProfile.QWEN_JSON_V1,
            capability_level=level,
            repair_policy=ToolArgumentRepairPolicy.BOUNDED_JSON_V1,
        )


@pytest.mark.parametrize(
    "level",
    [
        ToolCapabilityLevel.TEMPLATE_STRUCTURED,
        ToolCapabilityLevel.PROMPT_EMULATED,
    ],
)
def test_repair_policy_can_be_declared_only_by_eligible_profile_types(
    level: ToolCapabilityLevel,
) -> None:
    contract = ToolProfileContract(
        normalization_profile=NormalizationProfile.QWEN_JSON_V1,
        capability_level=level,
        repair_policy=ToolArgumentRepairPolicy.BOUNDED_JSON_V1,
    )

    assert contract.repair_policy is ToolArgumentRepairPolicy.BOUNDED_JSON_V1
