"""Tests for generic Tool/MCP JSON distillation."""
from __future__ import annotations

import json

from ppmlx.tool_distillers import GenericJsonToolDistiller


def test_generic_json_distiller_extracts_product_atoms():
    payload = {
        "products": [
            {
                "name": "LG OLED C4",
                "price": {"amount": 4599, "currency": "PLN"},
                "availability": "in_stock",
                "url": "https://example.test/lg-oled-c4",
                "specs": {"panel": "OLED", "hdmi_2_1": 4},
            },
            {
                "name": "Samsung CU8000",
                "rejected": True,
                "reason": "60Hz and no HDMI 2.1",
            },
        ]
    }

    candidates = GenericJsonToolDistiller().distill(
        {"role": "tool", "name": "product_search", "content": json.dumps(payload)},
        {"project_id": "tv-shopping", "session_id": "s1"},
    )

    texts = "\n".join(candidate.text for candidate in candidates)
    assert "Candidate: LG OLED C4." in texts
    assert "LG OLED C4 price: 4599 PLN." in texts
    assert "LG OLED C4 spec panel = OLED." in texts
    assert "LG OLED C4 spec hdmi_2_1 = 4." in texts
    assert "Rejected Samsung CU8000: 60Hz and no HDMI 2.1." in texts
    assert all(candidate.scope == "project" for candidate in candidates)
    assert all(candidate.source_quote for candidate in candidates)


def test_generic_json_distiller_does_not_promote_rejected_rows_as_candidates():
    payload = {
        "products": [
            {
                "name": "Samsung CU8000",
                "price": {"amount": 2599, "currency": "PLN"},
                "rejected": True,
                "reason": "60Hz and no HDMI 2.1",
                "specs": {"panel": "LED", "hdmi_2_1": 0},
            }
        ]
    }

    candidates = GenericJsonToolDistiller().distill(
        {"role": "tool", "name": "product_search", "content": json.dumps(payload)},
        {"project_id": "tv-shopping"},
    )

    texts = [candidate.text for candidate in candidates]
    assert texts == ["Rejected Samsung CU8000: 60Hz and no HDMI 2.1."]


def test_generic_json_distiller_fails_closed_on_large_low_salience_result_sets():
    products = [
        {"name": f"Generic LED {idx}", "price": {"amount": 3000 + idx, "currency": "PLN"}}
        for idx in range(12)
    ]
    products.append({
        "name": "LG OLED C4",
        "price": {"amount": 4599, "currency": "PLN"},
        "specs": {"panel": "OLED", "hdmi_2_1": 4},
    })

    candidates = GenericJsonToolDistiller(max_records=8).distill(
        {"role": "tool", "name": "product_search", "content": json.dumps({"products": products})},
        {"project_id": "tv-shopping"},
    )

    texts = "\n".join(candidate.text for candidate in candidates)
    assert "LG OLED C4" in texts
    assert "Generic LED" not in texts


def test_generic_json_distiller_ignores_non_tool_json_messages():
    payload = {"products": [{"name": "LG OLED C4", "price": 4599}]}

    candidates = GenericJsonToolDistiller().distill(
        {"role": "user", "content": json.dumps(payload)},
        {"project_id": "tv-shopping"},
    )

    assert candidates == []
