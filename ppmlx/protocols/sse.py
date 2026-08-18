"""Canonical Server-Sent Event helpers with bounded frame parsing."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

from ppmlx.protocols.base import AdapterLimits, ProtocolAdapterError
from ppmlx.protocols.json import parse_json_object


@dataclass(frozen=True, slots=True)
class SSEFrame:
    event: str | None
    data: Any


def encode_sse_frame(data: object, *, event: str | None = None) -> str:
    if event is not None and re.fullmatch(r"[A-Za-z0-9_.-]+", event) is None:
        raise ValueError("SSE event name is invalid")
    if type(data) is str and data == "[DONE]":
        encoded = data
    elif type(data) is dict:
        encoded = json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    else:
        raise ValueError("SSE data must be a JSON object or the terminal sentinel")
    event_line = f"event: {event}\n" if event is not None else ""
    return f"{event_line}data: {encoded}\n\n"


def encode_sse(
    frames: Iterable[SSEFrame],
    *,
    protocol: str = "sse",
    limits: AdapterLimits | None = None,
) -> str:
    selected_limits = limits or AdapterLimits()
    parts: list[str] = []
    total_bytes = 0
    for count, frame in enumerate(frames, start=1):
        if count > selected_limits.max_events:
            raise ProtocolAdapterError(protocol=protocol, code="too_many_sse_frames")
        encoded = encode_sse_frame(frame.data, event=frame.event)
        frame_bytes = len(encoded.encode("utf-8"))
        if frame_bytes > selected_limits.max_sse_frame_bytes:
            raise ProtocolAdapterError(protocol=protocol, code="sse_frame_too_large")
        total_bytes += frame_bytes
        if total_bytes > selected_limits.max_sse_stream_bytes:
            raise ProtocolAdapterError(protocol=protocol, code="sse_stream_too_large")
        parts.append(encoded)
    return "".join(parts)


def parse_sse(
    value: str | bytes,
    *,
    protocol: str,
    limits: AdapterLimits | None = None,
) -> tuple[SSEFrame, ...]:
    selected_limits = limits or AdapterLimits()
    decode_failed = False
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
    except UnicodeDecodeError:
        text = ""
        decode_failed = True
    if decode_failed or not isinstance(text, str):
        raise ProtocolAdapterError(protocol=protocol, code="invalid_sse")
    if len(text.encode("utf-8")) > selected_limits.max_sse_stream_bytes:
        raise ProtocolAdapterError(protocol=protocol, code="sse_stream_too_large")
    frames: list[SSEFrame] = []
    for raw_frame in text.replace("\r\n", "\n").split("\n\n"):
        if not raw_frame.strip():
            continue
        if len(frames) >= selected_limits.max_events:
            raise ProtocolAdapterError(protocol=protocol, code="too_many_sse_frames")
        if len(raw_frame.encode("utf-8")) > selected_limits.max_sse_frame_bytes:
            raise ProtocolAdapterError(protocol=protocol, code="sse_frame_too_large")
        event: str | None = None
        data_lines: list[str] = []
        for line in raw_frame.splitlines():
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[6:].lstrip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line:
                raise ProtocolAdapterError(protocol=protocol, code="invalid_sse")
        if not data_lines:
            raise ProtocolAdapterError(protocol=protocol, code="invalid_sse")
        raw_data = "\n".join(data_lines)
        if raw_data == "[DONE]":
            data: Any = raw_data
        else:
            try:
                data = parse_json_object(
                    raw_data,
                    protocol=protocol,
                    limits=selected_limits,
                )
            except ProtocolAdapterError:
                pass
            else:
                frames.append(SSEFrame(event=event, data=data))
                continue
            raise ProtocolAdapterError(protocol=protocol, code="invalid_sse")
        frames.append(SSEFrame(event=event, data=data))
    return tuple(frames)


__all__ = ["SSEFrame", "encode_sse", "encode_sse_frame", "parse_sse"]
