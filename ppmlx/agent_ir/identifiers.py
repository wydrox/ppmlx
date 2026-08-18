"""Validated identifiers and explicit identifier factories for Agent IR v1."""
from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from pydantic import Field, StrictInt, StrictStr, StringConstraints


ConversationId = Annotated[StrictStr, StringConstraints(pattern=r"^conv_[A-Za-z0-9_-]+$")]
RequestId = Annotated[StrictStr, StringConstraints(pattern=r"^req_[A-Za-z0-9_-]+$")]
CallId = Annotated[StrictStr, StringConstraints(min_length=1)]
OutputId = Annotated[StrictStr, StringConstraints(min_length=1)]
ParallelGroupId = Annotated[StrictStr, StringConstraints(min_length=1)]
Sequence = Annotated[StrictInt, Field(ge=0)]
ChoiceIndex = Annotated[StrictInt, Field(ge=0)]
ToolCallIndex = Annotated[StrictInt, Field(ge=0)]
ContentIndex = Annotated[StrictInt, Field(ge=0)]


def _new_identifier(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def new_conversation_id() -> str:
    return _new_identifier("conv")


def new_request_id() -> str:
    return _new_identifier("req")


def new_call_id() -> str:
    return _new_identifier("call")


def new_output_id() -> str:
    return _new_identifier("output")


def new_parallel_group_id() -> str:
    return _new_identifier("parallel")
