from __future__ import annotations
import time
import uuid
from typing import Literal
from pydantic import BaseModel, Field, field_validator


def make_request_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:12]


def make_completion_id() -> str:
    return "cmpl-" + uuid.uuid4().hex[:12]


def now_ts() -> int:
    return int(time.time())


# ── Content parts (for multimodal) ─────────────────────────────────────

class ImageURL(BaseModel):
    url: str  # https:// URL or data:image/...;base64,...


class ContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None


# ── Chat messages ───────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    model_config = {"extra": "allow"}
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[ContentPart] | None = None
    reasoning: str | None = None  # ppmlx extension: populated for <think> models


# ── Chat completion request ─────────────────────────────────────────────

class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=2.0)  # ppmlx extension
    think: bool | None = None  # ppmlx extension: force enable/disable thinking
    reasoning_budget: int | None = Field(default=None, ge=1, le=131072)  # ppmlx extension: max reasoning tokens
    draft_model: str | None = None  # ppmlx extension: speculative decoding
    speculative_tokens: int | None = None  # ppmlx extension: draft tokens per step

    @field_validator("messages")
    @classmethod
    def at_least_one_message(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("messages must contain at least one message")
        return v


# ── Chat completion response ────────────────────────────────────────────

class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"] | None = "stop"
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=make_request_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=now_ts)
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# ── Streaming chunk ─────────────────────────────────────────────────────

class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=make_request_id)
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=now_ts)
    model: str
    choices: list[ChatCompletionChunkChoice]


# ── Text completion (legacy) ────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str
    prompt: str
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    think: bool | None = None  # ppmlx extension: force enable/disable thinking
    reasoning_budget: int | None = Field(default=None, ge=1, le=131072)  # ppmlx extension: max reasoning tokens


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str | None = "stop"
    logprobs: None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=make_completion_id)
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=now_ts)
    model: str
    choices: list[CompletionChoice]
    usage: Usage


# ── Embeddings ──────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"

    @field_validator("input")
    @classmethod
    def batch_size_limit(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, list) and len(v) > 2048:
            raise ValueError("batch size must not exceed 2048")
        return v


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# ── Model list ──────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=now_ts)
    owned_by: str = "ppmlx"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ── Errors ──────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ── Tool calling (agent) ─────────────────────────────────────────────────

class FunctionParameters(BaseModel):
    """JSON Schema for function parameters."""
    type: Literal["object"] = "object"
    properties: dict = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    """OpenAI-compatible function definition."""
    name: str
    description: str = ""
    parameters: FunctionParameters = Field(default_factory=FunctionParameters)


class ToolDefinition(BaseModel):
    """OpenAI-compatible tool definition."""
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolCall(BaseModel):
    """A tool call parsed from model output."""
    name: str
    arguments: str  # JSON string


class ToolMessage(BaseModel):
    """Result of a tool execution, fed back to the model."""
    role: Literal["tool"] = "tool"
    name: str
    tool_call_id: str
    content: str
