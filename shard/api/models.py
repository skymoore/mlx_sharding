from pydantic import BaseModel
from typing import Optional, Union, List, Any, Dict, overload, TypeVar
from dataclasses import dataclass

T = TypeVar("T", bound=Union["ChatCompletionRequest", "CompletionRequest"])


@dataclass
class GenerationDefaults:
    """Default values for generation parameters, configurable at server startup."""

    temperature: float = 0.7
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_tokens: int = 2048
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mlx-sharding-v2"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


def apply_generation_defaults(request: T, defaults: GenerationDefaults) -> T:
    """
    Apply server-configured defaults to request parameters that are None.

    This allows the API server to override the Pydantic model defaults at startup
    without modifying the model classes themselves.

    Args:
        request: The incoming request with potentially None values
        defaults: Server-configured default values

    Returns:
        The request with defaults applied (modifies in place and returns same type)
    """
    if request.temperature is None:
        request.temperature = defaults.temperature
    if request.top_p is None:
        request.top_p = defaults.top_p
    if request.top_k is None:
        request.top_k = defaults.top_k
    if request.max_tokens is None:
        request.max_tokens = defaults.max_tokens
    if request.repetition_penalty is None:
        request.repetition_penalty = defaults.repetition_penalty
    if request.repetition_context_size is None:
        request.repetition_context_size = defaults.repetition_context_size

    return request
