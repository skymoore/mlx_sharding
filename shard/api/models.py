from pydantic import BaseModel
from typing import Optional, Union, List, Any, Dict

from shard.api.tool_calling import ToolDefinition


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.0
    repetition_context_size: Optional[int] = 20
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.0
    repetition_context_size: Optional[int] = 20


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mlx-sharding-v2"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
