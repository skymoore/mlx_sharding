"""
FastAPI V2 - Zero-Configuration Distributed Inference API Server

This server:
1. Runs the orchestrator to discover peers and distribute model
2. Starts the FastAPI server after setup is complete
3. Routes inference requests through the distributed system
"""

import os
import json
import logging
import time
import uuid
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path

# Suppress verbose gRPC error logs
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

import mlx.core as mx
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import grpc

from shard.grpc import mlx_tensor_pb2_grpc
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import hf_repo_to_path
from shard.server.utils import (
    create_generate_step_with_grpc,
    create_coordinator_generate_step,
    load_model,
)
from shard.api.tool_calling import (
    ToolDefinition,
    ToolCall,
    ToolCallManager,
    create_tool_call_manager,
)

from shard.orchestrator.orchestrator import APIServerOrchestrator as Orchestrator

# Setup logging
logger = logging.getLogger(__name__)


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


class OpenWebUIModelList(BaseModel):
    models: List[ModelInfo]


# Global state
app = FastAPI(title="MLX Sharding V2 API", version="2.0.0")
model_provider = None
api_keys: set[str] = set()
setup_complete = False
setup_info: Dict[str, Any] = {}
orchestrator_instance = None  # Store orchestrator for cleanup
discovered_peers = []  # Store peers for unclaiming

# Security
security = HTTPBearer(auto_error=False)


class GRPCConnectionPool:
    """
    Connection pool for creating isolated gRPC stubs per request.
    Enables true concurrency by preventing shared state between requests.
    """

    def __init__(self, peer_addresses: List[Tuple[str, int]]):
        """
        Initialize connection pool with peer addresses.

        Args:
            peer_addresses: List of (host, port) tuples for each peer
        """
        self.peer_addresses = peer_addresses
        self.channel_options = [
            ("grpc.max_metadata_size", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
            ("grpc.http2.max_frame_size", 4 * 1024 * 1024),
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 300000),
        ]
        logger.info(f"✓ Connection pool initialized with {len(peer_addresses)} peers")

    def create_stubs_for_request(self) -> Tuple[List, List]:
        """
        Create fresh gRPC stubs for a single request.
        Each request gets isolated channels to prevent concurrent interference.

        Returns:
            Tuple of (stubs, channels) - caller must close channels after use
        """
        stubs = []
        channels = []
        for host, port in self.peer_addresses:
            channel = grpc.insecure_channel(
                f"{host}:{port}", options=self.channel_options
            )
            stub = mlx_tensor_pb2_grpc.MLXTensorServiceStub(channel)
            stubs.append(stub)
            channels.append(channel)
        return stubs, channels


def load_api_keys() -> set[str]:
    """Load API keys from environment variable or file."""
    keys = set()

    # Load from environment variable (comma-separated)
    env_keys = os.environ.get("MLX_API_KEYS", "")
    if env_keys:
        keys.update(k.strip() for k in env_keys.split(",") if k.strip())

    # Load from file (one key per line)
    api_key_file = os.environ.get("MLX_API_KEY_FILE", ".api_keys")
    if os.path.exists(api_key_file):
        with open(api_key_file, "r") as f:
            keys.update(
                line.strip() for line in f if line.strip() and not line.startswith("#")
            )

    return keys


async def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> bool:
    """Verify API key from Authorization header."""
    # Skip authentication for health check and setup status
    if request.url.path in ["/health", "/v1/setup/status"]:
        return True

    # If no API keys configured, allow all requests
    if not api_keys:
        logging.warning("No API keys configured - authentication disabled")
        return True

    # Check for valid credentials
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify the API key
    if credentials.credentials not in api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MLXModelProvider:
    """Manages model loading and generation with distributed inference."""

    def __init__(
        self,
        model_path: str,
        start_layer: Optional[int],
        end_layer: Optional[int],
        connection_pool: Optional[GRPCConnectionPool] = None,
        custom_chat_template: Optional[str] = None,
    ):
        self.model_path = model_path
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.connection_pool = connection_pool

        # Load tokenizer (always needed)
        tokenizer_path = (
            Path(model_path)
            if Path(model_path).exists()
            else hf_repo_to_path(model_path)
        )
        self.tokenizer = load_tokenizer(tokenizer_path)

        # Coordinator mode: no local layers
        if start_layer is None and end_layer is None:
            logging.info(
                "Coordinator-only mode: no local model, pipeline coordination only"
            )
            self.model = None
            # Don't create generate_step here - will be created per-request

            # Get model type from config for tool calling and stop tokens
            config_path = tokenizer_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    self.model_type = config.get("model_type", "unknown")
            else:
                self.model_type = "unknown"
        else:
            # Peer mode: load local layers (not used in coordinator-only mode)
            logging.info(f"Peer mode: loading layers {start_layer}-{end_layer}")
            self.model, model_config = load_model(
                model_path, start_layer=start_layer, end_layer=end_layer
            )
            # For peer mode, would need stubs - not implemented in this coordinator-only setup
            self.model_type = model_config.get("model_type", "unknown")

        # Model info
        self.model_name = (
            Path(model_path).name if Path(model_path).exists() else model_path
        )
        self.created = int(time.time())

        # Initialize tool call manager
        self.tool_manager = create_tool_call_manager(self.model_type)
        logging.info(f"✓ Model provider initialized: {self.model_name}")
        logging.info(f"✓ Model type: {self.model_type}")
        logging.info(
            f"✓ Tool calling enabled with {self.tool_manager.parser.__class__.__name__}"
        )
        
        # Chat templates are handled by the tokenizer's built-in apply_chat_template()
        logging.info(f"✓ Chat template support: {hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None}")

    def get_stop_token_ids(self) -> set:
        """
        Get token IDs for stop sequences to check during generation.
        Uses the tokenizer's built-in eos_token_ids which is the correct way
        to determine when generation should stop.
        """
        # Use tokenizer's eos_token_ids directly - this is what mlx_lm does
        stop_token_ids = set(self.tokenizer.eos_token_ids)
        
        logging.debug(f"🛑 Using tokenizer.eos_token_ids: {stop_token_ids}")
        
        # Decode for debugging
        for token_id in stop_token_ids:
            try:
                decoded = self.tokenizer.decode([token_id])
                logging.debug(f"🛑 Stop token {token_id}: {repr(decoded)}")
            except:
                pass
        
        return stop_token_ids

    def generate(self, prompt: mx.array, **kwargs):
        """
        Generate tokens using the distributed model.
        Creates fresh gRPC stubs per request for true concurrency.
        """
        # Get fresh stubs from connection pool for this request
        if self.connection_pool is None:
            raise ValueError("Connection pool not initialized")
        
        grpc_stubs, channels = self.connection_pool.create_stubs_for_request()
        
        # Create generate_step function with fresh stubs
        generate_step = create_coordinator_generate_step(grpc_stubs)
        
        # Wrap the generator to close channels after exhaustion
        def generator_with_cleanup():
            try:
                # Yield from the actual generator
                for item in generate_step(
                    prompt=prompt,
                    temp=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 1.0),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                    repetition_context_size=kwargs.get("repetition_context_size", 20),
                ):
                    yield item
            finally:
                # Clean up channels after generator is exhausted or interrupted
                for channel in channels:
                    try:
                        channel.close()
                    except Exception as e:
                        logging.warning(f"Error closing channel: {e}")
        
        return generator_with_cleanup()



@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "setup_complete": setup_complete, "version": "2.0.0"}


@app.get("/v1/setup/status")
async def get_setup_status():
    """Get current setup status and distributed system info."""
    return {
        "setup_complete": setup_complete,
        "setup_info": setup_info,
        "timestamp": time.time(),
    }


@app.get("/v1/models")
async def list_models(
    request: Request, authenticated: bool = Security(verify_api_key)
) -> ModelList:
    """List available models (OpenAI format)."""
    if not setup_complete:
        raise HTTPException(status_code=503, detail="Setup not complete")

    logger.info(f"📋 /v1/models called - returning model: {model_provider.model_name}")
    return ModelList(
        data=[
            ModelInfo(
                id=model_provider.model_name,
                created=model_provider.created,
            )
        ]
    )


@app.get("/api/models")
async def list_models_openwebui(
    request: Request, authenticated: bool = Security(verify_api_key)
) -> OpenWebUIModelList:
    """List available models (Open WebUI format)."""
    if not setup_complete:
        raise HTTPException(status_code=503, detail="Setup not complete")

    return OpenWebUIModelList(
        models=[
            ModelInfo(
                id=model_provider.model_name,
                created=model_provider.created,
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    authenticated: bool = Security(verify_api_key),
):
    """Handle chat completion requests with tool calling support."""
    import json

    # Log the complete incoming request from OpenWebUI
    logger.info("=" * 80)
    logger.info("📥 INCOMING REQUEST TO /v1/chat/completions")
    logger.info("=" * 80)
    logger.info(f"Request JSON: {json.dumps(request.dict(), indent=2)}")
    logger.info(f"Model requested: {request.model}")
    logger.info(f"Number of messages: {len(request.messages)}")
    logger.info(f"Stream: {request.stream}")
    logger.info(f"Temperature: {request.temperature}")
    logger.info(f"Max tokens: {request.max_tokens}")
    logger.info("Messages:")
    for i, msg in enumerate(request.messages):
        logger.info(f"  [{i}] {msg.role}: {msg.content[:100]}...")
    logger.info("=" * 80)

    if not setup_complete:
        raise HTTPException(status_code=503, detail="Setup not complete")

    try:
        # Prepare messages
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Add tool descriptions to system message if tools provided
        if request.tools:
            tool_prompt = model_provider.tool_manager.format_tools_for_prompt(
                request.tools
            )
            # Find or create system message
            system_msg_idx = next(
                (i for i, m in enumerate(messages) if m["role"] == "system"), None
            )
            if system_msg_idx is not None:
                messages[system_msg_idx]["content"] += tool_prompt
            else:
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant.{tool_prompt}",
                    },
                )

        # Apply chat template using tokenizer's built-in support
        # The tokenizer handles system message compatibility automatically
        prompt = model_provider.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        prompt_array = mx.array(prompt)

        # Handle streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(request, prompt_array),
                media_type="text/event-stream",
            )
        else:
            return await generate_chat_completion(request, prompt_array)

    except Exception as e:
        logging.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    http_request: Request,
    authenticated: bool = Security(verify_api_key),
):
    """Handle text completion requests."""
    if not setup_complete:
        raise HTTPException(status_code=503, detail="Setup not complete")

    try:
        prompt = model_provider.tokenizer.encode(request.prompt)
        prompt_array = mx.array(prompt)

        if request.stream:
            return StreamingResponse(
                stream_completion(request, prompt_array), media_type="text/event-stream"
            )
        else:
            return await generate_completion(request, prompt_array)

    except Exception as e:
        logging.error(f"Error in completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def check_stop_sequences(text: str, stop_sequences: List[str]) -> tuple[bool, str]:
    """Check if text contains any stop sequences and trim if found."""
    for stop_seq in stop_sequences:
        if stop_seq in text:
            # Find the position and trim
            pos = text.find(stop_seq)
            return True, text[:pos]
    return False, text


async def generate_chat_completion(
    request: ChatCompletionRequest, prompt: mx.array
) -> Dict[str, Any]:
    """Generate non-streaming chat completion."""
    tokens = []
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()

    # Prepare stop sequences (user-provided strings only)
    stop_sequences = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_sequences = [request.stop]
        else:
            stop_sequences = request.stop

    # Get stop token IDs from tokenizer (this is what mlx_lm uses)
    stop_token_ids = model_provider.get_stop_token_ids()

    finish_reason = "length"

    for (token, _), n in zip(
        model_provider.generate(
            prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
        ),
        range(request.max_tokens),
    ):
        tokens.append(token)
        detokenizer.add_token(token)

        # Check for stop tokens by ID (faster and more reliable)
        # Convert MLX array to Python int for comparison
        token_id = int(token.item()) if hasattr(token, "item") else int(token)
        if token_id in stop_token_ids:
            logging.debug(f"Stop token detected: {token_id} in {stop_token_ids}")
            finish_reason = "stop"
            break

        # Debug: Log first few tokens to verify token IDs
        if len(tokens) <= 5 or token_id in [151336, 151337, 151338, 151329]:
            logging.debug(f"Token {len(tokens)}: {token_id}")

        # Also check for stop sequences in generated text (for multi-token stops)
        current_text = detokenizer.text
        stop_found, trimmed_text = check_stop_sequences(current_text, stop_sequences)
        if stop_found:
            finish_reason = "stop"
            break
        
        # 🔥 NEW: Periodic cache clearing to prevent memory accumulation
        if n % 256 == 0:
            mx.clear_cache()

    detokenizer.finalize()
    text = detokenizer.text

    # Final check and trim stop sequences
    _, text = check_stop_sequences(text, stop_sequences)

    # Parse tool calls if tools were provided
    cleaned_content = text
    tool_calls = []
    if request.tools:
        cleaned_content, tool_calls = model_provider.tool_manager.parse(text)

    # Build response
    response = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": cleaned_content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt) + len(tokens),
        },
    }

    # Add tool_calls if any were found
    if tool_calls:
        response["choices"][0]["message"]["tool_calls"] = [
            {"id": tc.id, "type": tc.type, "function": tc.function} for tc in tool_calls
        ]
        # When tool calls present, finish_reason should be "tool_calls"
        response["choices"][0]["finish_reason"] = "tool_calls"

    return response


async def stream_chat_completion(request: ChatCompletionRequest, prompt: mx.array):
    """Generate streaming chat completion with tool call support."""
    import json
    import asyncio

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()

    # Create streaming parser if tools provided
    streaming_parser = None
    if request.tools:
        streaming_parser = model_provider.tool_manager.create_streaming_parser()

    # Prepare stop sequences (user-provided strings only)
    stop_sequences = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_sequences = [request.stop]
        else:
            stop_sequences = request.stop

    # Get stop token IDs from tokenizer (this is what mlx_lm uses)
    stop_token_ids = model_provider.get_stop_token_ids()

    finish_reason = "length"

    try:
        for (token, _), n in zip(
            model_provider.generate(
                prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
            ),
            range(request.max_tokens),
        ):
            detokenizer.add_token(token)

            # Check for stop tokens by ID (faster and more reliable)
            # Convert MLX array to Python int for comparison
            token_id = int(token.item()) if hasattr(token, "item") else int(token)
            if token_id in stop_token_ids:
                finish_reason = "stop"
                break

            # Also check for stop sequences in full text (for multi-token stops)
            current_text = detokenizer.text
            stop_found, trimmed_text = check_stop_sequences(
                current_text, stop_sequences
            )
            if stop_found:
                finish_reason = "stop"
                break

            # Get the segment to send
            text = detokenizer.last_segment
            
            # 🔥 NEW: Periodic cache clearing to prevent memory accumulation
            if n % 256 == 0:
                mx.clear_cache()

            # Check if this segment contains a stop sequence
            if text:
                segment_stop_found, trimmed_segment = check_stop_sequences(
                    text, stop_sequences
                )
                if segment_stop_found:
                    finish_reason = "stop"
                    text = trimmed_segment

                # Parse for tool calls if enabled
                if streaming_parser and text:
                    content_to_emit, new_tool_calls = streaming_parser.add_chunk(text)

                    # Emit content chunk if any
                    if content_to_emit:
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content_to_emit},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # Emit tool call chunks if any
                    for tool_call in new_tool_calls:
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": tool_call.id,
                                                "type": "function",
                                                "function": {
                                                    "name": tool_call.function["name"],
                                                    "arguments": tool_call.function[
                                                        "arguments"
                                                    ],
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                elif text:
                    # No tool parsing, send text directly
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                await asyncio.sleep(0.001)  # Small delay to prevent socket overflow

                if segment_stop_found:
                    break

        # Finalize - get any remaining content
        if streaming_parser:
            remaining_content, remaining_tool_calls = streaming_parser.finalize()

            if remaining_content:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": remaining_content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            for tool_call in remaining_tool_calls:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.function["name"],
                                            "arguments": tool_call.function[
                                                "arguments"
                                            ],
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Update finish_reason if tool calls were emitted
            if streaming_parser.emitted_tool_calls:
                finish_reason = "tool_calls"

        # Send final chunk
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except GeneratorExit:
        # Client disconnected
        logging.info(f"Client disconnected during streaming (request {request_id})")
        raise

    except Exception as e:
        logging.error(f"Error in streaming: {e}", exc_info=True)
        try:
            error_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            pass


async def generate_completion(
    request: CompletionRequest, prompt: mx.array
) -> Dict[str, Any]:
    """Generate non-streaming completion."""
    tokens = []
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()

    for (token, _), _ in zip(
        model_provider.generate(
            prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
        ),
        range(request.max_tokens),
    ):
        tokens.append(token)
        detokenizer.add_token(token)

        if token == model_provider.tokenizer.eos_token_id:
            break

    detokenizer.finalize()
    text = detokenizer.text

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt) + len(tokens),
        },
    }


async def stream_completion(request: CompletionRequest, prompt: mx.array):
    """Generate streaming completion."""
    import json
    import asyncio

    request_id = f"cmpl-{uuid.uuid4()}"
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()

    try:
        for (token, _), _ in zip(
            model_provider.generate(
                prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
            ),
            range(request.max_tokens),
        ):
            detokenizer.add_token(token)
            text = detokenizer.last_segment

            if text:
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "finish_reason": None,
                        }
                    ],
                }

                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.001)

            if token == model_provider.tokenizer.eos_token_id:
                break

        yield "data: [DONE]\n\n"

    except GeneratorExit:
        logging.info(f"Client disconnected during streaming (request {request_id})")
        raise

    except Exception as e:
        logging.error(f"Error in streaming: {e}", exc_info=True)
        try:
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            pass


async def run_orchestrator_setup(
    model_path: str,
    grpc_port: int,
    http_port: int,
    custom_chat_template: Optional[str] = None,
    resource_strategy: str = "fewest-nodes",
) -> Dict[str, Any]:
    """Run the orchestrator setup process (coordinator-only, no local layers)."""
    global setup_complete, setup_info, model_provider, orchestrator_instance, discovered_peers

    logger.info("=" * 80)
    logger.info("🚀 MLX SHARDING V2 - ZERO-CONFIG SETUP (COORDINATOR-ONLY)")
    logger.info("=" * 80)
    logger.info("Note: API server does NOT load model layers")
    logger.info("      Start peer processes separately with: mlx-shard-peer")
    logger.info("=" * 80)

    # Create orchestrator (coordinator-only mode - no local layers)
    orchestrator = Orchestrator(
        model_name=model_path,
        grpc_port=grpc_port,
        http_port=http_port,
        resource_strategy=resource_strategy,
    )
    
    # Store orchestrator globally for cleanup
    orchestrator_instance = orchestrator

    # Run setup
    try:
        plan = await orchestrator.setup()
        
        # Store discovered peers for unclaiming on shutdown
        discovered_peers = list(orchestrator.discovery.peers.values())

        logger.info("=" * 80)
        logger.info("✓ SETUP COMPLETE")
        logger.info(f"  Total layers: {plan.total_layers}")
        logger.info(f"  Shards: {len(plan.shards)}")
        logger.info(f"  Memory required: {plan.total_memory_required_gb:.1f}GB")
        logger.info("=" * 80)

        # Store setup info
        setup_info = {
            "plan": plan.to_dict(),
            "model_path": model_path,
            "coordinator_only": True,
        }

        # Extract peer addresses for connection pool
        peer_addresses = []
        for shard in plan.shards:
            # Get peer info from orchestrator's discovered peers
            peer = next(
                (
                    p
                    for p in orchestrator.discovery.peers.values()
                    if p.id == shard.peer_id
                ),
                None,
            )
            if peer:
                peer_addresses.append((peer.host, peer.grpc_port))
                logger.info(
                    f"✓ Registered peer: {peer.host}:{peer.grpc_port} (layers {shard.start_layer}-{shard.end_layer})"
                )

        # Create connection pool for per-request stub creation
        connection_pool = GRPCConnectionPool(peer_addresses)

        # Initialize model provider (coordinator loads NO layers, only coordinates)
        model_provider = MLXModelProvider(
            model_path=model_path,
            start_layer=None,  # No local layers
            end_layer=None,  # No local layers
            connection_pool=connection_pool,
            custom_chat_template=custom_chat_template,
        )

        setup_complete = True
        logger.info("✓ API server ready - all inference will be distributed to peers")
        return {"plan": plan.to_dict()}

    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        raise


async def shutdown_coordinator():
    """Gracefully shutdown the coordinator and unclaim peers."""
    global orchestrator_instance, discovered_peers
    
    logger.info("=" * 80)
    logger.info("🛑 SHUTTING DOWN COORDINATOR")
    logger.info("=" * 80)
    
    if orchestrator_instance and discovered_peers:
        try:
            await orchestrator_instance.unclaim_peers(discovered_peers)
            logger.info("✓ Peers unclaimed successfully")
        except Exception as e:
            logger.error(f"Error unclaiming peers: {e}", exc_info=True)

    if orchestrator_instance:
        try:
            orchestrator_instance.cleanup()
            logger.info("✓ Orchestrator cleanup complete")
        except Exception as e:
            logger.error(f"Error during orchestrator cleanup: {e}", exc_info=True)

    logger.info("=" * 80)
    logger.info("✓ SHUTDOWN COMPLETE")
    logger.info("=" * 80)


# Entry point moved to shard.cli.commands.api
