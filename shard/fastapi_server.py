"""
FastAPI-based OpenAI-compatible API server for MLX Sharding.
A cleaner, more robust alternative to the built-in HTTP server.
"""
import os
import argparse
import logging
import time
import uuid
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

# Suppress verbose gRPC error logs (especially "Message too long" warnings)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import grpc

from .grpc import mlx_tensor_pb2_grpc
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import hf_repo_to_path
from .utils import create_generate_step_with_grpc, load_model


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 2048  # Increased for reasoning models with <think> blocks
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.0
    repetition_context_size: Optional[int] = 20


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 2048  # Increased for reasoning models
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mlx-sharding"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class OpenWebUIModelList(BaseModel):
    models: List[ModelInfo]


# Global state
app = FastAPI(title="MLX Sharding OpenAI API", version="1.0.0")
model_provider = None


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MLXModelProvider:
    """Manages model loading and generation."""
    
    def __init__(self, model_path: str, start_layer: Optional[int], end_layer: Optional[int], grpc_stubs: List):
        self.model_path = model_path
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.grpc_stubs = grpc_stubs
        
        # Load model
        logging.info(f"Loading model from {model_path}")
        self.model = load_model(model_path, start_layer=start_layer, end_layer=end_layer)
        
        # Load tokenizer
        tokenizer_path = Path(model_path) if Path(model_path).exists() else hf_repo_to_path(model_path)
        self.tokenizer = load_tokenizer(tokenizer_path)
        
        # Create generate function
        self.generate_step = create_generate_step_with_grpc(grpc_stubs)
        
        # Model info
        self.model_name = Path(model_path).name if Path(model_path).exists() else model_path
        self.created = int(time.time())
        
        logging.info(f"Model loaded: {self.model_name}")
    
    def get_default_stop_sequences(self) -> List[str]:
        """Get model-specific default stop sequences."""
        model_type = getattr(self.model, 'model_type', '')
        
        # Model-specific stop sequences
        if model_type == 'qwen3_moe' or model_type.startswith('qwen'):
            return ["<|im_end|>", "<|endoftext|>"]
        elif model_type == 'glm4_moe' or model_type.startswith('glm'):
            return ["<|user|>", "<|endoftext|>", "<|observation|>"]
        else:
            # Generic stop sequences
            return ["<|endoftext|>"]
    
    def generate(self, prompt: mx.array, **kwargs):
        """Generate tokens using the model."""
        return self.generate_step(
            prompt=prompt,
            model=self.model,
            temp=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            repetition_penalty=kwargs.get('repetition_penalty', 1.0),
            repetition_context_size=kwargs.get('repetition_context_size', 20),
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> ModelList:
    """List available models (OpenAI format)."""
    return ModelList(
        data=[
            ModelInfo(
                id=model_provider.model_name,
                created=model_provider.created,
            )
        ]
    )


@app.get("/api/models")
async def list_models_openwebui() -> OpenWebUIModelList:
    """List available models (Open WebUI format)."""
    return OpenWebUIModelList(
        models=[
            ModelInfo(
                id=model_provider.model_name,
                created=model_provider.created,
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    try:
        # Apply chat template
        if hasattr(model_provider.tokenizer, "apply_chat_template"):
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            prompt = model_provider.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            # Fallback: simple concatenation
            prompt_text = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
            prompt = model_provider.tokenizer.encode(prompt_text)
        
        prompt_array = mx.array(prompt)
        
        # Handle streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(request, prompt_array),
                media_type="text/event-stream"
            )
        else:
            return await generate_chat_completion(request, prompt_array)
    
    except Exception as e:
        logging.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Handle text completion requests."""
    try:
        prompt = model_provider.tokenizer.encode(request.prompt)
        prompt_array = mx.array(prompt)
        
        if request.stream:
            return StreamingResponse(
                stream_completion(request, prompt_array),
                media_type="text/event-stream"
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


async def generate_chat_completion(request: ChatCompletionRequest, prompt: mx.array) -> Dict[str, Any]:
    """Generate non-streaming chat completion."""
    tokens = []
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()
    
    # Prepare stop sequences
    stop_sequences = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_sequences = [request.stop]
        else:
            stop_sequences = request.stop
    
    # Add model-specific default stop sequences
    stop_sequences.extend(model_provider.get_default_stop_sequences())
    
    finish_reason = "length"
    
    for (token, _), _ in zip(
        model_provider.generate(prompt, temperature=request.temperature, top_p=request.top_p),
        range(request.max_tokens)
    ):
        tokens.append(token)
        detokenizer.add_token(token)
        
        # Check for EOS token
        if token == model_provider.tokenizer.eos_token_id:
            finish_reason = "stop"
            break
        
        # Check for stop sequences in generated text
        current_text = detokenizer.text
        stop_found, trimmed_text = check_stop_sequences(current_text, stop_sequences)
        if stop_found:
            finish_reason = "stop"
            break
    
    detokenizer.finalize()
    text = detokenizer.text
    
    # Final check and trim stop sequences
    _, text = check_stop_sequences(text, stop_sequences)
    
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
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


async def stream_chat_completion(request: ChatCompletionRequest, prompt: mx.array):
    """Generate streaming chat completion."""
    import json
    import asyncio
    
    request_id = f"chatcmpl-{uuid.uuid4()}"
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()
    
    # Prepare stop sequences
    stop_sequences = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_sequences = [request.stop]
        else:
            stop_sequences = request.stop
    stop_sequences.extend(model_provider.get_default_stop_sequences())
    
    finish_reason = "length"
    
    try:
        for (token, _), _ in zip(
            model_provider.generate(prompt, temperature=request.temperature, top_p=request.top_p),
            range(request.max_tokens)
        ):
            detokenizer.add_token(token)
            
            # Check for EOS
            if token == model_provider.tokenizer.eos_token_id:
                finish_reason = "stop"
                break
            
            # Check for stop sequences in full text
            current_text = detokenizer.text
            stop_found, trimmed_text = check_stop_sequences(current_text, stop_sequences)
            if stop_found:
                finish_reason = "stop"
                # Calculate what part of the trimmed text we haven't sent yet
                # This is tricky with streaming, so we just stop here
                break
            
            # Get the segment to send
            text = detokenizer.last_segment
            
            # Check if this segment contains a stop sequence
            if text:
                segment_stop_found, trimmed_segment = check_stop_sequences(text, stop_sequences)
                if segment_stop_found:
                    # Send only the part before the stop sequence
                    if trimmed_segment:
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {"content": trimmed_segment}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    finish_reason = "stop"
                    break
                
                # Send the full segment
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.001)  # Small delay to prevent socket overflow
        
        # Send final chunk
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    except GeneratorExit:
        # Client disconnected (user pressed stop button)
        logging.info(f"Client disconnected during streaming (request {request_id})")
        raise  # Re-raise to properly close the generator
    
    except Exception as e:
        logging.error(f"Error in streaming: {e}", exc_info=True)
        # Send error chunk
        try:
            error_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            pass  # Client already disconnected


async def generate_completion(request: CompletionRequest, prompt: mx.array) -> Dict[str, Any]:
    """Generate non-streaming completion."""
    tokens = []
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()
    
    for (token, _), _ in zip(
        model_provider.generate(prompt, temperature=request.temperature, top_p=request.top_p),
        range(request.max_tokens)
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
            model_provider.generate(prompt, temperature=request.temperature, top_p=request.top_p),
            range(request.max_tokens)
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
                await asyncio.sleep(0.001)  # Small delay to prevent socket overflow
            
            if token == model_provider.tokenizer.eos_token_id:
                break
        
        yield "data: [DONE]\n\n"
    
    except GeneratorExit:
        # Client disconnected (user pressed stop button)
        logging.info(f"Client disconnected during streaming (request {request_id})")
        raise  # Re-raise to properly close the generator
    
    except Exception as e:
        logging.error(f"Error in streaming: {e}", exc_info=True)
        try:
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            pass  # Client already disconnected


def main():
    import signal
    import sys
    
    parser = argparse.ArgumentParser(description="FastAPI OpenAI-compatible API server for MLX Sharding")
    parser.add_argument("--model", type=str, required=True, help="Path to the MLX model")
    parser.add_argument("--start-layer", type=int, default=None, help="Start layer for local model")
    parser.add_argument("--end-layer", type=int, default=None, help="End layer for local model")
    parser.add_argument("-s", "--llm-shard-addresses", type=str, default="", 
                        help="Comma-separated list of REMOTE gRPC shard addresses")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--cache-limit-gb", type=int, default=None, help="MLX cache limit in GB")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Set cache limit
    if args.cache_limit_gb:
        mx.metal.set_cache_limit(args.cache_limit_gb * 1024 * 1024 * 1024)
    
    # Connect to remote shards
    grpc_stubs = []
    if args.llm_shard_addresses:
        channel_options = [
            ('grpc.max_metadata_size', 64 * 1024 * 1024),  # 64MB metadata
            ('grpc.max_send_message_length', 4 * 1024 * 1024 * 1024),  # 4GB send
            ('grpc.max_receive_message_length', 4 * 1024 * 1024 * 1024),  # 4GB receive
            ('grpc.http2.max_frame_size', 16 * 1024 * 1024),  # 16MB frames
            ('grpc.http2.min_recv_ping_interval_without_data_ms', 300000),  # 5 minutes
        ]
        
        for addr in args.llm_shard_addresses.split(','):
            addr = addr.strip()
            if addr:
                channel = grpc.insecure_channel(addr, options=channel_options)
                stub = mlx_tensor_pb2_grpc.MLXTensorServiceStub(channel)
                grpc_stubs.append(stub)
                logging.info(f"Connected to remote shard: {addr}")
    
    # Initialize model provider
    global model_provider
    model_provider = MLXModelProvider(
        args.model,
        args.start_layer,
        args.end_layer,
        grpc_stubs
    )
    
    # Setup signal handlers for clean shutdown
    def signal_handler(sig, frame):
        logging.info("\nShutting down server...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server
    logging.info(f"Starting FastAPI server on {args.host}:{args.port}")
    logging.info(f"OpenAI API endpoint: http://{args.host}:{args.port}/v1")
    logging.info(f"Open WebUI endpoint: http://{args.host}:{args.port}/api/models")
    logging.info(f"Health check: http://{args.host}:{args.port}/health")
    logging.info("Press Ctrl+C to stop")
    
    try:
        uvicorn.run(
            app, 
            host=args.host, 
            port=args.port, 
            log_level=args.log_level.lower(),
            access_log=False  # Reduce noise
        )
    except KeyboardInterrupt:
        logging.info("\nServer stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
