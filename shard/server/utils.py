from importlib import import_module
import glob
import json
import logging
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Generator, Optional, Tuple, List
from mlx_lm.sample_utils import apply_top_p, make_logits_processors
from mlx_lm.utils import hf_repo_to_path
import numpy as np
import uuid
from shard.grpc import mlx_tensor_pb2

# Setup logger for this module
logger = logging.getLogger(__name__)

# Configuration
# Reduced from 10MB to 2MB to avoid gRPC "Message too long" errors
CHUNK_SIZE_MB = 2
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024

# 🔥 NEW: Dedicated stream for generation (like mlx_lm)
# This enables better async execution and memory management
generation_stream = mx.new_stream(mx.default_device())

MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
}


def _get_classes(config: dict):
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = import_module(f".model.{model_type}", package="shard.server")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logger.error(msg)
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs


def load_model(
    path_or_hf_repo: str, 
    start_layer: int = None, 
    end_layer: int = None,
    lazy: bool = False,
    strict: bool = True,
):
    """
    Load model with optional layer slicing for distributed inference.
    
    This function extends mlx_lm's load_model with layer slicing capabilities
    for distributed inference across multiple peers.
    
    Args:
        path_or_hf_repo: Local path or HuggingFace repo ID
        start_layer: Starting layer index (inclusive) for this shard
        end_layer: Ending layer index (exclusive) for this shard
        lazy: If False, eval model parameters immediately
        strict: Whether to raise exception if weights don't match
        
    Returns:
        Tuple of (model, config) - model with loaded weights and configuration
    """
    from pathlib import Path

    # Check if it's a local path or HuggingFace repo
    if Path(path_or_hf_repo).exists():
        path = Path(path_or_hf_repo)
    else:
        path = hf_repo_to_path(path_or_hf_repo)
    
    # Load config
    with open(path / "config.json", "r") as f:
        config = json.load(f)
        
    # Add layer slicing to config (our unique feature for distributed inference!)
    if start_layer is not None and end_layer is not None:
        config["start_layer"] = start_layer
        config["end_layer"] = end_layer
        logger.info(f"Loading model shard: layers {start_layer}-{end_layer-1} (inclusive)")
    
    # Load weights
    weight_files = glob.glob(str(path / "model*.safetensors"))
    if not weight_files:
        if strict:
            raise FileNotFoundError(f"No safetensors found in {path}")
        weight_files = []
    
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    
    # Get model classes
    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # Sanitize weights if needed
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Handle quantization
    if (quantization := config.get("quantization", None)) is not None:
        def class_predicate(p, m):
            # Handle custom per layer quantizations
            if p in config.get("quantization", {}):
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )
    
    # Load weights into model
    model.load_weights(list(weights.items()), strict=strict)
    
    # Evaluate parameters if not lazy
    if not lazy:
        mx.eval(model.parameters())
    
    model.eval()
    
    return model, config


def send_tensor(stub, tensor: mx.array, session_id: str = None):
    """Send tensor, automatically chunking if needed."""
    import sys
    
    tensor_bytes = tensor_to_bytes(tensor)
    message_size_mb = len(tensor_bytes) / (1024 * 1024)

    # Small tensor - send directly (backward compatible)
    if len(tensor_bytes) < CHUNK_SIZE_BYTES:
        # Print progress indicator
        print(".", end="", flush=True)
        
        tensor_message = mlx_tensor_pb2.Tensor(
            tensor_data=tensor_bytes, shape=list(tensor.shape), dtype=str(tensor.dtype)
        )
        request = mlx_tensor_pb2.SendTensorRequest(
            full_tensor=tensor_message, session_id=session_id or ""
        )

        try:
            response = stub.SendTensor(request)
            return response
        except Exception as e:
            logger.error(f"Failed to send {message_size_mb:.2f}MB tensor: {e}")
            raise

    # Large tensor - chunk it
    else:
        tensor_id = str(uuid.uuid4())
        total_chunks = (len(tensor_bytes) + CHUNK_SIZE_BYTES - 1) // CHUNK_SIZE_BYTES

        for chunk_idx in range(total_chunks):
            # Print progress indicator
            print(".", end="", flush=True)
            
            start = chunk_idx * CHUNK_SIZE_BYTES
            end = min(start + CHUNK_SIZE_BYTES, len(tensor_bytes))
            chunk_data = tensor_bytes[start:end]

            chunk_message = mlx_tensor_pb2.TensorChunk(
                tensor_id=tensor_id,
                chunk_index=chunk_idx,
                total_chunks=total_chunks,
                chunk_data=chunk_data,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
            )
            request = mlx_tensor_pb2.SendTensorRequest(
                chunked_tensor=chunk_message, session_id=session_id or ""
            )

            try:
                response = stub.SendTensor(request)

                # Only the last chunk returns the processed tensor
                if chunk_idx == total_chunks - 1:
                    return response

            except Exception as e:
                logger.error(f"Failed to send chunk {chunk_idx+1}/{total_chunks}: {e}")
                raise


def response_to_mlx_array(response):
    """Convert a TensorResponse protobuf message to an MLX array."""
    try:
        # Check if response is valid
        if not hasattr(response, "success"):
            logger.error(f"Invalid response object: {type(response)}")
            return None

        if not response.success:
            logger.error(f"Error from shard: {response.message}")
            return None

        if response.tensor is None:
            logger.error(f"No tensor in response: {response.message}")
            return None

        # Debug: log tensor info at DEBUG level
        logger.debug(
            f"Converting tensor: dtype={response.tensor.dtype}, shape={response.tensor.shape}, data_len={len(response.tensor.tensor_data)}"
        )

        tensor = bytes_to_tensor(response.tensor.tensor_data, response.tensor.dtype)
        tensor = tensor.reshape(response.tensor.shape)
        return tensor
    except Exception as e:
        logger.error(f"Error converting response to MLX array: {e}", exc_info=True)
        return None


def tensor_to_bytes(tensor):
    """Convert an MLX tensor to bytes."""
    if tensor is None:
        raise ValueError("Cannot convert None to bytes")
    # Ensure tensor is evaluated before converting to bytes
    mx.eval(tensor)
    return bytes(memoryview(tensor))


def bytes_to_tensor(byte_data, dtype_str):
    """Convert bytes to an MLX tensor of the specified dtype."""
    dtype_map = {
        "mlx.core.float32": np.float32,
        "mlx.core.int32": np.int32,
        "mlx.core.int64": np.int64,
        "mlx.core.float16": np.float16,
        "mlx.core.bfloat16": np.uint16,  # bfloat16 stored as uint16 in numpy
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    np_dtype = dtype_map.get(dtype_str, np.float32)
    np_array = np.frombuffer(byte_data, dtype=np_dtype)

    mx_dtype_str = dtype_str.replace("mlx.core.", "")
    mx_dtype = getattr(mx, mx_dtype_str, mx.float32)

    # Special handling for bfloat16: use .view() to reinterpret bits
    if dtype_str == "mlx.core.bfloat16":
        return mx.array(np_array).view(mx.bfloat16)
    else:
        return mx.array(np_array, dtype=mx_dtype)


def create_generate_step_with_grpc(grpc_stubs: List):
    def generate_step(
        prompt: mx.array,
        model: nn.Module,
        temp: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:

        for stub in grpc_stubs:
            reset_response = stub.ResetCache(mlx_tensor_pb2.ResetCacheRequest())
            logger.debug(f"ResetCache Response: {reset_response.message}")

        def sample(logits: mx.array) -> Tuple[mx.array, float]:
            # logit_bias is now handled by logits_processors
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            if temp == 0:
                token = mx.argmax(logits, axis=-1)
            else:
                if top_p > 0 and top_p < 1.0:
                    # apply_top_p modifies logprobs, then sample from them
                    modified_logprobs = apply_top_p(logprobs, top_p)
                    token = mx.random.categorical(modified_logprobs * (1 / temp))
                else:
                    token = mx.random.categorical(logits * (1 / temp))
            return token, logprobs

        y = prompt
        if hasattr(model, "make_cache"):
            cache = model.make_cache()
        else:
            raise ValueError(
                "Model does not have make_cache() method. Please use a compatible model."
            )

        repetition_context = prompt.tolist()
        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]

        # Create logits processors (including repetition penalty if specified)
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        def _step(y):
            nonlocal repetition_context
            # Ensure y has shape (batch, seq_len)
            if y.ndim == 0:  # scalar
                y = y.reshape(1, 1)
            elif y.ndim == 1:  # (seq_len,)
                y = y.reshape(1, -1)
            # else y is already (batch, seq_len)

            output = model(y, cache=cache)
            if output.dtype == mx.bfloat16:
                output = output.astype(mx.float16)

            for stub in grpc_stubs:
                response = send_tensor(stub, output)
                output = response_to_mlx_array(response)
                if output is None:
                    raise ValueError("Shard returned None")

            logits = output[:, -1, :]

            # Apply logits processors (repetition penalty, logit bias, etc.)
            for processor in logits_processors:
                logits = processor(mx.array(repetition_context), logits)

            y, logprobs = sample(logits)
            if repetition_penalty:
                repetition_context.append(y.item())
            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:
                    repetition_context = repetition_context[-repetition_context_size:]
            return y, logprobs.squeeze(0)

        y, logprobs = _step(y)
        mx.async_eval(y)
        while True:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y)
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs

    return generate_step


class PipelineModel(nn.Module):
    """
    Wrapper that makes a distributed gRPC pipeline look like a local model.
    This allows us to use mlx_lm's generate_step directly without reimplementing it.
    """
    def __init__(self, grpc_stubs: List, session_id: str):
        super().__init__()
        self.grpc_stubs = grpc_stubs
        self.session_id = session_id
        # Add empty layers list to satisfy mlx_lm's cache creation
        # Peers manage their own caches, so we don't need a local cache
        self.layers = []
        
    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        """
        Forward pass through the distributed pipeline.
        
        Args:
            inputs: Token IDs with shape (batch, seq_len)
            cache: Ignored - peers manage their own caches
            
        Returns:
            Logits with shape (batch, seq_len, vocab_size)
        """
        # Ensure inputs are int32 token IDs with shape (batch, seq_len)
        if inputs.dtype != mx.int32:
            inputs = inputs.astype(mx.int32)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        # Send through pipeline
        tensor = inputs
        for i, stub in enumerate(self.grpc_stubs):
            response = send_tensor(stub, tensor, session_id=self.session_id)
            tensor = response_to_mlx_array(response)
            if tensor is None:
                raise ValueError(f"Peer {i} returned None")
        
        # tensor is now logits from last peer with shape (batch, seq_len, vocab_size)
        return tensor


def create_coordinator_generate_step(grpc_stubs: List, tokenizer):
    """
    Create generation function for coordinator-only mode (no local model).
    
    This wraps the distributed gRPC pipeline in a PipelineModel and uses
    mlx_lm's stream_generate, ensuring identical behavior to local inference
    INCLUDING proper EOS token detection.

    Pipeline flow:
    1. Coordinator sends token IDs (int32) to first peer
    2. First peer embeds tokens and processes through its layers
    3. Each subsequent peer processes hidden states through its layers
    4. Last peer returns logits to coordinator
    5. mlx_lm's stream_generate handles sampling, EOS detection, and generation loop

    Args:
        grpc_stubs: Ordered list of gRPC stubs (by layer range)
        tokenizer: The tokenizer (needed for EOS token detection)

    Returns:
        Generator function that yields (token, logprobs) tuples
    """
    from mlx_lm.generate import stream_generate
    
    def generate_step(
        prompt: mx.array,
        temp: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: int = 256,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        
        # 🔍 DEBUG: Log generation parameters
        logger.info("=" * 80)
        logger.info("🚀 DISTRIBUTED GENERATION START")
        logger.info("=" * 80)
        logger.info(f"Prompt shape: {prompt.shape}")
        logger.info(f"Prompt tokens: {prompt.tolist() if prompt.size < 100 else f'{prompt.tolist()[:20]}...'}")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"Temperature: {temp}")
        logger.info(f"Top-p: {top_p}")
        logger.info(f"Repetition penalty: {repetition_penalty}")
        
        # 🔍 DEBUG: Log tokenizer EOS configuration
        if hasattr(tokenizer, 'eos_token_ids'):
            logger.info(f"Tokenizer EOS token IDs: {tokenizer.eos_token_ids}")
            for eos_id in tokenizer.eos_token_ids:
                try:
                    decoded = tokenizer.decode([eos_id])
                    logger.info(f"  EOS token {eos_id} decodes to: {repr(decoded)}")
                except:
                    pass
        logger.info("=" * 80)
        
        # Generate unique session ID for this generation
        session_id = str(uuid.uuid4())
        logger.info(f"Session ID: {session_id}")
        
        # Reset all peer caches at start of generation
        for i, stub in enumerate(grpc_stubs):
            reset_response = stub.ResetCache(
                mlx_tensor_pb2.ResetCacheRequest(session_id=session_id)
            )
            logger.debug(f"Peer {i} ResetCache Response: {reset_response.message}")
        
        # Create pipeline model wrapper
        pipeline_model = PipelineModel(grpc_stubs, session_id)
        
        # Create logits processors
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
        
        # Create sampler
        if temp == 0:
            sampler = lambda x: mx.argmax(x, axis=-1)
        else:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temp, top_p=top_p)
        
        # 🔍 DEBUG: Track generation
        token_count = 0
        
        # Use mlx_lm's stream_generate!
        # This includes EOS token detection, which generate_step does NOT have
        for response in stream_generate(
            model=pipeline_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            token_count += 1
            
            # 🔍 DEBUG: Log first 10 tokens and every 50th token
            if token_count <= 10 or token_count % 50 == 0:
                try:
                    decoded = tokenizer.decode([response.token])
                    logger.info(f"Token {token_count}: {response.token} (decoded: {repr(decoded)})")
                except:
                    logger.info(f"Token {token_count}: {response.token}")
            
            # 🔍 DEBUG: Check if this is an EOS token
            if hasattr(tokenizer, 'eos_token_ids') and response.token in tokenizer.eos_token_ids:
                logger.info(f"🛑 EOS token detected at position {token_count}: {response.token}")
                logger.info(f"   This will cause stream_generate to stop")
            
            # stream_generate yields GenerationResponse objects
            # We need to yield (token, logprobs) tuples for compatibility
            yield response.token, response.logprobs
        
        logger.info("=" * 80)
        logger.info(f"🏁 DISTRIBUTED GENERATION END - Generated {token_count} tokens")
        logger.info("=" * 80)
    
    return generate_step

