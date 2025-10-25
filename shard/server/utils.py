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
CHUNK_SIZE_MB = 10
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024

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


def load_model(path_or_hf_repo: str, start_layer: int = None, end_layer: int = None):
    from pathlib import Path

    # Check if it's a local path or HuggingFace repo
    if Path(path_or_hf_repo).exists():
        path = Path(path_or_hf_repo)
    else:
        path = hf_repo_to_path(path_or_hf_repo)
    with open(path / "config.json", "r") as f:
        config = json.load(f)
        if start_layer is not None and end_layer is not None:
            config["start_layer"] = start_layer
            config["end_layer"] = end_layer
    weight_files = glob.glob(str(path / "*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:

        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )
    model.load_weights(list(weights.items()))
    model.eval()
    return model


def send_tensor(stub, tensor: mx.array, session_id: str = None):
    """Send tensor, automatically chunking if needed."""
    tensor_bytes = tensor_to_bytes(tensor)
    message_size_mb = len(tensor_bytes) / (1024 * 1024)

    # Small tensor - send directly (backward compatible)
    if len(tensor_bytes) < CHUNK_SIZE_BYTES:
        logger.info(
            f"📤 Sending tensor: shape={tensor.shape}, size={message_size_mb:.2f}MB (direct), session={session_id}"
        )

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

        logger.info(
            f"📦 Chunking tensor: shape={tensor.shape}, size={message_size_mb:.2f}MB into {total_chunks} chunks"
        )

        for chunk_idx in range(total_chunks):
            start = chunk_idx * CHUNK_SIZE_BYTES
            end = min(start + CHUNK_SIZE_BYTES, len(tensor_bytes))
            chunk_data = tensor_bytes[start:end]
            chunk_size_mb = len(chunk_data) / (1024 * 1024)

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
                logger.info(
                    f"  📤 Sending chunk {chunk_idx+1}/{total_chunks} ({chunk_size_mb:.2f}MB)"
                )
                response = stub.SendTensor(request)

                # Only the last chunk returns the processed tensor
                if chunk_idx == total_chunks - 1:
                    logger.info(f"✅ All chunks sent successfully")
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


def create_coordinator_generate_step(grpc_stubs: List):
    """
    Create generation function for coordinator-only mode (no local model).
    Coordinator sends tokens through pipeline and samples from returned logits.

    Pipeline flow:
    1. Coordinator sends token IDs (int32) to first peer
    2. First peer embeds tokens and processes through its layers
    3. Each subsequent peer processes hidden states through its layers
    4. Last peer returns logits to coordinator
    5. Coordinator samples next token and repeats

    Args:
        grpc_stubs: Ordered list of gRPC stubs (by layer range)

    Returns:
        Generator function that yields (token, logprobs) tuples
    """

    def generate_step(
        prompt: mx.array,  # Token IDs from tokenizer
        temp: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:

        # Generate unique session ID for this generation
        session_id = str(uuid.uuid4())
        logger.info(f"🆔 Starting generation with session_id: {session_id}")

        # Reset all peer caches at start of generation
        for stub in grpc_stubs:
            reset_response = stub.ResetCache(
                mlx_tensor_pb2.ResetCacheRequest(session_id=session_id)
            )
            logger.debug(f"ResetCache Response: {reset_response.message}")

        def sample(logits: mx.array) -> Tuple[mx.array, mx.array]:
            """Sample next token from logits."""
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            if temp == 0:
                token = mx.argmax(logits, axis=-1)
            else:
                if top_p > 0 and top_p < 1.0:
                    modified_logprobs = apply_top_p(logprobs, top_p)
                    token = mx.random.categorical(modified_logprobs * (1 / temp))
                else:
                    token = mx.random.categorical(logits * (1 / temp))
            return token, logprobs

        # Initialize
        y = prompt  # Token IDs (int32)
        repetition_context = prompt.tolist()
        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]

        # Create logits processors
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        def _step(y):
            """Process one generation step through the pipeline."""
            nonlocal repetition_context

            # Ensure y is int32 token IDs with shape (batch, seq_len)
            if y.dtype != mx.int32:
                y = y.astype(mx.int32)
            if y.ndim == 0:  # scalar
                y = y.reshape(1, 1)
            elif y.ndim == 1:  # (seq_len,)
                y = y.reshape(1, -1)

            logger.info(
                f"🎯 Coordinator sending to pipeline: shape={y.shape}, dtype={y.dtype}"
            )

            # Send through pipeline
            tensor = y
            for i, stub in enumerate(grpc_stubs):
                logger.info(
                    f"  → Sending to peer {i}: shape={tensor.shape}, dtype={tensor.dtype}"
                )
                response = send_tensor(stub, tensor, session_id=session_id)
                tensor = response_to_mlx_array(response)
                if tensor is None:
                    raise ValueError(f"Peer {i} returned None")
                logger.info(
                    f"  ← Received from peer {i}: shape={tensor.shape}, dtype={tensor.dtype}"
                )

            # tensor is now logits from last peer
            logger.info(
                f"🎯 Final tensor from pipeline: shape={tensor.shape}, dtype={tensor.dtype}"
            )
            logits = tensor[:, -1, :]
            logger.info(f"🎯 Extracted logits: shape={logits.shape}")
            logger.info(
                f"🔍 Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}, std={logits.std().item():.4f}"
            )
            logger.info(
                f"🔍 Has NaN: {mx.isnan(logits).any().item()}, Has Inf: {mx.isinf(logits).any().item()}"
            )
            logger.info(f"🔍 Top 5 token IDs: {mx.argsort(logits[0])[-5:].tolist()}")
            logger.info(f"🔍 Bottom 5 token IDs: {mx.argsort(logits[0])[:5].tolist()}")

            # Apply logits processors (repetition penalty, logit bias, etc.)
            for processor in logits_processors:
                logits = processor(mx.array(repetition_context), logits)

            # Sample next token
            token, logprobs = sample(logits)
            logger.info(f"🎯 Sampled token: {token.item()}")
            if repetition_penalty:
                repetition_context.append(token.item())
            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:
                    repetition_context = repetition_context[-repetition_context_size:]

            return token, logprobs.squeeze(0)

        # Generate tokens
        y, logprobs = _step(y)
        mx.async_eval(y)
        while True:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y)
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs

    return generate_step

