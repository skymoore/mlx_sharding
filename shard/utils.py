from importlib import import_module
import glob
import json
import struct
import logging
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Generator, Optional, Tuple, List
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import apply_top_p, make_logits_processors
from mlx_lm.utils import hf_repo_to_path
import numpy as np
from .grpc import mlx_tensor_pb2

# Setup logger for this module
logger = logging.getLogger(__name__)

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


def send_tensor(stub, tensor: mx.array):
    tensor_bytes = tensor_to_bytes(tensor)
    message_size_mb = len(tensor_bytes) / (1024 * 1024)
    logger.info(f"Sending tensor: shape={tensor.shape}, dtype={tensor.dtype}, size={message_size_mb:.2f}MB")
    
    tensor_message = mlx_tensor_pb2.Tensor(
        tensor_data=tensor_bytes, shape=list(tensor.shape), dtype=str(tensor.dtype)
    )
    
    try:
        response = stub.SendTensor(tensor_message)
        return response
    except Exception as e:
        logger.error(f"Failed to send {message_size_mb:.2f}MB tensor: {e}")
        raise


def response_to_mlx_array(response):
    """Convert a TensorResponse protobuf message to an MLX array."""
    try:
        # Check if response is valid
        if not hasattr(response, 'success'):
            logger.error(f"Invalid response object: {type(response)}")
            return None
            
        if not response.success:
            logger.error(f"Error from shard: {response.message}")
            return None
            
        if response.tensor is None:
            logger.error(f"No tensor in response: {response.message}")
            return None
        
        # Debug: log tensor info at DEBUG level
        logger.debug(f"Converting tensor: dtype={response.tensor.dtype}, shape={response.tensor.shape}, data_len={len(response.tensor.tensor_data)}")
        
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
            raise ValueError("Model does not have make_cache() method. Please use a compatible model.")

        repetition_context = prompt.tolist()
        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]
        
        # Create logits processors (including repetition penalty if specified)
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size
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
