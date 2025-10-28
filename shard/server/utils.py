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
import pyarrow as pa
import hashlib
import time
from pyarrow import flight

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


def send_tensor(client: flight.FlightClient, tensor: mx.array, session_id: str = None):
    """Send tensor using Arrow Flight, automatically chunking if needed."""
    arrow_tensor, original_dtype = mlx_to_arrow(tensor)
    np_tensor = arrow_tensor.to_numpy()
    total_size = np_tensor.nbytes
    item_size = np_tensor.itemsize
    chunk_items = CHUNK_SIZE_BYTES // item_size
    total_items = np_tensor.size
    total_chunks = (total_items + chunk_items - 1) // chunk_items

    descriptor = flight.FlightDescriptor.for_command("SendTensor")
    writer, reader = client.do_exchange(descriptor)

    # Flatten and compute checksum from the actual bytes we'll send
    flat_np = np_tensor.flatten()
    full_bytes = flat_np.tobytes()
    checksum = hashlib.md5(full_bytes).hexdigest()
    
    # Send metadata in first write (no data)
    # IMPORTANT: Send original_dtype so receiver can reconstruct bfloat16
    meta = {
        "session_id": session_id or "default",
        "total_chunks": total_chunks,
        "shape": list(tensor.shape),
        "dtype": original_dtype,  # Original MLX dtype (e.g., "mlx.core.bfloat16")
        "md5": checksum
    }
    schema = pa.schema([pa.field("chunk", pa.binary())])
    writer.begin(schema)
    
    # Send metadata as first batch WITH chunk 0 data
    if total_chunks > 0:
        start = 0
        end = min(chunk_items * item_size, len(full_bytes))
        chunk0_bytes = full_bytes[start:end]
        meta_batch = pa.RecordBatch.from_arrays([pa.array([chunk0_bytes])], schema=schema)
    else:
        meta_batch = pa.RecordBatch.from_arrays([pa.array([b""])], schema=schema)
    
    writer.write_with_metadata(meta_batch, json.dumps(meta).encode())

    # Send remaining chunks (1 through total_chunks-1)
    for i in range(1, total_chunks):
        start = i * chunk_items * item_size
        end = min(start + chunk_items * item_size, len(full_bytes))
        chunk_bytes = full_bytes[start:end]
        chunk_array = pa.array([chunk_bytes])
        batch = pa.RecordBatch.from_arrays([chunk_array], schema=schema)
        writer.write_with_metadata(batch, json.dumps({"chunk_index": i}).encode())

    writer.done_writing()
    return writer, reader  # Return both so we can send ACK after reading


def response_to_mlx_array(reader: flight.FlightStreamReader):
    """Convert Flight response to MLX array."""
    try:
        # Read metadata batch
        try:
            batch, meta = reader.read_chunk()
        except StopIteration as e:
            logger.error(f"StopIteration when reading first chunk: {e}")
            raise ValueError("No data received from server - stream ended prematurely. The server may have crashed or encountered an error before sending a response.")
        except Exception as e:
            logger.error(f"Exception when reading first chunk: {type(e).__name__}: {e}")
            raise
            
        if meta is None:
            raise ValueError("No metadata in response")
        meta_dict = json.loads(meta.to_pybytes())

        if not meta_dict.get("success"):
            error_msg = meta_dict.get('message', 'Unknown error')
            raise ValueError(f"Error from shard: {error_msg}")

        total_chunks = meta_dict["total_chunks"]
        shape = tuple(meta_dict["shape"])
        dtype_str = meta_dict["dtype"]

        dtype_map = {
            "mlx.core.float32": np.float32,
            "mlx.core.int32": np.int32,
            "mlx.core.int64": np.int64,
            "mlx.core.float16": np.float16,
            "mlx.core.bfloat16": np.uint16,
        }
        np_dtype = dtype_map.get(dtype_str, np.float32)

        # Collect chunks
        # Chunk 0 comes with metadata batch
        chunks = {}
        if batch and batch.num_rows > 0:
            chunks[0] = batch[0][0].as_py()
        
        # Read remaining chunks (1 through N-1)
        for i in range(1, total_chunks):
            try:
                batch, chunk_meta = reader.read_chunk()
            except StopIteration:
                raise ValueError(f"Stream ended prematurely at chunk {i}/{total_chunks}")
                
            if chunk_meta is None:
                raise ValueError(f"Missing chunk metadata for chunk {i}")
            chunk_dict = json.loads(chunk_meta.to_pybytes())
            chunk_idx = chunk_dict["chunk_index"]
            chunks[chunk_idx] = batch[0][0].as_py()

        # Reassemble in order
        full_bytes = b''.join(chunks[i] for i in range(total_chunks))
        
        # Verify checksum if provided
        if "md5" in meta_dict and meta_dict["md5"]:
            received_md5 = hashlib.md5(full_bytes).hexdigest()
            expected_md5 = meta_dict["md5"]
            if received_md5 != expected_md5:
                raise ValueError(f"Response checksum mismatch: expected={expected_md5}, received={received_md5}")
        
        # Check if size matches
        expected_size = np.prod(shape) * np.dtype(np_dtype).itemsize
        actual_size = len(full_bytes)
        
        if expected_size != actual_size:
            logger.error(f"Size mismatch! Expected {expected_size} bytes for shape {shape} with dtype {np_dtype}, "
                        f"but received {actual_size} bytes")
            # Try to infer what went wrong
            if actual_size == expected_size * 2:
                logger.error("Received 2x expected bytes - possible dtype mismatch (sent float32, expected float16?)")
            elif actual_size == expected_size // 2:
                logger.error("Received 0.5x expected bytes - possible dtype mismatch (sent float16, expected float32?)")
        
        np_array = np.frombuffer(full_bytes, dtype=np_dtype).reshape(shape)
        arrow_tensor = pa.Tensor.from_numpy(np_array)
        # Pass original dtype to reconstruct bfloat16 correctly
        result = arrow_to_mlx(arrow_tensor, dtype_str)
        return result

    except ValueError:
        # Re-raise ValueError as-is (these are our custom error messages)
        raise
    except Exception as e:
        logger.error(f"Error converting response to MLX array: {e}", exc_info=True)
        raise ValueError(f"Failed to process server response: {e}") from e


def send_and_receive_tensor(client, tensor, session_id=None, max_retries=3, backoff=1):
    for attempt in range(max_retries):
        try:
            writer, reader = send_tensor(client, tensor, session_id)
            result = response_to_mlx_array(reader)
            
            # Send ACK to server to prevent premature stream closure
            try:
                ack_schema = pa.schema([pa.field("ack", pa.binary())])
                ack_batch = pa.RecordBatch.from_arrays([pa.array([b"ACK"])], schema=ack_schema)
                writer.write_metadata(b"ACK")
                logger.debug(f"[{session_id or 'default'}] Sent ACK to server")
            except Exception as ack_error:
                logger.debug(f"[{session_id or 'default'}] Could not send ACK: {ack_error}")
            
            return result

        except flight.FlightInternalError as e:
            error_str = str(e)
            if "Checksum mismatch" in error_str:
                if attempt < max_retries - 1:
                    logger.warning(f"Checksum mismatch, retrying ({attempt+1}/{max_retries})")
                    time.sleep(backoff)
                    continue
                else:
                    raise
            else:
                raise
        except ValueError as e:
            error_str = str(e)
            # Retry on premature stream end errors
            if "stream ended prematurely" in error_str.lower() or "no data received" in error_str.lower():
                if attempt < max_retries - 1:
                    logger.warning(f"Stream ended prematurely, retrying ({attempt+1}/{max_retries}): {e}")
                    time.sleep(backoff * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise
            else:
                raise
        except Exception as e:
            raise
    raise ValueError("Max retries exceeded")


def tensor_to_bytes(tensor):
    """Convert an MLX tensor to bytes."""
    if tensor is None:
        raise ValueError("Cannot convert None to bytes")
    # Ensure tensor is evaluated before converting to bytes
    mx.eval(tensor)
    if tensor.dtype == mx.bfloat16:
        tensor = tensor.view(mx.uint16)

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


def create_generate_step_with_flight(flight_clients: List[flight.FlightClient]):
    def generate_step(
        prompt: mx.array,
        model: nn.Module,
        temp: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:

        for client in flight_clients:
            client.do_action(flight.Action("ResetCache", json.dumps({"session_id": ""}).encode()))

        def sample(logits: mx.array) -> Tuple[mx.array, float]:
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

        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        def _step(y):
            nonlocal repetition_context
            if y.ndim == 0:
                y = y.reshape(1, 1)
            elif y.ndim == 1:
                y = y.reshape(1, -1)

            output = model(y, cache=cache) 

            for client in flight_clients:
                output = send_and_receive_tensor(client, output)

            logits = output[:, -1, :]

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
        mx.eval(y)
        while True:
            next_y, next_logprobs = _step(y)
            mx.eval(next_y)
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs

    return generate_step


class PipelineModel(nn.Module):
    """
    Wrapper that makes a distributed Flight pipeline look like a local model.
    """
    def __init__(self, flight_clients: List[flight.FlightClient], session_id: str):
        super().__init__()
        self.flight_clients = flight_clients
        self.session_id = session_id
        self.layers = []

    def make_cache(self):
        return []
        
    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        if inputs.dtype != mx.int32:
            inputs = inputs.astype(mx.int32)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        step = 2048
        if inputs.dtype == mx.int32 and inputs.shape[1] > step:
            processed = None
            while inputs.shape[1] > 0:
                chunk_size = min(step, inputs.shape[1])
                tensor = inputs[:, :chunk_size]
                for i, client in enumerate(self.flight_clients):
                    tensor = send_and_receive_tensor(client, tensor, self.session_id)
                processed = tensor
                inputs = inputs[:, chunk_size:]
                mx.clear_cache()
            return processed
        else:
            tensor = inputs
            for i, client in enumerate(self.flight_clients):
                tensor = send_and_receive_tensor(client, tensor, self.session_id)
        
        return tensor


def create_coordinator_generate_step(flight_clients: List[flight.FlightClient], tokenizer):
    from mlx_lm.generate import stream_generate
    
    def generate_step(
        prompt: mx.array,
        temp: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: int = 256,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        
        logger.info("=" * 80)
        logger.info("🚀 DISTRIBUTED GENERATION START")
        logger.info("=" * 80)
        logger.info(f"Prompt shape: {prompt.shape}")
        logger.info(f"Prompt tokens: {prompt.tolist() if prompt.size < 100 else f'{prompt.tolist()[:20]}...'}")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"Temperature: {temp}")
        logger.info(f"Top-p: {top_p}")
        logger.info(f"Top-k: {top_k}")
        logger.info(f"Repetition penalty: {repetition_penalty}")
        
        if hasattr(tokenizer, 'eos_token_ids'):
            logger.info(f"Tokenizer EOS token IDs: {tokenizer.eos_token_ids}")
            for eos_id in tokenizer.eos_token_ids:
                try:
                    decoded = tokenizer.decode([eos_id])
                    logger.info(f"  EOS token {eos_id} decodes to: {repr(decoded)}")
                except:
                    pass
        logger.info("=" * 80)
        
        session_id = str(uuid.uuid4())
        logger.info(f"Session ID: {session_id}")
        
        for client in flight_clients:
            client.do_action(flight.Action("ResetCache", json.dumps({"session_id": session_id}).encode()))
        
        pipeline_model = PipelineModel(flight_clients, session_id)
        
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
        
        if temp == 0:
            sampler = lambda x: mx.argmax(x, axis=-1)
        else:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
        
        token_count = 0
        
        for response in stream_generate(
            model=pipeline_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            token_count += 1
            
            if token_count <= 10 or token_count % 50 == 0:
                try:
                    decoded = tokenizer.decode([response.token])
                    logger.info(f"Token {token_count}: {response.token} (decoded: {repr(decoded)})")
                except:
                    logger.info(f"Token {token_count}: {response.token}")
            
            if hasattr(tokenizer, 'eos_token_ids') and response.token in tokenizer.eos_token_ids:
                logger.info(f"🛑 EOS token detected at position {token_count}: {response.token}")
                logger.info("   This will cause stream_generate to stop")

            yield response.token, response.logprobs

        logger.info("=" * 80)
        logger.info(f"🏁 DISTRIBUTED GENERATION END - Generated {token_count} tokens")
        logger.info("=" * 80)

    return generate_step


def mlx_to_arrow(tensor: mx.array) -> Tuple[pa.Tensor, str]:
    """
    Convert MLX array to PyArrow Tensor via NumPy interop.
    Preserves dtype and shape for bit-exact serialization.
    
    Returns:
        Tuple of (arrow_tensor, original_dtype_str) where original_dtype_str
        is needed to reconstruct bfloat16 (which is stored as uint16)
    """
    original_dtype = str(tensor.dtype)
    
    # Handle bfloat16 by viewing as uint16 (preserves exact bits, no precision loss)
    # NumPy/Arrow don't support bfloat16, but we can store the raw 16-bit values
    if tensor.dtype == mx.bfloat16:
        tensor = tensor.view(mx.uint16)
    
    np_array = np.array(tensor, copy=False)  # Zero-copy view when possible
    return pa.Tensor.from_numpy(np_array), original_dtype


def arrow_to_mlx(arrow_tensor: pa.Tensor, original_dtype: str = None) -> mx.array:
    """
    Convert PyArrow Tensor back to MLX array.
    Preserves original dtype through NumPy.
    
    Args:
        arrow_tensor: PyArrow tensor to convert
        original_dtype: Original MLX dtype string (e.g., "mlx.core.bfloat16")
                       Required for bfloat16 reconstruction
    """
    np_array = arrow_tensor.to_numpy()
    mlx_array = mx.array(np_array)
    
    # If original dtype was bfloat16, view the uint16 data back as bfloat16
    if original_dtype == "mlx.core.bfloat16":
        mlx_array = mlx_array.view(mx.bfloat16)

    return mlx_array

