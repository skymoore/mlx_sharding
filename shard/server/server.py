import grpc
from concurrent import futures
from shard.grpc import mlx_tensor_pb2, mlx_tensor_pb2_grpc
from shard.server.utils import bytes_to_tensor, load_model, tensor_to_bytes
import mlx.core as mx
import threading
import time
import logging
from mlx.utils import tree_reduce

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL = None
CACHES = {}  # session_id -> list[KVCache]
CACHE_REQUEST_COUNTS = {}  # session_id -> int (track requests per session)

# Prefill configuration for chunking large prompts
PREFILL_STEP_SIZE = 2048

# Global chunk buffer
CHUNK_BUFFERS = (
    {}
)  # Key: tensor_id, Value: {'chunks': {}, 'timestamp': float, 'total': int, 'shape': tuple, 'dtype': str}
CHUNK_BUFFER_LOCK = threading.Lock()
CHUNK_TIMEOUT_SECONDS = 60


def cleanup_stale_chunks():
    """Background task to clean up incomplete chunk transfers."""
    while True:
        time.sleep(30)  # Run every 30 seconds
        current_time = time.time()

        with CHUNK_BUFFER_LOCK:
            stale_ids = [
                tid
                for tid, data in CHUNK_BUFFERS.items()
                if current_time - data["timestamp"] > CHUNK_TIMEOUT_SECONDS
            ]

            for tid in stale_ids:
                logger.info(f"Cleaning up stale chunks for tensor_id: {tid}")
                del CHUNK_BUFFERS[tid]


def reset_cache(session_id):
    if not session_id:
        raise ValueError("No session_id provided")
    if hasattr(MODEL, "make_cache"):
        CACHES[session_id] = MODEL.make_cache()
        CACHE_REQUEST_COUNTS[session_id] = 0  # Reset request count
    else:
        raise ValueError(
            "Model does not have make_cache() method. Please use a compatible model."
        )
    logger.info(f"Cache reset for session {session_id}")


class MLXTensorServicer(mlx_tensor_pb2_grpc.MLXTensorServiceServicer):
    def SendTensor(self, request, context):
        try:
            import time

            start_time = time.time()

            # Extract session_id from request
            session_id = request.session_id if request.session_id else None

            # Check which type of tensor we received
            if request.HasField("full_tensor"):
                # Original non-chunked path (backward compatible)
                tensor_msg = request.full_tensor
                tensor = bytes_to_tensor(tensor_msg.tensor_data, tensor_msg.dtype)
                tensor = mx.reshape(tensor, tensor_msg.shape)

            elif request.HasField("chunked_tensor"):
                # NEW: Chunked tensor path
                chunk = request.chunked_tensor
                tensor_id = chunk.tensor_id
                chunk_idx = chunk.chunk_index
                total_chunks = chunk.total_chunks

                with CHUNK_BUFFER_LOCK:
                    # Initialize buffer for this tensor if first chunk
                    if tensor_id not in CHUNK_BUFFERS:
                        CHUNK_BUFFERS[tensor_id] = {
                            "chunks": {},
                            "timestamp": time.time(),
                            "total": total_chunks,
                            "shape": tuple(chunk.shape),
                            "dtype": chunk.dtype,
                        }

                    # Store this chunk
                    CHUNK_BUFFERS[tensor_id]["chunks"][chunk_idx] = chunk.chunk_data
                    CHUNK_BUFFERS[tensor_id][
                        "timestamp"
                    ] = time.time()  # Update timestamp

                    # Check if we have all chunks
                    if len(CHUNK_BUFFERS[tensor_id]["chunks"]) == total_chunks:
                        # Reassemble in order
                        full_data = b"".join(
                            [
                                CHUNK_BUFFERS[tensor_id]["chunks"][i]
                                for i in range(total_chunks)
                            ]
                        )

                        # Clean up buffer
                        shape = CHUNK_BUFFERS[tensor_id]["shape"]
                        dtype = CHUNK_BUFFERS[tensor_id]["dtype"]
                        del CHUNK_BUFFERS[tensor_id]

                        # Convert to tensor
                        tensor = bytes_to_tensor(full_data, dtype)
                        tensor = mx.reshape(tensor, shape)
                    else:
                        # Not all chunks received yet, return success but no tensor
                        return mlx_tensor_pb2.TensorResponse(
                            success=True,
                            message=f"Chunk {chunk_idx+1}/{total_chunks} received",
                            tensor=None,
                        )
            else:
                return mlx_tensor_pb2.TensorResponse(
                    success=False,
                    message="Invalid request: no tensor payload",
                    tensor=None,
                )

            # Process the tensor (same for both paths)
            if MODEL is not None:
                # Print progress indicator
                print(".", end="", flush=True)
                
                # Pipeline parallelism: Each peer maintains its OWN cache in memory
                # The cache persists across tokens within a generation
                # ResetCache RPC clears it between generations
                # NO cache synchronization needed - each peer's cache is independent
                if session_id is None:
                    session_id = "default"
                if session_id not in CACHES:
                    reset_cache(session_id)
                cache_to_use = CACHES[session_id]

                # 🔥 NEW: Prefill chunking for large inputs (like initial prompt)
                # This prevents OOM on long prompts by processing in chunks
                if tensor.shape[1] > PREFILL_STEP_SIZE:
                    logger.info(f"Chunking large input: {tensor.shape[1]} tokens")
                    
                    # Process all but the last chunk
                    while tensor.shape[1] > PREFILL_STEP_SIZE:
                        chunk = tensor[:, :PREFILL_STEP_SIZE]
                        MODEL(chunk, cache=cache_to_use)
                        mx.eval([c.state for c in cache_to_use])
                        tensor = tensor[:, PREFILL_STEP_SIZE:]
                        mx.clear_cache()
                        print(".", end="", flush=True)  # Progress for each chunk
                    
                    # Process remaining tokens (if any)
                    if tensor.shape[1] > 0:
                        processed_tensor = MODEL(tensor, cache=cache_to_use)
                    else:
                        # All tokens were processed in chunks, return dummy tensor
                        # This shouldn't happen but handle it gracefully
                        processed_tensor = mx.zeros((1, 1, MODEL.args.hidden_size))
                else:
                    # Normal single-token or small batch processing
                    processed_tensor = MODEL(tensor, cache=cache_to_use)

                # 🔥 NEW: Track request count for periodic cache clearing
                if session_id not in CACHE_REQUEST_COUNTS:
                    CACHE_REQUEST_COUNTS[session_id] = 0
                CACHE_REQUEST_COUNTS[session_id] += 1
                
                # Clear cache every 256 requests to prevent memory accumulation
                if CACHE_REQUEST_COUNTS[session_id] % 256 == 0:
                    mx.clear_cache()
                    logger.debug(f"Cleared cache after {CACHE_REQUEST_COUNTS[session_id]} requests for session {session_id}")

                # NEVER reduce to last token on the peer side
                # The coordinator will extract the last position for sampling
                # Peers must always return full sequence to build KV cache correctly

                processed_bytes = tensor_to_bytes(processed_tensor)

                response_tensor = mlx_tensor_pb2.Tensor(
                    tensor_data=processed_bytes,
                    shape=list(processed_tensor.shape),
                    dtype=str(processed_tensor.dtype),
                )
                return mlx_tensor_pb2.TensorResponse(
                    success=True,
                    message="Tensor processed successfully",
                    tensor=response_tensor,
                )
            else:
                return mlx_tensor_pb2.TensorResponse(
                    success=False, message="Model not loaded", tensor=None
                )

        except Exception as e:
            logger.error(f"Error processing tensor: {e}", exc_info=True)
            return mlx_tensor_pb2.TensorResponse(
                success=False, message=str(e), tensor=None
            )

    def ResetCache(self, request, context):
        try:
            reset_cache(request.session_id)
            return mlx_tensor_pb2.ResetCacheResponse(
                success=True, message="Cache reset successfully"
            )
        except Exception as e:
            logger.error(f"Error resetting cache: {e}", exc_info=True)
            return mlx_tensor_pb2.ResetCacheResponse(
                success=False, message=f"Error resetting cache: {str(e)}"
            )


def serve(
    model_path, start_layer=None, end_layer=None, port=50051, preloaded_model=None
):
    global MODEL

    # Use preloaded model if provided, otherwise load it
    if preloaded_model is not None:
        MODEL = preloaded_model
    else:
        MODEL = load_model(model_path, start_layer=start_layer, end_layer=end_layer)

    # Model loaded successfully
    logger.info(f"✓ Model loaded: {type(MODEL).__name__}")

    # 🔥 NEW: Set wired limit for optimal Metal memory management
    # This is critical for preventing memory pressure and performance issues
    if mx.metal.is_available():
        try:
            model_bytes = tree_reduce(
                lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
                MODEL,
                0
            )
            max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
            
            model_mb = model_bytes // (1024 * 1024)
            max_rec_mb = max_rec_size // (1024 * 1024)
            
            if model_bytes > 0.9 * max_rec_size:
                logger.warning(
                    f"Model requires {model_mb} MB which is close to the "
                    f"maximum recommended size of {max_rec_mb} MB. "
                    "This may impact performance."
                )
            
            mx.set_wired_limit(max_rec_size)
            logger.info(f"✓ Wired limit set to {max_rec_mb} MB for peer")
            logger.info(f"✓ Model size: {model_mb} MB")
        except Exception as e:
            logger.warning(f"Could not set wired limit: {e}")
    else:
        logger.info("Metal not available, skipping wired limit setup")

    # No initial reset needed - caches created per session

    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_stale_chunks, daemon=True)
    cleanup_thread.start()
    logger.info("Started chunk cleanup thread")

    server_options = [
        ("grpc.max_metadata_size", 64 * 1024 * 1024),  # 64MB metadata
        ("grpc.max_send_message_length", -1),  # Unlimited send
        ("grpc.max_receive_message_length", -1),  # Unlimited receive
        ("grpc.http2.max_frame_size", 4 * 1024 * 1024),  # 4MB frames (reduced from 16MB to avoid "Message too long" errors)
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10), options=server_options
    )
    mlx_tensor_pb2_grpc.add_MLXTensorServiceServicer_to_server(
        MLXTensorServicer(), server
    )

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Server started, listening on 0.0.0.0:{port}")
    if start_layer is not None or end_layer is not None:
        # end_layer is exclusive, so actual last layer is end_layer-1
        actual_start = start_layer or 0
        actual_end = (end_layer - 1) if end_layer else "end"
        logger.info(f"Model loaded with layers {actual_start} to {actual_end} (inclusive)")
    server.wait_for_termination()
