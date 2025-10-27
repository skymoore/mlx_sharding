import pyarrow as pa
from pyarrow import flight
import hashlib
import json
import logging
import mlx.core as mx
from mlx.utils import tree_reduce
from shard.server.utils import load_model, mlx_to_arrow, arrow_to_mlx
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = None
CACHES = {}  # session_id -> list[KVCache]
CACHE_REQUEST_COUNTS = {}  # session_id -> int (track requests per session)

# Chunk size for large tensors
CHUNK_SIZE_BYTES = 2 * 1024 * 1024  # 2MB chunks


class MLXFlightAuth(flight.ServerAuthHandler):
    def is_valid(self, token):
        return b""  # No authentication for simplicity


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


class MLXFlightServer(flight.FlightServerBase):
    def __init__(self, location, *args, **kwargs):
        super().__init__(location, auth_handler=MLXFlightAuth(), *args, **kwargs)
        self.caches = CACHES
        self.cache_request_counts = CACHE_REQUEST_COUNTS
        self.model = MODEL

    def do_action(self, context, action):
        try:
            if action.type == "ResetCache":
                body = json.loads(action.body.to_pybytes())
                session_id = body.get("session_id")
                reset_cache(session_id)
                return [flight.Result(b"Cache reset successfully")]
            raise flight.FlightUnavailableError(f"Unknown action: {action.type}")
        except Exception as e:
            logger.error(f"Error in do_action: {e}", exc_info=True)
            raise flight.FlightInternalError(str(e))

    def do_exchange(self, context, descriptor, reader, writer):
        session_id = "unknown"
        resp_schema = pa.schema([pa.field("chunk", pa.binary())])
        writer_begun = False
        
        try:
            command = descriptor.command.decode()
            if command != "SendTensor":
                raise flight.FlightInvalidArgument("Unknown command")

            logger.debug("Starting do_exchange - reading metadata")
            # Read metadata from first chunk
            batch, meta = reader.read_chunk()
            if meta is None:
                raise flight.FlightInvalidArgument("No metadata provided")

            meta_dict = json.loads(meta.to_pybytes())
            session_id = meta_dict.get("session_id", "default")
            total_chunks = meta_dict["total_chunks"]
            shape = tuple(meta_dict["shape"])
            dtype_str = meta_dict["dtype"]
            
            logger.debug(f"[{session_id}] Received metadata: shape={shape}, dtype={dtype_str}, chunks={total_chunks}")

            # Map to NumPy dtype
            dtype_map = {
                "mlx.core.float32": np.float32,
                "mlx.core.int32": np.int32,
                "mlx.core.int64": np.int64,
                "mlx.core.float16": np.float16,
                "mlx.core.bfloat16": np.uint16,
            }
            np_dtype = dtype_map.get(dtype_str, np.float32)

            # Collect chunks
            chunks = {}
            if batch and batch.num_rows > 0:
                chunks[0] = batch[0][0].as_py()  # First chunk data if present

            logger.debug(f"[{session_id}] Reading {total_chunks} chunks")
            for i in range(1, total_chunks):
                batch, chunk_meta = reader.read_chunk()
                if chunk_meta is None:
                    raise flight.FlightInvalidArgument("Missing chunk metadata")
                chunk_dict = json.loads(chunk_meta.to_pybytes())
                chunk_idx = chunk_dict["chunk_index"]
                chunks[chunk_idx] = batch[0][0].as_py()

            # Reassemble bytes
            full_bytes = b"".join(chunks.get(i, b"") for i in range(total_chunks))
            received_md5 = hashlib.md5(full_bytes).hexdigest()
            expected_md5 = meta_dict["md5"]
            
            logger.debug(f"[{session_id}] Received tensor: shape={shape}, dtype={dtype_str}, "
                        f"size={len(full_bytes)} bytes, chunks={total_chunks}, "
                        f"expected_checksum={expected_md5}, received_checksum={received_md5}")
            
            if received_md5 != expected_md5:
                logger.error(f"[{session_id}] Checksum mismatch: "
                           f"expected={expected_md5}, received={received_md5}")
                raise flight.FlightInternalError("Checksum mismatch")
            if len(full_bytes) == 0:
                raise flight.FlightInvalidArgument("No data received")

            # Convert to NumPy and reshape
            logger.debug(f"[{session_id}] Converting to NumPy array")
            np_array = np.frombuffer(full_bytes, dtype=np_dtype)
            np_array = np_array.reshape(shape)

            # Convert to MLX
            logger.debug(f"[{session_id}] Converting to MLX tensor")
            arrow_tensor = pa.Tensor.from_numpy(np_array)
            tensor = arrow_to_mlx(arrow_tensor)

            # Process tensor
            if self.model is None:
                raise flight.FlightInternalError("Model not loaded")

            if session_id not in self.caches:
                logger.info(f"[{session_id}] Creating new cache for session")
                reset_cache(session_id)
            cache_to_use = self.caches[session_id]

            logger.info(f"[{session_id}] Processing tensor through model - shape={tensor.shape}, dtype={tensor.dtype}")
            try:
                processed_tensor = self.model(tensor, cache=cache_to_use)
                logger.info(f"[{session_id}] Model processing complete - output shape={processed_tensor.shape}")
            except Exception as model_error:
                logger.error(f"[{session_id}] Model processing failed: {model_error}", exc_info=True)
                raise

            # Update request count and clear cache if needed
            if session_id not in self.cache_request_counts:
                self.cache_request_counts[session_id] = 0
            self.cache_request_counts[session_id] += 1
            if self.cache_request_counts[session_id] % 256 == 0:
                mx.clear_cache()
                logger.debug(
                    f"Cleared cache after {self.cache_request_counts[session_id]} requests for session {session_id}"
                )

            # Prepare response: chunk if large
            logger.info(f"[{session_id}] Preparing response - tensor shape={processed_tensor.shape}, dtype={processed_tensor.dtype}")
            arrow_processed = mlx_to_arrow(processed_tensor)
            np_processed = arrow_processed.to_numpy()
            total_size = np_processed.nbytes
            item_size = np_processed.itemsize
            chunk_items = CHUNK_SIZE_BYTES // item_size
            total_items = np_processed.size
            total_chunks_resp = (total_items + chunk_items - 1) // chunk_items
            
            logger.info(f"[{session_id}] Response tensor: shape={np_processed.shape}, dtype={np_processed.dtype}, "
                       f"nbytes={total_size}, itemsize={item_size}, total_chunks={total_chunks_resp}")
            
            # Begin writer now that we're ready to send response
            logger.info(f"[{session_id}] Beginning writer with schema")
            try:
                writer.begin(resp_schema)
                writer_begun = True
                logger.info(f"[{session_id}] Writer begun successfully")
            except Exception as begin_error:
                logger.error(f"[{session_id}] Failed to begin writer: {begin_error}", exc_info=True)
                raise

            # Prepare chunks
            flat_np = np_processed.flatten()
            
            # Send metadata with chunk 0 data
            resp_meta = {
                "success": True,
                "message": "Tensor processed successfully",
                "shape": list(processed_tensor.shape),
                "dtype": str(processed_tensor.dtype),
                "total_chunks": total_chunks_resp,
                "md5": hashlib.md5(flat_np.tobytes()).hexdigest(),
            }
            
            logger.info(f"[{session_id}] Sending metadata and chunk 0 (total_chunks={total_chunks_resp})")
            # First chunk (chunk 0) sent with metadata
            if total_chunks_resp > 0:
                start = 0
                end = min(chunk_items, total_items)
                chunk_np = flat_np[start:end]
                chunk_bytes = chunk_np.tobytes()
                chunk_array = pa.array([chunk_bytes])
                meta_batch = pa.RecordBatch.from_arrays([chunk_array], schema=resp_schema)
                try:
                    writer.write_with_metadata(meta_batch, json.dumps(resp_meta).encode())
                    logger.info(f"[{session_id}] Successfully wrote metadata and chunk 0")
                except Exception as write_error:
                    logger.error(f"[{session_id}] Failed to write metadata and chunk 0: {write_error}", exc_info=True)
                    raise
            else:
                # Edge case: empty tensor
                logger.warning(f"[{session_id}] Empty tensor response (total_chunks=0)")
                meta_batch = pa.RecordBatch.from_arrays([pa.array([b""], type=pa.binary())], schema=resp_schema)
                writer.write_with_metadata(meta_batch, json.dumps(resp_meta).encode())

            # Send remaining chunks (1 through N-1)
            logger.debug(f"[{session_id}] Sending remaining {total_chunks_resp - 1} chunks")
            for i in range(1, total_chunks_resp):
                start = i * chunk_items
                end = min(start + chunk_items, total_items)
                chunk_np = flat_np[start:end]
                chunk_bytes = chunk_np.tobytes()
                chunk_array = pa.array([chunk_bytes])
                batch = pa.RecordBatch.from_arrays([chunk_array], schema=resp_schema)
                writer.write_with_metadata(
                    batch, json.dumps({"chunk_index": i}).encode()
                )

            logger.info(f"[{session_id}] All response data written")
            
            # CRITICAL FIX for race condition:
            # In bidirectional streaming, if the server returns immediately after writing,
            # the stream can close before the client has time to read the data, causing
            # StopIteration on the client side.
            #
            # Solution: Wait for the client to send an ACK (empty metadata message) to
            # signal it has received all the data. This ensures the stream stays open
            # until the client is done reading.
            try:
                logger.debug(f"[{session_id}] Waiting for client ACK...")
                ack_batch, ack_meta = reader.read_chunk()
                logger.info(f"[{session_id}] Received client ACK, closing stream")
            except StopIteration:
                # Client closed their end, which is fine
                logger.debug(f"[{session_id}] Client closed stream (no ACK needed)")
            except Exception as ack_error:
                # Don't fail if ACK fails - just log it
                logger.warning(f"[{session_id}] Error waiting for ACK: {ack_error}")
            
            logger.info(f"[{session_id}] do_exchange completing. Stream will close on return.")

        except Exception as e:
            logger.error(f"[{session_id}] Error in do_exchange: {e}", exc_info=True)
            # Always try to send error response through stream
            try:
                # If writer was already begun, we can't call begin() again
                # But we can still try to write an error batch
                if not writer_begun:
                    logger.debug(f"[{session_id}] Beginning writer for error response")
                    writer.begin(resp_schema)
                
                error_meta = {
                    "success": False,
                    "message": str(e),
                    "shape": [0],
                    "dtype": "mlx.core.float32",
                    "total_chunks": 1,
                    "md5": "",
                }
                error_batch = pa.RecordBatch.from_arrays([pa.array([b""])], schema=resp_schema)
                writer.write_with_metadata(error_batch, json.dumps(error_meta).encode())
                logger.info(f"[{session_id}] Error response sent, do_exchange will complete and close stream")
                # Stream closes automatically when do_exchange returns
            except Exception as write_error:
                logger.error(f"[{session_id}] Failed to send error response: {write_error}", exc_info=True)
                # If we can't send error through stream, raise it to propagate to client
                raise flight.FlightInternalError(str(e)) from write_error


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

    # Set wired limit for optimal Metal memory management
    if mx.metal.is_available():
        try:
            model_bytes = tree_reduce(
                lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
                MODEL,
                0,
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

    # Start Flight server
    location = f"grpc://0.0.0.0:{port}"
    server = MLXFlightServer(location)
    logger.info(f"Flight server started, listening on {location}")
    if start_layer is not None or end_layer is not None:
        actual_start = start_layer or 0
        actual_end = (end_layer - 1) if end_layer else "end"
        logger.info(
            f"Model loaded with layers {actual_start} to {actual_end} (inclusive)"
        )
    server.serve()  # Blocks until shutdown

