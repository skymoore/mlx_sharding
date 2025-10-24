
import grpc
from concurrent import futures
from ..grpc import mlx_tensor_pb2, mlx_tensor_pb2_grpc
from ..utils import bytes_to_tensor, load_model, tensor_to_bytes
import mlx.core as mx
from mlx_lm.models.cache import KVCache
import threading
import time
from typing import Optional
from ..redis_cache import RedisKVCache

MODEL = None
CACHE = None
REDIS_CACHE: Optional[RedisKVCache] = None

# Global chunk buffer
CHUNK_BUFFERS = {}  # Key: tensor_id, Value: {'chunks': {}, 'timestamp': float, 'total': int, 'shape': tuple, 'dtype': str}
CHUNK_BUFFER_LOCK = threading.Lock()
CHUNK_TIMEOUT_SECONDS = 60

def cleanup_stale_chunks():
    """Background task to clean up incomplete chunk transfers."""
    while True:
        time.sleep(30)  # Run every 30 seconds
        current_time = time.time()
        
        with CHUNK_BUFFER_LOCK:
            stale_ids = [
                tid for tid, data in CHUNK_BUFFERS.items()
                if current_time - data['timestamp'] > CHUNK_TIMEOUT_SECONDS
            ]
            
            for tid in stale_ids:
                print(f"🧹 Cleaning up stale chunks for tensor_id: {tid}")
                del CHUNK_BUFFERS[tid]

def reset_cache():
    global CACHE
    if hasattr(MODEL, "make_cache"):
        CACHE = MODEL.make_cache()
    else:
        raise ValueError("Model does not have make_cache() method. Please use a compatible model.")
    print("Cache has been reset")


class MLXTensorServicer(mlx_tensor_pb2_grpc.MLXTensorServiceServicer):
    def SendTensor(self, request, context):
        try:
            import time
            start_time = time.time()
            
            # Extract session_id from request
            session_id = request.session_id if request.session_id else None
            
            # Check which type of tensor we received
            if request.HasField('full_tensor'):
                # Original non-chunked path (backward compatible)
                tensor_msg = request.full_tensor
                tensor = bytes_to_tensor(tensor_msg.tensor_data, tensor_msg.dtype)
                tensor = mx.reshape(tensor, tensor_msg.shape)
                recv_time = time.time() - start_time
                tensor_size_mb = len(tensor_msg.tensor_data) / (1024 * 1024)
                print(f"📥 Received full tensor: shape={tensor.shape}, size={tensor_size_mb:.2f}MB, time={recv_time:.2f}s, session={session_id}")
                
            elif request.HasField('chunked_tensor'):
                # NEW: Chunked tensor path
                chunk = request.chunked_tensor
                tensor_id = chunk.tensor_id
                chunk_idx = chunk.chunk_index
                total_chunks = chunk.total_chunks
                chunk_size_mb = len(chunk.chunk_data) / (1024 * 1024)
                
                print(f"📥 Received chunk {chunk_idx+1}/{total_chunks} for {tensor_id[:8]}... ({chunk_size_mb:.2f}MB), session={session_id}")
                
                with CHUNK_BUFFER_LOCK:
                    # Initialize buffer for this tensor if first chunk
                    if tensor_id not in CHUNK_BUFFERS:
                        CHUNK_BUFFERS[tensor_id] = {
                            'chunks': {},
                            'timestamp': time.time(),
                            'total': total_chunks,
                            'shape': tuple(chunk.shape),
                            'dtype': chunk.dtype
                        }
                    
                    # Store this chunk
                    CHUNK_BUFFERS[tensor_id]['chunks'][chunk_idx] = chunk.chunk_data
                    CHUNK_BUFFERS[tensor_id]['timestamp'] = time.time()  # Update timestamp
                    
                    # Check if we have all chunks
                    if len(CHUNK_BUFFERS[tensor_id]['chunks']) == total_chunks:
                        print(f"✅ All {total_chunks} chunks received, reassembling...")
                        
                        # Reassemble in order
                        full_data = b''.join([
                            CHUNK_BUFFERS[tensor_id]['chunks'][i]
                            for i in range(total_chunks)
                        ])
                        
                        # Clean up buffer
                        shape = CHUNK_BUFFERS[tensor_id]['shape']
                        dtype = CHUNK_BUFFERS[tensor_id]['dtype']
                        del CHUNK_BUFFERS[tensor_id]
                        
                        # Convert to tensor
                        tensor = bytes_to_tensor(full_data, dtype)
                        tensor = mx.reshape(tensor, shape)
                        total_size_mb = len(full_data) / (1024 * 1024)
                        reassemble_time = time.time() - start_time
                        print(f"🔧 Reassembled tensor: shape={tensor.shape}, size={total_size_mb:.2f}MB, time={reassemble_time:.2f}s")
                    else:
                        # Not all chunks received yet, return success but no tensor
                        return mlx_tensor_pb2.TensorResponse(
                            success=True,
                            message=f"Chunk {chunk_idx+1}/{total_chunks} received",
                            tensor=None
                        )
            else:
                return mlx_tensor_pb2.TensorResponse(
                    success=False,
                    message="Invalid request: no tensor payload",
                    tensor=None
                )
            
            # Process the tensor (same for both paths)
            if MODEL is not None:
                # Load cache from Redis if session_id provided and Redis is available
                cache_to_use = CACHE
                if session_id and REDIS_CACHE is not None:
                    try:
                        # Get this peer's global layer range
                        start_layer = MODEL.start_layer
                        end_layer = MODEL.end_layer + 1  # Exclusive
                        
                        redis_cache = REDIS_CACHE.get_cache(session_id, start_layer, end_layer)
                        if redis_cache is not None:
                            # Validate cache size matches expected layer count
                            expected_layers = end_layer - start_layer
                            if len(redis_cache) != expected_layers:
                                print(f"⚠️  Cache size mismatch: got {len(redis_cache)}, expected {expected_layers}")
                            else:
                                cache_to_use = redis_cache
                                print(f"📦 Loaded cache from Redis (layers {start_layer}-{end_layer-1})")
                        else:
                            print(f"📦 No cache in Redis (layers {start_layer}-{end_layer-1}), using local cache")
                    except Exception as e:
                        print(f"⚠️  Failed to load cache from Redis: {e}, using local cache")
                        import traceback
                        traceback.print_exc()
                
                process_start = time.time()
                processed_tensor = MODEL(tensor, cache=cache_to_use)
                process_time = time.time() - process_start
                print(f"⚙️  Processed: shape={processed_tensor.shape}, time={process_time:.2f}s")
                
                # Save cache to Redis if session_id provided and Redis is available
                if session_id and REDIS_CACHE is not None and cache_to_use is not None:
                    try:
                        # Get this peer's global layer range
                        start_layer = MODEL.start_layer
                        end_layer = MODEL.end_layer + 1  # Exclusive
                        expected_layers = end_layer - start_layer
                        
                        # Validate cache size before saving
                        if len(cache_to_use) != expected_layers:
                            print(f"⚠️  Cache size mismatch before save: got {len(cache_to_use)}, expected {expected_layers}")
                        
                        REDIS_CACHE.set_cache(session_id, cache_to_use, start_layer)
                        print(f"💾 Saved cache to Redis (layers {start_layer}-{end_layer-1})")
                    except Exception as e:
                        print(f"⚠️  Failed to save cache to Redis: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Only reduce to last token if this is the last peer (has lm_head)
                # Intermediate peers need to pass full sequence for KV cache building
                is_last_peer = hasattr(MODEL, 'lm_head')
                print(f"🔍 Debug: is_last_peer={is_last_peer}, has_lm_head={hasattr(MODEL, 'lm_head')}, model_type={type(MODEL).__name__}")
                if hasattr(MODEL, 'start_layer') and hasattr(MODEL, 'end_layer'):
                    print(f"🔍 Debug: start_layer={MODEL.start_layer}, end_layer={MODEL.end_layer}")
                if hasattr(MODEL, 'args'):
                    print(f"🔍 Debug: num_hidden_layers={MODEL.args.num_hidden_layers}")
                
                if is_last_peer and len(processed_tensor.shape) == 3 and processed_tensor.shape[1] > 1:
                    processed_tensor = processed_tensor[:, -1:, :]
                    print(f"✂️  Reduced to last token: {processed_tensor.shape}")
                elif len(processed_tensor.shape) == 3 and processed_tensor.shape[1] > 1:
                    print(f"⏩ Passing full sequence (intermediate peer): {processed_tensor.shape}")
                
                serialize_start = time.time()
                processed_bytes = tensor_to_bytes(processed_tensor)
                serialize_time = time.time() - serialize_start
                response_size_mb = len(processed_bytes) / (1024 * 1024)
                print(f"📤 Sending response: size={response_size_mb:.2f}MB, time={serialize_time:.2f}s")
                
                response_tensor = mlx_tensor_pb2.Tensor(
                    tensor_data=processed_bytes,
                    shape=list(processed_tensor.shape),
                    dtype=str(processed_tensor.dtype)
                )
                return mlx_tensor_pb2.TensorResponse(
                    success=True,
                    message="Tensor processed successfully",
                    tensor=response_tensor
                )
            else:
                return mlx_tensor_pb2.TensorResponse(
                    success=False,
                    message="Model not loaded",
                    tensor=None
                )
                
        except Exception as e:
            print(f"❌ Error processing tensor: {e}")
            import traceback
            traceback.print_exc()
            return mlx_tensor_pb2.TensorResponse(success=False, message=str(e), tensor=None)

    def ResetCache(self, request, context):
        try:
            reset_cache()
            return mlx_tensor_pb2.ResetCacheResponse(
                success=True,
                message="Cache reset successfully"
            )
        except Exception as e:
            print(f"Error resetting cache: {e}")
            return mlx_tensor_pb2.ResetCacheResponse(
                success=False,
                message=f"Error resetting cache: {str(e)}"
            )


def serve(model_path, start_layer=None, end_layer=None, port=50051, preloaded_model=None, redis_url=None):
    global MODEL, REDIS_CACHE
    
    # Use preloaded model if provided (V2 architecture), otherwise load it (V1 architecture)
    if preloaded_model is not None:
        MODEL = preloaded_model
    else:
        MODEL = load_model(model_path, start_layer=start_layer, end_layer=end_layer)
    
    # Debug: Print model structure
    print(f"🔍 Model loaded: type={type(MODEL).__name__}")
    print(f"🔍 Model attributes: {dir(MODEL)}")
    print(f"🔍 Has lm_head: {hasattr(MODEL, 'lm_head')}")
    if hasattr(MODEL, 'start_layer'):
        print(f"🔍 start_layer: {MODEL.start_layer}")
    if hasattr(MODEL, 'end_layer'):
        print(f"🔍 end_layer: {MODEL.end_layer}")
    if hasattr(MODEL, 'args'):
        print(f"🔍 num_hidden_layers: {MODEL.args.num_hidden_layers}")
    
    # Initialize Redis cache if URL provided
    if redis_url:
        try:
            print(f"🔌 Connecting to Redis at {redis_url}...")
            REDIS_CACHE = RedisKVCache(redis_url=redis_url)
            # Test the connection with a simple operation
            test_key = "mlx:test:connection"
            REDIS_CACHE.client.set(test_key, "test", ex=5)
            test_value = REDIS_CACHE.client.get(test_key)
            REDIS_CACHE.client.delete(test_key)
            print(f"✅ Redis cache initialized and tested: {redis_url}")
            print(f"✅ Redis connection verified (ping successful, read/write test passed)")
        except Exception as e:
            print(f"⚠️  Failed to initialize Redis cache: {e}")
            print(f"⚠️  Continuing without Redis cache")
            REDIS_CACHE = None
    else:
        print(f"ℹ️  No Redis URL provided, using local cache only")
        REDIS_CACHE = None
    
    reset_cache()
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_stale_chunks, daemon=True)
    cleanup_thread.start()
    print("🧹 Started chunk cleanup thread")
    
    server_options = [
        ('grpc.max_metadata_size', 64 * 1024 * 1024),  # 64MB metadata
        ('grpc.max_send_message_length', -1),  # Unlimited send
        ('grpc.max_receive_message_length', -1),  # Unlimited receive
        ('grpc.http2.max_frame_size', 16 * 1024 * 1024),  # 16MB frames
    ]
    server = grpc.server(futures.ThreadPoolExecutor(
        max_workers=10), options=server_options)
    mlx_tensor_pb2_grpc.add_MLXTensorServiceServicer_to_server(
        MLXTensorServicer(), server)

    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Server started, listening on 0.0.0.0:{port}")
    if start_layer is not None or end_layer is not None:
        # end_layer is exclusive, so actual last layer is end_layer-1
        actual_start = start_layer or 0
        actual_end = (end_layer - 1) if end_layer else 'end'
        print(f"Model loaded with layers {actual_start} to {actual_end} (inclusive)")
    server.wait_for_termination()

