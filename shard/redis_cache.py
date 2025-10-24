"""
Redis-backed distributed KV cache for MLX model inference.

This module provides a Redis-backed cache that synchronizes KV cache state
across distributed peers during inference. Each peer reads the cache state
from Redis before processing and writes it back after processing.
"""

import logging
from typing import Optional, Tuple
import redis
import msgpack
import mlx.core as mx

logger = logging.getLogger(__name__)


class RedisKVCache:
    """
    Redis-backed KV cache for distributed inference.
    
    Cache keys follow the pattern:
        mlx:session:{session_id}:layer:{layer_idx}:cache
    
    Each cache entry contains a tuple of (keys, values) tensors serialized
    with msgpack.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,
        enable_pipelining: bool = True
    ):
        """
        Initialize Redis KV cache.
        
        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379")
            default_ttl: Default TTL for cache entries in seconds (default: 1 hour)
            enable_pipelining: Whether to use Redis pipelining for batch operations
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.enable_pipelining = enable_pipelining
        
        try:
            self.client = redis.from_url(redis_url, decode_responses=False)
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {redis_url}: {e}")
            raise
    
    def _make_key(self, session_id: str, layer_idx: int) -> str:
        """Generate Redis key for a specific layer's cache."""
        return f"mlx:session:{session_id}:layer:{layer_idx}:cache"
    
    def _serialize_cache(self, cache) -> bytes:
        """
        Serialize KV cache to bytes using msgpack.
        
        Args:
            cache: Either a tuple of (keys, values) or a KVCache object with keys/values attributes
            
        Returns:
            Serialized bytes
        """
        # Handle KVCache objects (from mlx_lm.models.cache)
        # Check if it has keys/values attributes (not None)
        if hasattr(cache, 'keys') and hasattr(cache, 'values'):
            keys = cache.keys
            values = cache.values
            
            # Skip empty caches (keys/values are None)
            if keys is None or values is None:
                logger.debug(f"Skipping empty KVCache (keys={keys}, values={values})")
                return msgpack.packb({"empty": True}, use_bin_type=True)
            
            logger.debug(f"Serializing KVCache object with keys/values")
        # Handle raw tuples
        elif isinstance(cache, tuple) and len(cache) == 2:
            logger.debug(f"Serializing raw tuple cache")
            keys, values = cache
        else:
            error_msg = f"Unsupported cache type: {type(cache)}, has_keys={hasattr(cache, 'keys')}, has_values={hasattr(cache, 'values')}, is_tuple={isinstance(cache, tuple)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        data = {
            "keys": {
                "data": bytes(memoryview(keys)),
                "shape": list(keys.shape),
                "dtype": str(keys.dtype)
            },
            "values": {
                "data": bytes(memoryview(values)),
                "shape": list(values.shape),
                "dtype": str(values.dtype)
            }
        }
        return msgpack.packb(data, use_bin_type=True)
    
    def _deserialize_cache(self, data: bytes) -> Tuple[mx.array, mx.array]:
        """
        Deserialize KV cache from bytes.
        
        Args:
            data: Serialized cache bytes
            
        Returns:
            Tuple of (keys, values) as mx.array, or None if cache was empty
        """
        import numpy as np
        unpacked = msgpack.unpackb(data, raw=False)
        
        # Handle empty cache marker
        if unpacked.get("empty"):
            logger.debug("Deserializing empty cache marker")
            return None
        
        keys_data = unpacked["keys"]
        # Convert bytes to numpy array first, then to MLX array
        dtype_str = keys_data["dtype"].replace("mlx.core.", "")
        mx_dtype = getattr(mx, dtype_str)
        np_array_keys = np.frombuffer(keys_data["data"], dtype=self._mx_to_np_dtype(dtype_str))
        keys = mx.array(np_array_keys, dtype=mx_dtype).reshape(keys_data["shape"])
        
        values_data = unpacked["values"]
        dtype_str = values_data["dtype"].replace("mlx.core.", "")
        mx_dtype = getattr(mx, dtype_str)
        np_array_values = np.frombuffer(values_data["data"], dtype=self._mx_to_np_dtype(dtype_str))
        values = mx.array(np_array_values, dtype=mx_dtype).reshape(values_data["shape"])
        
        return (keys, values)
    
    def _mx_to_np_dtype(self, mx_dtype_str: str):
        """Convert MLX dtype string to numpy dtype."""
        import numpy as np
        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": np.uint16,  # bfloat16 stored as uint16
            "int32": np.int32,
            "int64": np.int64,
        }
        return dtype_map.get(mx_dtype_str, np.float32)
    
    def get_cache(
        self,
        session_id: str,
        num_layers: int
    ) -> Optional[list]:
        """
        Read cache for all layers from Redis and reconstruct KVCache objects.
        
        Args:
            session_id: Unique session identifier
            num_layers: Number of layers to read cache for
            
        Returns:
            List of KVCache objects (one per layer), or None if not found
        """
        try:
            from mlx_lm.models.cache import KVCache
            
            if self.enable_pipelining:
                # Use pipeline for batch read
                pipe = self.client.pipeline()
                for layer_idx in range(num_layers):
                    key = self._make_key(session_id, layer_idx)
                    pipe.get(key)
                results = pipe.execute()
            else:
                # Sequential reads
                results = []
                for layer_idx in range(num_layers):
                    key = self._make_key(session_id, layer_idx)
                    results.append(self.client.get(key))
            
            # Check if any cache exists
            if all(r is None for r in results):
                logger.debug(f"No cache found for session {session_id}")
                return None
            
            # Deserialize cache entries and create KVCache objects
            cache = []
            for layer_idx, data in enumerate(results):
                if data is not None:
                    deserialized = self._deserialize_cache(data)
                    # Create KVCache object
                    kv_cache = KVCache()
                    
                    # Only set state if cache had data (not empty marker)
                    if deserialized is not None:
                        keys, values = deserialized
                        kv_cache.state = (keys, values)
                    # else: leave as empty KVCache
                    
                    cache.append(kv_cache)
                else:
                    # Missing layer cache - create empty KVCache
                    logger.warning(f"Missing cache for session {session_id} layer {layer_idx}")
                    cache.append(KVCache())
            
            logger.debug(f"Retrieved cache for session {session_id} ({num_layers} layers)")
            return cache
            
        except Exception as e:
            logger.error(f"Failed to read cache from Redis: {e}")
            return None
    
    def set_cache(
        self,
        session_id: str,
        cache: list,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Write cache for all layers to Redis.
        
        Args:
            session_id: Unique session identifier
            cache: List of cache tuples (one per layer)
            ttl: Time-to-live in seconds (uses default_ttl if None)
            
        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl
        
        try:
            logger.debug(f"set_cache called with cache type: {type(cache)}, length: {len(cache) if hasattr(cache, '__len__') else 'N/A'}")
            if self.enable_pipelining:
                # Use pipeline for batch write
                pipe = self.client.pipeline()
                for layer_idx, layer_cache in enumerate(cache):
                    if layer_cache is not None:
                        logger.debug(f"  Layer {layer_idx}: type={type(layer_cache)}, has_state={hasattr(layer_cache, 'state')}")
                        key = self._make_key(session_id, layer_idx)
                        data = self._serialize_cache(layer_cache)
                        pipe.setex(key, ttl, data)
                pipe.execute()
            else:
                # Sequential writes
                for layer_idx, layer_cache in enumerate(cache):
                    if layer_cache is not None:
                        logger.debug(f"  Layer {layer_idx}: type={type(layer_cache)}, has_state={hasattr(layer_cache, 'state')}")
                        key = self._make_key(session_id, layer_idx)
                        data = self._serialize_cache(layer_cache)
                        self.client.setex(key, ttl, data)
            
            logger.debug(f"Stored cache for session {session_id} ({len(cache)} layers, TTL={ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write cache to Redis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def delete_session(self, session_id: str, num_layers: int) -> bool:
        """
        Delete all cache entries for a session.
        
        Args:
            session_id: Unique session identifier
            num_layers: Number of layers to delete cache for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            keys = [self._make_key(session_id, i) for i in range(num_layers)]
            deleted = self.client.delete(*keys)
            logger.debug(f"Deleted {deleted} cache entries for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session cache: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """
        Cleanup expired cache entries (Redis handles this automatically via TTL).
        
        This method is a no-op since Redis automatically removes expired keys.
        Provided for API compatibility.
        
        Returns:
            0 (Redis handles cleanup automatically)
        """
        return 0
    
    def close(self):
        """Close Redis connection."""
        try:
            self.client.close()
            logger.info("Closed Redis connection")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
