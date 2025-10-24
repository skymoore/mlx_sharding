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
    
    def _serialize_cache(self, cache: Tuple[mx.array, mx.array]) -> bytes:
        """
        Serialize KV cache tuple to bytes using msgpack.
        
        Args:
            cache: Tuple of (keys, values) as mx.array
            
        Returns:
            Serialized bytes
        """
        keys, values = cache
        data = {
            "keys": {
                "data": keys.tobytes(),
                "shape": list(keys.shape),
                "dtype": str(keys.dtype)
            },
            "values": {
                "data": values.tobytes(),
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
            Tuple of (keys, values) as mx.array
        """
        unpacked = msgpack.unpackb(data, raw=False)
        
        keys_data = unpacked["keys"]
        keys = mx.frombuffer(
            keys_data["data"],
            dtype=getattr(mx, keys_data["dtype"].replace("mlx.core.", ""))
        ).reshape(keys_data["shape"])
        
        values_data = unpacked["values"]
        values = mx.frombuffer(
            values_data["data"],
            dtype=getattr(mx, values_data["dtype"].replace("mlx.core.", ""))
        ).reshape(values_data["shape"])
        
        return (keys, values)
    
    def get_cache(
        self,
        session_id: str,
        num_layers: int
    ) -> Optional[list]:
        """
        Read cache for all layers from Redis.
        
        Args:
            session_id: Unique session identifier
            num_layers: Number of layers to read cache for
            
        Returns:
            List of cache tuples (one per layer), or None if not found
        """
        try:
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
            
            # Deserialize cache entries
            cache = []
            for layer_idx, data in enumerate(results):
                if data is not None:
                    cache.append(self._deserialize_cache(data))
                else:
                    # Missing layer cache - this shouldn't happen in normal operation
                    logger.warning(f"Missing cache for session {session_id} layer {layer_idx}")
                    cache.append(None)
            
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
            if self.enable_pipelining:
                # Use pipeline for batch write
                pipe = self.client.pipeline()
                for layer_idx, layer_cache in enumerate(cache):
                    if layer_cache is not None:
                        key = self._make_key(session_id, layer_idx)
                        data = self._serialize_cache(layer_cache)
                        pipe.setex(key, ttl, data)
                pipe.execute()
            else:
                # Sequential writes
                for layer_idx, layer_cache in enumerate(cache):
                    if layer_cache is not None:
                        key = self._make_key(session_id, layer_idx)
                        data = self._serialize_cache(layer_cache)
                        self.client.setex(key, ttl, data)
            
            logger.debug(f"Stored cache for session {session_id} ({len(cache)} layers, TTL={ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write cache to Redis: {e}")
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
