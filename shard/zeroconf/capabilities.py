"""
System capability detection for MLX sharding.
Detects available RAM, CPU, and estimates model memory requirements.
"""

import psutil
import platform
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from mlx_lm.utils import hf_repo_to_path

logger = logging.getLogger(__name__)


class SystemCapabilities:
    """Detect and report system capabilities."""
    
    @staticmethod
    def get_capabilities(max_ram_gb: Optional[float] = None) -> Dict[str, Any]:
        """
        Get current system capabilities.
        
        Args:
            max_ram_gb: Optional maximum RAM limit in GB (caps available RAM)
        
        Returns:
            Dictionary with system information
        """
        mem = psutil.virtual_memory()
        
        ram_available_gb = round(mem.available / (1024**3), 2)
        
        # Apply max RAM limit if specified
        if max_ram_gb is not None and max_ram_gb > 0:
            ram_available_gb = min(ram_available_gb, max_ram_gb)
        
        capabilities = {
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_available_gb": ram_available_gb,
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "cpu_cores": psutil.cpu_count(logical=False) or 1,
            "cpu_threads": psutil.cpu_count(logical=True) or 1,
            "platform": platform.system().lower(),
            "architecture": platform.machine(),
            "mlx_available": True,  # Since we're using MLX
            "gpu_memory_gb": 0,  # MLX uses unified memory on Apple Silicon
        }
        
        logger.debug(f"System capabilities: {capabilities}")
        return capabilities
    
    @staticmethod
    def estimate_model_memory(model_path_or_repo: str, context_length: int = 8192) -> Dict[str, float]:
        """
        Estimate memory requirements for a model.
        
        Args:
            model_path_or_repo: Path to model or HuggingFace repo
            context_length: Target context length
            
        Returns:
            Dictionary with memory estimates in GB
        """
        try:
            # Get model path
            if Path(model_path_or_repo).exists():
                model_path = Path(model_path_or_repo)
            else:
                model_path = hf_repo_to_path(model_path_or_repo)
            
            # Load config
            config_path = model_path / "config.json"
            if not config_path.exists():
                logger.warning(f"Config not found at {config_path}, using defaults")
                return SystemCapabilities._default_memory_estimate(context_length)
            
            with open(config_path) as f:
                config = json.load(f)
            
            # Extract model parameters
            num_layers = config.get("num_hidden_layers", 32)
            hidden_size = config.get("hidden_size", 4096)
            num_attention_heads = config.get("num_attention_heads", 32)
            vocab_size = config.get("vocab_size", 32000)
            
            # Get actual model file size (most reliable method)
            total_file_size = 0
            for pattern in ["*.safetensors", "*.bin"]:
                for file_path in model_path.glob(pattern):
                    if file_path.is_file():
                        total_file_size += file_path.stat().st_size
            
            # Use actual file size as weights estimate (most accurate)
            weights_gb = total_file_size / (1024**3)
            
            # Check for quantization info (for logging purposes)
            quantization = config.get("quantization", {})
            quantization_config = config.get("quantization_config", {})
            
            bits = None
            if quantization and isinstance(quantization, dict):
                bits = quantization.get("bits")
            if bits is None and quantization_config and isinstance(quantization_config, dict):
                bits = quantization_config.get("bits")
            if bits is None:
                bits = 16  # Default assumption
            
            logger.info(f"Model file size: {weights_gb:.1f}GB ({bits}-bit quantization)")
            
            # KV cache estimation
            # KV cache per layer: 2 (K and V) * batch_size * num_heads * seq_len * head_dim
            head_dim = hidden_size // num_attention_heads
            batch_size = 1  # Assume batch size of 1
            kv_cache_elements = (
                2 *  # K and V
                batch_size *
                num_layers *
                num_attention_heads *
                context_length *
                head_dim
            )
            kv_cache_gb = (kv_cache_elements * 2) / (1024**3)  # 2 bytes per element (float16)
            
            # Activation memory (rough estimate: ~10% of weights)
            activation_overhead_gb = weights_gb * 0.1
            
            # Total estimates for different context lengths
            estimates = {
                "weights_gb": round(weights_gb, 2),
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "quantization_bits": bits,
            }
            
            # Add context-specific estimates
            for ctx_len in [4096, 8192, 16384, 32768]:
                kv_gb = (kv_cache_elements * ctx_len / context_length * 2) / (1024**3)
                total_gb = weights_gb + kv_gb + activation_overhead_gb
                estimates[f"kv_cache_{ctx_len//1024}k_gb"] = round(kv_gb, 2)
                estimates[f"total_{ctx_len//1024}k_gb"] = round(total_gb, 2)
            
            estimates["activation_overhead_gb"] = round(activation_overhead_gb, 2)
            
            logger.info(f"Model memory estimate: {weights_gb:.1f}GB weights, "
                       f"{estimates[f'kv_cache_{context_length//1024}k_gb']:.1f}GB KV cache @ {context_length} ctx")
            
            return estimates
        
        except Exception as e:
            logger.warning(f"Failed to estimate model memory: {e}, using defaults")
            return SystemCapabilities._default_memory_estimate(context_length)
    
    @staticmethod
    def _default_memory_estimate(context_length: int) -> Dict[str, float]:
        """Return default memory estimates when config is unavailable."""
        # Conservative defaults for a ~7B model
        weights_gb = 8.0
        kv_cache_gb = context_length / 8192 * 2.0  # Scale with context length
        activation_gb = 2.0
        
        return {
            "weights_gb": weights_gb,
            "num_layers": 32,
            "hidden_size": 4096,
            "vocab_size": 32000,
            "quantization_bits": 16,
            "kv_cache_4k_gb": 1.0,
            "kv_cache_8k_gb": 2.0,
            "kv_cache_16k_gb": 4.0,
            "kv_cache_32k_gb": 8.0,
            "total_4k_gb": weights_gb + 1.0 + activation_gb,
            "total_8k_gb": weights_gb + 2.0 + activation_gb,
            "total_16k_gb": weights_gb + 4.0 + activation_gb,
            "total_32k_gb": weights_gb + 8.0 + activation_gb,
            "activation_overhead_gb": activation_gb,
        }
