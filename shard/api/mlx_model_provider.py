from typing import Optional
from shard.api.grpc import FlightConnectionPool
from pathlib import Path
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import hf_repo_to_path
from logging import getLogger
from shard.api.tool_calling import create_tool_call_manager
from shard.server.utils import (
    create_coordinator_generate_step,
    load_model,
)
import time
import json
import mlx.core as mx

log = getLogger(__name__)


class MLXModelProvider:
    """Manages model loading and generation with distributed inference."""

    def __init__(
        self,
        model_path: str,
        start_layer: Optional[int],
        end_layer: Optional[int],
        connection_pool: Optional[FlightConnectionPool] = None,
    ):
        self.model_path = model_path
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.connection_pool = connection_pool

        # Load tokenizer (always needed)
        tokenizer_path = (
            Path(model_path)
            if Path(model_path).exists()
            else hf_repo_to_path(model_path)
        )
        self.tokenizer = load_tokenizer(tokenizer_path)

        config_path = tokenizer_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                # Set eos_token_ids from config if present
                if "eos_token_id" in config:
                    eos_ids = config["eos_token_id"]
                    # Handle both single ID and list of IDs
                    if isinstance(eos_ids, list):
                        self.tokenizer.eos_token_ids = set(eos_ids)
                    else:
                        self.tokenizer.eos_token_ids = {eos_ids}
                    log.info(
                        f"✓ Set tokenizer.eos_token_ids from config: {self.tokenizer.eos_token_ids}"
                    )

        # Coordinator mode: no local layers
        if start_layer is None and end_layer is None:
            log.info(
                "Coordinator-only mode: no local model, pipeline coordination only"
            )
            self.model = None
            # Don't create generate_step here - will be created per-request

            # Get model type from config for tool calling and stop tokens
            config_path = tokenizer_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    self.model_type = config.get("model_type", "unknown")
            else:
                self.model_type = "unknown"
        else:
            # Peer mode: load local layers (not used in coordinator-only mode)
            log.info(f"Peer mode: loading layers {start_layer}-{end_layer}")
            self.model, model_config = load_model(
                model_path, start_layer=start_layer, end_layer=end_layer
            )
            # For peer mode, would need stubs - not implemented in this coordinator-only setup
            self.model_type = model_config.get("model_type", "unknown")

        # Model info
        self.model_name = (
            Path(model_path).name if Path(model_path).exists() else model_path
        )
        self.created = int(time.time())

        # Initialize tool call manager
        self.tool_manager = create_tool_call_manager(self.model_type)
        log.info(f"✓ Model provider initialized: {self.model_name}")
        log.info(f"✓ Model type: {self.model_type}")
        log.info(
            f"✓ Tool calling enabled with {self.tool_manager.parser.__class__.__name__}"
        )

        # Chat templates are handled by the tokenizer's built-in apply_chat_template()
        log.info(
            f"✓ Chat template support: {hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None}"
        )

    def get_stop_token_ids(self) -> set:
        """
        Get token IDs for stop sequences to check during generation.
        Uses the tokenizer's built-in eos_token_ids which is the correct way
        to determine when generation should stop.
        """
        # Use tokenizer's eos_token_ids directly - this is what mlx_lm uses
        stop_token_ids = set(self.tokenizer.eos_token_ids)

        log.info(f"🛑 Using tokenizer.eos_token_ids: {stop_token_ids}")
        
        # Decode for debugging
        for token_id in stop_token_ids:
            try:
                decoded = self.tokenizer.decode([token_id])
                log.info(f"🛑 Stop token {token_id}: {repr(decoded)}")
            except:
                pass

        return stop_token_ids

    def generate(self, prompt: mx.array, **kwargs):
        """
        Generate tokens using the distributed model.
        Creates fresh gRPC stubs per request for true concurrency.
        """
        # Get fresh stubs from connection pool for this request
        if self.connection_pool is None:
            raise ValueError("Connection pool not initialized")

        clients = self.connection_pool.create_clients_for_request()

        # 🔍 DEBUG: Log tokenizer EOS configuration before generation
        log.info(f"🔍 Tokenizer EOS token IDs before generation: {self.tokenizer.eos_token_ids}")
        for eos_id in self.tokenizer.eos_token_ids:
            try:
                decoded = self.tokenizer.decode([eos_id])
                log.info(f"🔍 EOS token {eos_id} decodes to: {repr(decoded)}")
            except:
                pass

        # Create generate_step function with fresh clients and tokenizer
        generate_step = create_coordinator_generate_step(clients, self.tokenizer)

        # Wrap the generator to close clients after exhaustion
        def generator_with_cleanup():
            generator = None
            try:
                # Create the generator
                generator = generate_step(
                    prompt=prompt,
                    temp=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 1.0),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                    repetition_context_size=kwargs.get("repetition_context_size", 20),
                    max_tokens=kwargs.get("max_tokens", 256),
                )
                
                # Yield all items from the generator
                for item in generator:
                    yield item
                    
            except GeneratorExit:
                # Generator was closed early by consumer
                log.debug("Generator closed early by consumer")
                raise
            except Exception as e:
                # Log and re-raise any errors during generation
                log.error(f"Error during generation: {e}", exc_info=True)
                raise
            finally:
                # Ensure generator is closed before cleaning up clients
                if generator is not None:
                    try:
                        generator.close()
                    except:
                        pass
                
                # Clean up clients after generator is fully exhausted
                log.debug("Cleaning up Flight clients")
                for i, client in enumerate(clients):
                    try:
                        client.close()
                        log.debug(f"Closed client {i}")
                    except Exception as e:
                        log.warning(f"Error closing client {i}: {e}")

        return generator_with_cleanup()
