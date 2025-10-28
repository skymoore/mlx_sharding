"""
MLX Shard Peer Server - V2 Zero-Configuration Worker Node

This is a lightweight worker node that:
- Announces itself on the network
- Receives model files from coordinator
- Loads assigned model layers
- Executes inference via Arrow Flight
"""

import logging
import threading
import signal
import sys
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, File, Form, Request
import uvicorn

from shard.zeroconf.discovery import PeerDiscovery
from shard.zeroconf.capabilities import SystemCapabilities
from shard.server.server import serve as flight_serve  # Updated to Flight
from shard.server.utils import load_model
from mlx_lm.tokenizer_utils import load_tokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ShardAssignment:
    """Assignment of model layers to this peer."""

    model_name: str
    start_layer: int
    end_layer: int
    estimated_memory_gb: float

    @classmethod
    def from_dict(cls, data: Dict) -> "ShardAssignment":
        return cls(**data)

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "estimated_memory_gb": self.estimated_memory_gb,
        }


class PeerServer:
    """HTTP server for peer control and file reception."""

    def __init__(
        self,
        grpc_port: int = 50051,  # Now used for Flight port
        http_port: int = 8081,
        cache_dir: str = "~/.cache/mlx-sharding",
        bind_ip: Optional[str] = None,
        max_ram_gb: Optional[float] = None,
    ):
        self.app = FastAPI(title="MLX Shard Peer", version="2.0.0")
        self.grpc_port = grpc_port  # Renamed but kept for Flight
        self.http_port = http_port
        self.bind_ip = bind_ip
        self.max_ram_gb = max_ram_gb
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.discovery = PeerDiscovery(role="peer", bind_ip=bind_ip)
        self.capabilities = SystemCapabilities.get_capabilities(max_ram_gb=max_ram_gb)
        self.state = "idle"  # idle, receiving_files, loading_model, ready, error
        self.model = None
        self.tokenizer = None  # Store tokenizer for chat template support
        self.model_config = None  # Store model config
        self.assignment: Optional[ShardAssignment] = None
        self.coordinator_id: Optional[str] = None
        self.file_buffers: Dict[str, Dict] = {}  # For chunked file reception
        self.local_model_path: Optional[str] = (
            None  # For localhost peers using local files
        )

        # Flight server thread (replaced gRPC)
        self.flight_thread: Optional[threading.Thread] = None

        self._setup_routes()

        logger.info(f"Peer server initialized (Flight={grpc_port}, HTTP={http_port})")
        if bind_ip:
            logger.info(f"Binding to IP: {bind_ip}")
        if max_ram_gb:
            logger.info(f"RAM limit: {max_ram_gb:.1f}GB (capped)")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(
            f"Capabilities: {self.capabilities['ram_available_gb']:.1f}GB RAM, "
            f"{self.capabilities['cpu_cores']} cores"
        )

    def _setup_routes(self):
        """Setup HTTP API routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "ok", "state": self.state}

        @self.app.get("/api/status")
        async def get_status():
            """Return current peer status."""
            return {
                "state": self.state,
                "capabilities": self.capabilities,
                "model_loaded": self.model is not None,
                "assignment": self.assignment.to_dict() if self.assignment else None,
                "coordinator_id": self.coordinator_id,
            }

        @self.app.post("/api/claim")
        async def claim_peer(coordinator_id: str):
            """Claim this peer for a coordinator."""
            if self.coordinator_id and self.coordinator_id != coordinator_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"Already claimed by coordinator {self.coordinator_id}",
                )

            self.coordinator_id = coordinator_id
            logger.info(f"Claimed by coordinator {coordinator_id[:8]}")

            return {"success": True, "message": "Peer claimed"}

        @self.app.post("/api/unclaim")
        async def unclaim_peer(coordinator_id: Optional[str] = None):
            """Release this peer from coordinator claim."""
            if coordinator_id and self.coordinator_id != coordinator_id:
                raise HTTPException(
                    status_code=403,
                    detail=f"Cannot unclaim: claimed by different coordinator {self.coordinator_id}",
                )

            old_coordinator = self.coordinator_id
            self.coordinator_id = None

            # Automatically unload model when unclaimed
            if self.model is not None:
                logger.info("Unloading model due to unclaim")
                self.model = None
                self.tokenizer = None
                self.model_config = None
                self.assignment = None
                self.state = "idle"
                self.discovery.update_status("idle", model_loaded="", layers_loaded="")

            logger.info(
                f"Unclaimed from coordinator {old_coordinator[:8] if old_coordinator else 'none'}"
            )

            return {"success": True, "message": "Peer unclaimed"}

        @self.app.post("/api/set_local_model_path")
        async def set_local_model_path(request: Request):
            """Set local model path for localhost peers (avoids file transfer)."""
            try:
                data = await request.json()
                model_path = data.get("model_path")

                if not model_path:
                    raise HTTPException(status_code=400, detail="model_path required")

                from pathlib import Path

                if not Path(model_path).exists():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model path does not exist: {model_path}",
                    )

                self.local_model_path = model_path
                logger.info(f"Local model path set to: {model_path}")

                return {
                    "success": True,
                    "message": "Local model path set",
                    "path": model_path,
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error setting local model path: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/receive_file_chunk")
        async def receive_file_chunk(
            model_name: str = Form(...),
            file_name: str = Form(...),
            chunk_num: int = Form(...),
            total_chunks: int = Form(...),
            chunk_data: bytes = File(...),
        ):
            """Receive a file chunk."""
            try:
                self.state = "receiving_files"

                # Create model cache directory
                model_cache = self.cache_dir / model_name
                model_cache.mkdir(parents=True, exist_ok=True)

                # Initialize buffer for this file
                file_key = f"{model_name}/{file_name}"
                if file_key not in self.file_buffers:
                    self.file_buffers[file_key] = {
                        "chunks": {},
                        "total_chunks": total_chunks,
                        "file_path": model_cache / file_name,
                    }

                # Store chunk
                self.file_buffers[file_key]["chunks"][chunk_num] = chunk_data

                logger.debug(
                    f"Received chunk {chunk_num+1}/{total_chunks} for {file_name}"
                )

                # Check if all chunks received
                if len(self.file_buffers[file_key]["chunks"]) == total_chunks:
                    # Reassemble file
                    file_path = self.file_buffers[file_key]["file_path"]

                    import hashlib

                    hasher = hashlib.sha256()

                    with open(file_path, "wb") as f:
                        for i in range(total_chunks):
                            chunk = self.file_buffers[file_key]["chunks"][i]
                            f.write(chunk)
                            hasher.update(chunk)

                    # Save hash to metadata file
                    hash_file = model_cache / ".hashes.json"
                    hashes = {}
                    if hash_file.exists():
                        import json

                        with open(hash_file, "r") as f:
                            hashes = json.load(f)

                    hashes[file_name] = hasher.hexdigest()

                    import json

                    with open(hash_file, "w") as f:
                        json.dump(hashes, f, indent=2)

                    # Clean up buffer
                    del self.file_buffers[file_key]

                    logger.info(f"✓ File complete: {file_name}")

                    return {
                        "success": True,
                        "message": f"File {file_name} complete",
                        "complete": True,
                    }

                return {
                    "success": True,
                    "message": f"Chunk {chunk_num+1}/{total_chunks} received",
                    "complete": False,
                }

            except Exception as e:
                logger.error(f"Error receiving file chunk: {e}", exc_info=True)
                self.state = "error"
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/cache/check")
        async def check_cache(model: str):
            """Check which files are already cached."""
            import hashlib
            import json

            model_cache = self.cache_dir / model
            if not model_cache.exists():
                return {}

            # Try to load cached hashes first
            hash_file = model_cache / ".hashes.json"
            if hash_file.exists():
                try:
                    with open(hash_file, "r") as f:
                        hashes = json.load(f)
                    logger.info(
                        f"Cache check for {model}: {len(hashes)} files cached (from metadata)"
                    )
                    return hashes
                except Exception as e:
                    logger.warning(f"Failed to load hash cache: {e}, recomputing...")

            # Fallback: compute hashes (slow for large models)
            logger.info(
                f"Computing hashes for {model} (this may take a while for large models)..."
            )
            hashes = {}
            for file_path in model_cache.glob("*"):
                if file_path.is_file() and file_path.name != ".hashes.json":
                    # Compute hash
                    hasher = hashlib.sha256()
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            hasher.update(chunk)
                    hashes[file_path.name] = hasher.hexdigest()

            # Save computed hashes for next time
            try:
                with open(hash_file, "w") as f:
                    json.dump(hashes, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save hash cache: {e}")

            logger.info(f"Cache check for {model}: {len(hashes)} files cached")
            return hashes

        @self.app.post("/api/load_model")
        async def load_model_endpoint(
            model_name: str = Form(...),
            start_layer: int = Form(...),
            end_layer: int = Form(...),
            estimated_memory_gb: float = Form(...),
            coordinator_id: str = Form(...),
        ):
            """Load assigned model layers."""
            try:
                # Verify coordinator
                if self.coordinator_id != coordinator_id:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Not claimed by coordinator {coordinator_id}",
                    )

                self.state = "loading_model"

                # Create assignment
                self.assignment = ShardAssignment(
                    model_name=model_name,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    estimated_memory_gb=estimated_memory_gb,
                )

                # end_layer is exclusive, so actual last layer is end_layer-1
                actual_end_layer = end_layer - 1
                logger.info(
                    f"Loading model: {model_name} layers {start_layer}-{actual_end_layer} (inclusive)"
                )

                # Determine model path - use local path if set (for localhost peers), otherwise use cache
                if self.local_model_path:
                    model_path = Path(self.local_model_path)
                    logger.info(f"Using local model path: {model_path}")
                else:
                    model_path = self.cache_dir / model_name
                    logger.info(f"Using cached model path: {model_path}")

                if not model_path.exists():
                    raise FileNotFoundError(f"Model not found at: {model_path}")

                # Load model and config
                self.model, self.model_config = load_model(
                    str(model_path), start_layer=start_layer, end_layer=end_layer
                )

                # Load tokenizer for chat template support and debugging
                logger.info(f"Loading tokenizer from {model_path}")
                self.tokenizer = load_tokenizer(
                    model_path,
                    eos_token_ids=self.model_config.get("eos_token_id", None),
                )
                logger.info(
                    f"✓ Tokenizer loaded with chat template support: {hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None}"
                )

                # Start Flight server now that model is loaded
                if self.flight_thread is None:
                    self.start_flight_server(str(model_path), start_layer, end_layer)

                self.state = "ready"

                # Update discovery status
                self.discovery.update_status(
                    "ready",
                    model_loaded=model_name,
                    layers_loaded=f"{start_layer}-{end_layer}",
                )

                logger.info(f"✓ Model loaded successfully")

                return {
                    "success": True,
                    "message": "Model loaded",
                    "assignment": self.assignment.to_dict(),
                }

            except Exception as e:
                logger.error(f"Error loading model: {e}", exc_info=True)
                self.state = "error"
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/unload_model")
        async def unload_model():
            """Unload current model."""
            self.model = None
            self.tokenizer = None
            self.model_config = None
            self.assignment = None
            self.coordinator_id = None
            self.state = "idle"

            self.discovery.update_status("idle", model_loaded="", layers_loaded="")

            logger.info("Model unloaded")
            return {"success": True, "message": "Model unloaded"}

    def start_flight_server(self, model_path: str, start_layer: int, end_layer: int):
        """Start Arrow Flight server in background thread (only after model is loaded)."""
        if self.model is None:
            logger.warning("Cannot start Flight server without a loaded model")
            return

        def run_flight():
            # Start Flight server with preloaded model
            flight_serve(
                model_path,
                start_layer,
                end_layer,
                self.grpc_port,
                preloaded_model=self.model,
            )

        self.flight_thread = threading.Thread(target=run_flight, daemon=True)
        self.flight_thread.start()
        logger.info(f"✓ Flight server started on port {self.grpc_port}")

    def start(self):
        """Start peer server."""
        # Announce on network
        self.discovery.announce(self.grpc_port, self.http_port, self.capabilities)
        logger.info(f"✓ Announced on network")

        # Start HTTP server
        logger.info(f"🚀 Peer server starting on 0.0.0.0:{self.http_port}")
        logger.info(f"   State: {self.state}")
        logger.info(f"   Ready to receive assignments")

        uvicorn.run(self.app, host="0.0.0.0", port=self.http_port, log_level="info")

    def stop(self):
        """Stop peer server."""
        logger.info("Stopping peer server...")
        self.discovery.stop()
        logger.info("✓ Peer server stopped")


# Entry point moved to shard.cli.commands.peer
