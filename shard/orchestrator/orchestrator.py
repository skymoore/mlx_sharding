"""
API Server Orchestrator - Coordinates the entire setup process.
Discovers peers, plans sharding, distributes files, and monitors progress.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Optional, Dict, List
from pathlib import Path

from shard.zeroconf.discovery import PeerDiscovery, PeerInfo
from shard.zeroconf.capabilities import SystemCapabilities
from shard.orchestrator.planner import (
    ShardingPlanner,
    ShardingPlan,
    ShardAssignment,
    InsufficientMemoryError,
)
from shard.orchestrator.distributor import ModelFileDistributor
from mlx_lm.utils import hf_repo_to_path

logger = logging.getLogger(__name__)


class SetupError(Exception):
    """Raised when setup fails."""

    pass


class ProgressTracker:
    """
    Track and broadcast setup progress.
    Provides real-time updates via callbacks.
    """

    def __init__(self):
        self.phase = "initializing"
        self.peers: Dict[str, Dict] = {}
        self.overall_progress = 0.0
        self.callbacks: List = []
        self.start_time = time.time()

    def add_callback(self, callback):
        """Add a progress callback."""
        self.callbacks.append(callback)

    def update_phase(self, phase: str, message: str = ""):
        """Update current setup phase."""
        self.phase = phase
        elapsed = time.time() - self.start_time

        event = {
            "type": "phase_change",
            "phase": phase,
            "message": message,
            "elapsed": elapsed,
            "timestamp": time.time(),
        }

        logger.info(f"Phase: {phase} - {message}")
        self._broadcast(event)

    def update_peer_status(self, peer_id: str, status: str, **kwargs):
        """Update status of a specific peer."""
        if peer_id not in self.peers:
            self.peers[peer_id] = {}

        self.peers[peer_id].update(
            {"status": status, "timestamp": time.time(), **kwargs}
        )

        event = {"type": "peer_update", "peer_id": peer_id, "status": status, **kwargs}

        self._broadcast(event)
        self._calculate_overall_progress()

    def update_file_transfer(
        self, peer_id: str, file_name: str, status: str, progress: float
    ):
        """Update file transfer progress."""
        event = {
            "type": "file_transfer",
            "peer_id": peer_id,
            "file_name": file_name,
            "status": status,
            "progress": progress,
            "timestamp": time.time(),
        }

        self._broadcast(event)

    def _calculate_overall_progress(self):
        """Calculate overall setup progress."""
        if not self.peers:
            self.overall_progress = 0.0
            return

        # Weight different phases
        phase_weights = {
            "discovered": 10,
            "receiving_files": 40,
            "loading_model": 30,
            "ready": 20,
        }

        total_progress = 0
        for peer_data in self.peers.values():
            status = peer_data.get("status", "idle")
            total_progress += phase_weights.get(status, 0)

        self.overall_progress = total_progress / len(self.peers)

        event = {
            "type": "overall_progress",
            "progress": self.overall_progress,
            "timestamp": time.time(),
        }

        self._broadcast(event)

    def _broadcast(self, event: Dict):
        """Broadcast event to all callbacks."""
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class APIServerOrchestrator:
    """
    Orchestrates the entire setup process.

    Flow:
    1. Discover peers on network
    2. Assess total resources
    3. Create sharding plan
    4. Distribute model files
    5. Instruct peers to load models
    6. Monitor until all ready
    """

    def __init__(
        self,
        model_name: str,
        context_length: int = 8192,
        safety_margin: float = 0.15,
        discovery_timeout: float = 10.0,
        grpc_port: int = 50051,
        http_port: int = 8081,
    ):
        """
        Initialize orchestrator.

        Args:
            model_name: HuggingFace model name or local path
            context_length: Target context length for KV cache
            safety_margin: Memory safety margin (default 15%)
            discovery_timeout: Peer discovery timeout in seconds
            grpc_port: gRPC port for announcement (default: 50051)
            http_port: HTTP port for announcement (default: 8081)
        """
        self.model_name = model_name
        self.context_length = context_length
        self.safety_margin = safety_margin
        self.discovery_timeout = discovery_timeout
        self.grpc_port = grpc_port
        self.http_port = http_port

        self.progress = ProgressTracker()
        self.discovery: Optional[PeerDiscovery] = None
        self.plan: Optional[ShardingPlan] = None
        self.model_path: Optional[Path] = None

        logger.info(f"Initialized orchestrator for {model_name}")

    async def setup(self) -> ShardingPlan:
        """
        Execute full setup orchestration.

        Returns:
            ShardingPlan with all peers ready

        Raises:
            SetupError: If setup fails
        """
        try:
            # Phase 1: Discovery
            self.progress.update_phase("discovering", "Discovering peers on network...")
            peers = await self._discover_peers()

            # Phase 2: Planning
            self.progress.update_phase("planning", "Creating optimal sharding plan...")
            self.plan = await self._create_sharding_plan(peers)

            # Phase 3: File Distribution
            self.progress.update_phase(
                "distributing", "Distributing model files to peers..."
            )
            await self._distribute_files(self.plan, peers)

            # Phase 4: Model Loading
            self.progress.update_phase("loading", "Loading models on peers...")
            await self._load_models(self.plan, peers)

            # Phase 5: Ready
            self.progress.update_phase("ready", "All peers ready for inference!")

            logger.info("✓ Setup complete!")
            return self.plan

        except Exception as e:
            self.progress.update_phase("error", f"Setup failed: {e}")
            logger.error(f"Setup failed: {e}", exc_info=True)
            raise SetupError(f"Setup failed: {e}") from e

    async def _discover_peers(self) -> List[PeerInfo]:
        """Discover available peers."""
        logger.info(f"Discovering peers (timeout={self.discovery_timeout}s)...")

        self.discovery = PeerDiscovery(role="coordinator")

        # Always announce to enable discovery
        caps = SystemCapabilities.get_capabilities()
        self.discovery.announce(self.grpc_port, self.http_port, caps)
        logger.info("Announced as coordinator (discovery only)")

        # Discover peers
        peers = self.discovery.discover_peers(self.discovery_timeout)

        for peer in peers:
            self.progress.update_peer_status(
                peer.id,
                "discovered",
                address=peer.address,
                ram_gb=peer.ram_available_gb,
                cpu_cores=peer.cpu_cores,
            )

        if not peers:
            raise SetupError(
                "No peers discovered on network. Ensure:\n"
                "  1. Peer servers are running (mlx-shard-peer)\n"
                "  2. Firewall allows mDNS and UDP broadcast\n"
                "  3. All machines are on same network"
            )

        logger.info(f"✓ Discovered {len(peers)} peer(s)")
        return peers

    async def _create_sharding_plan(self, peers: List[PeerInfo]) -> ShardingPlan:
        """Create optimal sharding plan."""
        logger.info("Creating sharding plan...")

        # Download/locate model
        try:
            # Check if it's already a local path
            if Path(self.model_name).exists():
                self.model_path = Path(self.model_name)
                logger.info(f"Using local model path: {self.model_path}")
            else:
                # Try to download from HuggingFace
                self.model_path = hf_repo_to_path(self.model_name)
                logger.info(f"Downloaded model to: {self.model_path}")
        except Exception as e:
            raise SetupError(f"Failed to locate model {self.model_name}: {e}")

        # Create planner
        try:
            planner = ShardingPlanner(
                str(self.model_path), peers, self.context_length, self.safety_margin
            )

            plan = planner.calculate_sharding_plan()

            logger.info(f"✓ Sharding plan created:")
            logger.info(f"  Total layers: {plan.total_layers}")
            logger.info(f"  Shards: {len(plan.shards)}")
            logger.info(f"  Memory required: {plan.total_memory_required_gb:.1f}GB")
            logger.info(f"  Memory available: {plan.total_memory_available_gb:.1f}GB")

            return plan

        except InsufficientMemoryError as e:
            raise SetupError(str(e))

    async def _distribute_files(self, plan: ShardingPlan, peers: List[PeerInfo]):
        """Distribute model files to all peers."""
        logger.info("Distributing model files...")

        distributor = ModelFileDistributor(str(self.model_path))

        # Progress callback
        def progress_callback(peer_id, file_name, status, progress):
            self.progress.update_file_transfer(peer_id, file_name, status, progress)

        # Distribute
        results = await distributor.distribute_to_peers(
            peers, plan.shards, progress_callback
        )

        # Check results
        failed = [peer_id for peer_id, success in results.items() if not success]
        if failed:
            raise SetupError(f"File distribution failed for peers: {failed}")

        logger.info("✓ All files distributed")

    async def _claim_peers(self, peers: List[PeerInfo]):
        """Claim all peers for this coordinator."""
        logger.info("Claiming peers...")

        tasks = []
        for peer in peers:
            task = self._claim_peer(peer)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            raise SetupError(f"Failed to claim peers: {errors[0]}")

        logger.info("✓ All peers claimed")

    async def _claim_peer(self, peer: PeerInfo):
        """Claim a single peer."""
        try:
            coordinator_id = self.discovery.peer_id if self.discovery else "unknown"
            url = f"http://{peer.host}:{peer.http_port}/api/claim?coordinator_id={coordinator_id}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    result = await resp.json()

                    if not result.get("success"):
                        raise SetupError(
                            f"Failed to claim peer {peer.id[:8]}: {result}"
                        )

                    logger.info(f"✓ Claimed peer {peer.id[:8]}")

        except Exception as e:
            logger.error(f"Failed to claim peer {peer.id[:8]}: {e}")
            raise

    async def _load_models(self, plan: ShardingPlan, peers: List[PeerInfo]):
        """Instruct all peers to load their assigned layers."""
        logger.info("Loading models on peers...")

        # First, claim all peers
        await self._claim_peers(peers)

        tasks = []
        for shard in plan.shards:
            peer = next((p for p in peers if p.id == shard.peer_id), None)
            if not peer:
                logger.warning(f"Peer {shard.peer_id} not found")
                continue

            task = self._load_model_on_peer(peer, shard, plan.model_name)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            raise SetupError(f"Model loading failed: {errors[0]}")

        logger.info("✓ All models loaded")

    async def _load_model_on_peer(
        self, peer: PeerInfo, shard: ShardAssignment, model_name: str
    ):
        """Send load command to a peer."""
        logger.info(f"Loading model on peer {peer.id[:8]}...")

        self.progress.update_peer_status(peer.id, "loading_model")

        try:
            url = f"http://{peer.host}:{peer.http_port}/api/load_model"

            # Extract model name from path
            model_basename = Path(model_name).name

            data = {
                "model_name": model_basename,
                "start_layer": shard.start_layer,
                "end_layer": shard.end_layer,
                "estimated_memory_gb": shard.estimated_memory_gb,
                "coordinator_id": (
                    self.discovery.peer_id if self.discovery else "unknown"
                ),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, data=data, timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    result = await resp.json()

                    if result.get("success"):
                        self.progress.update_peer_status(peer.id, "ready")
                        logger.info(f"✓ Peer {peer.id[:8]} ready")
                    else:
                        error = result.get("error", "Unknown error")
                        self.progress.update_peer_status(peer.id, "error", error=error)
                        raise SetupError(f"Peer {peer.id[:8]} failed to load: {error}")

        except Exception as e:
            self.progress.update_peer_status(peer.id, "error", error=str(e))
            raise

    async def unclaim_peers(self, peers: List[PeerInfo]):
        """Release all peers from this coordinator's claim."""
        logger.info("Unclaiming peers...")
        
        tasks = []
        for peer in peers:
            task = self._unclaim_peer(peer)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"✓ Unclaimed {success_count}/{len(peers)} peer(s)")
        
    async def _unclaim_peer(self, peer: PeerInfo):
        """Unclaim a single peer."""
        try:
            coordinator_id = self.discovery.peer_id if self.discovery else "unknown"
            url = f"http://{peer.host}:{peer.http_port}/api/unclaim?coordinator_id={coordinator_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    result = await resp.json()
                    
                    if result.get("success"):
                        logger.info(f"✓ Unclaimed peer {peer.id[:8]}")
                    else:
                        logger.warning(f"Failed to unclaim peer {peer.id[:8]}: {result}")
                        
        except Exception as e:
            logger.warning(f"Failed to unclaim peer {peer.id[:8]}: {e}")

    def cleanup(self):
        """Cleanup resources."""
        if self.discovery:
            try:
                self.discovery.stop()
            except Exception as e:
                logger.warning(f"Discovery cleanup error: {e}")
