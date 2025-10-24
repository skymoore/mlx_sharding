"""
Intelligent sharding planner for distributing model layers across peers.
Considers available RAM, KV cache requirements, and network topology.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .discovery import PeerInfo
from .capabilities import SystemCapabilities
from .network import NetworkDiscovery

logger = logging.getLogger(__name__)


class InsufficientMemoryError(Exception):
    """Raised when total available memory is insufficient for the model."""
    pass


class NoViablePeersError(Exception):
    """Raised when no viable peers are available."""
    pass


@dataclass
class ShardAssignment:
    """Assignment of model layers to a specific peer."""
    peer_id: str
    peer_address: str
    grpc_address: str  # Optimal network address for gRPC
    start_layer: int
    end_layer: int
    estimated_memory_gb: float
    has_embedding: bool = False
    has_lm_head: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ShardAssignment":
        return cls(**data)


@dataclass
class ShardingPlan:
    """Complete sharding plan for a model."""
    model_name: str
    total_layers: int
    context_length: int
    shards: List[ShardAssignment]
    total_memory_required_gb: float
    total_memory_available_gb: float
    safety_margin_gb: float
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "total_layers": self.total_layers,
            "context_length": self.context_length,
            "shards": [s.to_dict() for s in self.shards],
            "total_memory_required_gb": self.total_memory_required_gb,
            "total_memory_available_gb": self.total_memory_available_gb,
            "safety_margin_gb": self.safety_margin_gb,
        }
    
    def __str__(self) -> str:
        lines = [
            f"Sharding Plan for {self.model_name}",
            f"  Total Layers: {self.total_layers}",
            f"  Context Length: {self.context_length}",
            f"  Memory Required: {self.total_memory_required_gb:.1f}GB",
            f"  Memory Available: {self.total_memory_available_gb:.1f}GB",
            f"  Safety Margin: {self.safety_margin_gb:.1f}GB",
            f"  Shards: {len(self.shards)}",
            "",
        ]
        
        for i, shard in enumerate(self.shards):
            lines.append(f"  Shard {i+1}:")
            lines.append(f"    Peer: {shard.peer_id[:8]}...")
            lines.append(f"    Address: {shard.grpc_address}")
            lines.append(f"    Layers: {shard.start_layer}-{shard.end_layer} ({shard.end_layer - shard.start_layer} layers)")
            lines.append(f"    Memory: {shard.estimated_memory_gb:.1f}GB")
            if shard.has_embedding:
                lines.append(f"    Components: Embedding")
            if shard.has_lm_head:
                lines.append(f"    Components: LM Head")
            lines.append("")
        
        return "\n".join(lines)


class ShardingPlanner:
    """
    Calculate optimal model sharding across peers.
    
    Strategy:
    1. Estimate total memory needed (weights + KV cache + overhead)
    2. Reserve memory for KV cache and activations on each peer
    3. Distribute layers proportionally to available RAM
    4. Ensure first peer gets embedding, last peer gets LM head
    5. Validate total memory fits with safety margin
    6. Select optimal network paths for gRPC communication
    """
    
    def __init__(self, model_path_or_repo: str, peers: List[PeerInfo],
                 context_length: int = 8192, safety_margin: float = 0.15):
        """
        Initialize sharding planner.
        
        Args:
            model_path_or_repo: Path to model or HuggingFace repo
            peers: List of available peers
            context_length: Target context length for KV cache estimation
            safety_margin: Safety margin as fraction of total memory (default 15%)
        """
        self.model_path_or_repo = model_path_or_repo
        self.peers = sorted(peers, key=lambda p: p.ram_available_gb, reverse=True)
        self.context_length = context_length
        self.safety_margin = safety_margin
        
        # Get model memory estimates
        self.memory_estimates = SystemCapabilities.estimate_model_memory(
            model_path_or_repo, context_length
        )
        
        self.total_layers = self.memory_estimates.get("num_layers", 32)
        
        logger.info(f"Initialized ShardingPlanner for {model_path_or_repo}")
        logger.info(f"  Total layers: {self.total_layers}")
        logger.info(f"  Context length: {context_length}")
        logger.info(f"  Model weights: {self.memory_estimates['weights_gb']:.1f}GB")
        logger.info(f"  KV cache @ {context_length}: {self.memory_estimates.get(f'kv_cache_{context_length//1024}k_gb', 0):.1f}GB")
        logger.info(f"  Available peers: {len(self.peers)}")
        for peer in self.peers:
            logger.info(f"    - {peer.id[:8]}: {peer.ram_available_gb:.1f}GB RAM")
    
    def calculate_sharding_plan(self) -> ShardingPlan:
        """
        Calculate optimal layer distribution.
        
        Returns:
            ShardingPlan with layer assignments
            
        Raises:
            InsufficientMemoryError: If total memory is insufficient
            NoViablePeersError: If no viable peers available
        """
        if not self.peers:
            raise NoViablePeersError("No peers available for sharding")
        
        # Calculate memory requirements
        weights_gb = self.memory_estimates["weights_gb"]
        kv_cache_key = f"kv_cache_{self.context_length//1024}k_gb"
        kv_cache_total_gb = self.memory_estimates.get(kv_cache_key, 2.0)
        activation_overhead_gb = self.memory_estimates.get("activation_overhead_gb", 2.0)
        
        # Total memory needed
        total_memory_needed = weights_gb + kv_cache_total_gb + activation_overhead_gb
        
        # Total memory available
        total_memory_available = sum(p.ram_available_gb for p in self.peers)
        
        # Apply safety margin
        usable_memory = total_memory_available * (1 - self.safety_margin)
        
        logger.info(f"Memory analysis:")
        logger.info(f"  Weights: {weights_gb:.1f}GB")
        logger.info(f"  KV cache: {kv_cache_total_gb:.1f}GB")
        logger.info(f"  Activation overhead: {activation_overhead_gb:.1f}GB")
        logger.info(f"  Total needed: {total_memory_needed:.1f}GB")
        logger.info(f"  Total available: {total_memory_available:.1f}GB")
        logger.info(f"  Usable (with {self.safety_margin*100:.0f}% margin): {usable_memory:.1f}GB")
        
        # Check if we have enough memory
        if total_memory_needed > usable_memory:
            raise InsufficientMemoryError(
                f"Insufficient memory for model. "
                f"Need {total_memory_needed:.1f}GB, "
                f"have {usable_memory:.1f}GB usable across {len(self.peers)} peer(s). "
                f"Options:\n"
                f"  1. Add more peers with available RAM\n"
                f"  2. Use a smaller/quantized model\n"
                f"  3. Reduce --context-length to decrease KV cache overhead\n"
                f"  4. Reduce safety margin (risky)"
            )
        
        # Calculate per-layer memory
        layer_memory_gb = weights_gb / self.total_layers
        
        # Distribute KV cache and overhead across peers
        kv_cache_per_peer = kv_cache_total_gb / len(self.peers)
        overhead_per_peer = activation_overhead_gb / len(self.peers)
        
        # Calculate how many layers each peer can handle
        peer_layer_capacity = []
        for peer in self.peers:
            # Apply safety margin to each peer
            available = peer.ram_available_gb * (1 - self.safety_margin)
            # Reserve space for KV cache and overhead
            available_for_layers = available - kv_cache_per_peer - overhead_per_peer
            # Calculate max layers
            max_layers = max(1, int(available_for_layers / layer_memory_gb))
            peer_layer_capacity.append(max_layers)
            
            logger.debug(f"Peer {peer.id[:8]}: {available:.1f}GB available, "
                        f"can handle ~{max_layers} layers")
        
        # Validate we can fit the model
        if sum(peer_layer_capacity) < self.total_layers:
            raise InsufficientMemoryError(
                f"Cannot fit {self.total_layers} layers across {len(self.peers)} peer(s). "
                f"Total capacity: {sum(peer_layer_capacity)} layers. "
                f"Need {self.total_layers * layer_memory_gb:.1f}GB for layers, "
                f"have {sum(p.ram_available_gb for p in self.peers):.1f}GB total."
            )
        
        # Distribute layers across peers
        shards = []
        current_layer = 0
        
        for i, (peer, capacity) in enumerate(zip(self.peers, peer_layer_capacity)):
            # Calculate layers for this peer
            if i == len(self.peers) - 1:
                # Last peer gets remaining layers
                end_layer = self.total_layers
            else:
                # Distribute proportionally, but don't exceed capacity
                layers_for_peer = min(capacity, self.total_layers - current_layer)
                end_layer = current_layer + layers_for_peer
            
            # Calculate memory for this shard
            num_layers = end_layer - current_layer
            shard_memory = (layer_memory_gb * num_layers + 
                          kv_cache_per_peer + overhead_per_peer)
            
            # Determine optimal gRPC address (use discovery address for now)
            # TODO: Integrate network path selection
            grpc_address = f"{peer.host}:{peer.grpc_port}"
            
            shard = ShardAssignment(
                peer_id=peer.id,
                peer_address=peer.address,
                grpc_address=grpc_address,
                start_layer=current_layer,
                end_layer=end_layer,
                estimated_memory_gb=shard_memory,
                has_embedding=(current_layer == 0),
                has_lm_head=(end_layer == self.total_layers),
            )
            
            shards.append(shard)
            
            logger.info(f"Assigned layers {current_layer}-{end_layer} to peer {peer.id[:8]} "
                       f"({shard_memory:.1f}GB)")
            
            current_layer = end_layer
            
            if current_layer >= self.total_layers:
                break
        
        # Create plan
        plan = ShardingPlan(
            model_name=self.model_path_or_repo,
            total_layers=self.total_layers,
            context_length=self.context_length,
            shards=shards,
            total_memory_required_gb=total_memory_needed,
            total_memory_available_gb=total_memory_available,
            safety_margin_gb=total_memory_available * self.safety_margin,
        )
        
        logger.info(f"✓ Sharding plan created with {len(shards)} shard(s)")
        
        return plan
    
    def optimize_network_paths(self, plan: ShardingPlan, 
                               coordinator_interfaces: List) -> ShardingPlan:
        """
        Optimize network paths for gRPC communication.
        
        Args:
            plan: Initial sharding plan
            coordinator_interfaces: Network interfaces of coordinator
            
        Returns:
            Updated plan with optimal gRPC addresses
        """
        # For each shard, find the best network path
        for shard in plan.shards:
            peer = next((p for p in self.peers if p.id == shard.peer_id), None)
            if not peer or not peer.network_interfaces:
                continue
            
            # Convert peer network interfaces from dict to NetworkInterface objects
            from .network import NetworkInterface
            peer_interfaces = [
                NetworkInterface.from_dict(iface_data)
                for iface_data in peer.network_interfaces.values()
            ]
            
            # Find best path
            best_path = NetworkDiscovery.select_best_network(
                coordinator_interfaces,
                peer_interfaces,
                peer.grpc_port
            )
            
            if best_path:
                local_ip, remote_ip = best_path
                shard.grpc_address = f"{remote_ip}:{peer.grpc_port}"
                logger.info(f"Optimized path for {shard.peer_id[:8]}: {shard.grpc_address}")
        
        return plan
