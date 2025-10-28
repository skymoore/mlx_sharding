"""
Intelligent sharding planner for distributing model layers across peers.
Considers available RAM, KV cache requirements, and network topology.
"""

import logging
from typing import List, Dict
from dataclasses import dataclass, asdict

from shard.zeroconf.discovery import PeerInfo
from shard.zeroconf.capabilities import SystemCapabilities
from shard.zeroconf.network import NetworkDiscovery

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
            lines.append(
                f"    Layers: {shard.start_layer}-{shard.end_layer} ({shard.end_layer - shard.start_layer} layers)"
            )
            lines.append(f"    Memory: {shard.estimated_memory_gb:.1f}GB")
            if shard.has_embedding:
                lines.append("    Components: Embedding")
            if shard.has_lm_head:
                lines.append("    Components: LM Head")
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

    def __init__(
        self,
        model_path_or_repo: str,
        peers: List[PeerInfo],
        context_length: int = 8192,
        safety_margin: float = 0.10,
        resource_strategy: str = "fewest-nodes",
    ):
        """
        Initialize sharding planner.

        Args:
            model_path_or_repo: Path to model or HuggingFace repo
            peers: List of available peers
            context_length: Target context length for KV cache estimation
            safety_margin: Safety margin as fraction of total memory (default 10%)
            resource_strategy: Resource allocation strategy - "fewest-nodes" or "proportionally"
        """
        self.model_path_or_repo = model_path_or_repo
        self.peers = sorted(peers, key=lambda p: p.ram_available_gb, reverse=True)
        self.context_length = context_length
        self.safety_margin = safety_margin
        self.resource_strategy = resource_strategy

        # Get model memory estimates
        self.memory_estimates = SystemCapabilities.estimate_model_memory(
            model_path_or_repo, context_length
        )

        self.total_layers = self.memory_estimates.get("num_layers", 32)

        logger.info(f"Initialized ShardingPlanner for {model_path_or_repo}")
        logger.info(f"  Total layers: {self.total_layers}")
        logger.info(f"  Requested context length: {context_length}")
        logger.info(f"  Model weights: {self.memory_estimates['weights_gb']:.1f}GB")
        logger.info(
            f"  KV cache @ {context_length}: {self.memory_estimates.get(f'kv_cache_{context_length//1024}k_gb', 0):.1f}GB"
        )
        logger.info(f"  Available peers: {len(self.peers)}")
        for peer in self.peers:
            logger.info(f"    - {peer.id[:8]}: {peer.ram_available_gb:.1f}GB RAM")
    
    def _calculate_max_context_length(self, usable_memory: float) -> int:
        """
        Calculate the maximum context length that fits in available memory.
        
        Args:
            usable_memory: Total usable memory in GB (after safety margin)
            
        Returns:
            Maximum context length in tokens
        """
        weights_gb = self.memory_estimates["weights_gb"]
        activation_overhead_gb = self.memory_estimates.get("activation_overhead_gb", 2.0)
        
        # Memory available for KV cache
        memory_for_kv = usable_memory - weights_gb - activation_overhead_gb
        
        if memory_for_kv <= 0:
            return 0
        
        # Get model architecture parameters for KV cache calculation
        num_layers = self.memory_estimates.get("num_layers", 32)
        num_key_value_heads = self.memory_estimates.get("num_key_value_heads", 
                                                         self.memory_estimates.get("num_attention_heads", 32))
        head_dim = self.memory_estimates.get("head_dim", 128)
        
        # KV cache formula: 2 * batch_size * num_layers * num_kv_heads * context_length * head_dim * 2 bytes
        # Solve for context_length: context_length = memory_for_kv_bytes / (2 * 1 * num_layers * num_kv_heads * head_dim * 2)
        batch_size = 1
        bytes_per_element = 2  # float16
        
        kv_cache_bytes_available = memory_for_kv * (1024**3)
        max_context = kv_cache_bytes_available / (2 * batch_size * num_layers * num_key_value_heads * head_dim * bytes_per_element)
        
        # Round down to nearest 1024 for cleaner numbers
        max_context = int(max_context // 1024) * 1024
        
        return max(1024, max_context)  # Minimum 1024 tokens

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
        activation_overhead_gb = self.memory_estimates.get(
            "activation_overhead_gb", 2.0
        )

        # Total memory needed
        total_memory_needed = weights_gb + kv_cache_total_gb + activation_overhead_gb

        # Total memory available
        total_memory_available = sum(p.ram_available_gb for p in self.peers)

        # Apply safety margin
        usable_memory = total_memory_available * (1 - self.safety_margin)

        logger.info("Memory analysis:")
        logger.info(f"  Weights: {weights_gb:.1f}GB")
        logger.info(f"  KV cache: {kv_cache_total_gb:.1f}GB")
        logger.info(f"  Activation overhead: {activation_overhead_gb:.1f}GB")
        logger.info(f"  Total needed: {total_memory_needed:.1f}GB")
        logger.info(f"  Total available: {total_memory_available:.1f}GB")
        logger.info(
            f"  Usable (with {self.safety_margin*100:.0f}% margin): {usable_memory:.1f}GB"
        )

        # Check if we have enough memory
        if total_memory_needed > usable_memory:
            # Try to calculate maximum context length that fits
            max_context = self._calculate_max_context_length(usable_memory)
            
            if max_context < 1024:
                # Not enough memory even for minimal context
                raise InsufficientMemoryError(
                    "Insufficient memory for model. "
                    f"Need {total_memory_needed:.1f}GB, "
                    f"have {usable_memory:.1f}GB usable across {len(self.peers)} peer(s). "
                    f"Not enough memory even for minimal context length (1024 tokens). "
                    f"Options:\n"
                    f"  1. Add more peers with available RAM\n"
                    f"  2. Use a smaller/quantized model\n"
                    f"  3. Reduce safety margin (risky)"
                )
            
            # Auto-reduce context length
            logger.warning("=" * 80)
            logger.warning(f"⚠️  NOT ENOUGH RAM FOR REQUESTED CONTEXT LENGTH")
            logger.warning(f"   Requested: {self.context_length:,} tokens ({total_memory_needed:.1f}GB needed)")
            logger.warning(f"   Available: {usable_memory:.1f}GB")
            logger.warning(f"   Reducing context length to: {max_context:,} tokens")
            logger.warning("=" * 80)
            
            # Update context length and recalculate memory requirements
            self.context_length = max_context
            
            # Re-estimate memory with new context length
            self.memory_estimates = SystemCapabilities.estimate_model_memory(
                self.model_path_or_repo, self.context_length
            )
            
            # Recalculate memory requirements
            kv_cache_key = f"kv_cache_{self.context_length//1024}k_gb"
            kv_cache_total_gb = self.memory_estimates.get(kv_cache_key, 2.0)
            total_memory_needed = weights_gb + kv_cache_total_gb + activation_overhead_gb
            
            logger.info(f"Updated memory requirements:")
            logger.info(f"  KV cache @ {self.context_length:,}: {kv_cache_total_gb:.1f}GB")
            logger.info(f"  Total needed: {total_memory_needed:.1f}GB")
            logger.info(f"  Usable memory: {usable_memory:.1f}GB")
            logger.info(f"  Margin: {usable_memory - total_memory_needed:.1f}GB")

        # Calculate per-layer memory
        layer_memory_gb = weights_gb / self.total_layers

        # Distribute KV cache and overhead across peers
        kv_cache_per_peer = kv_cache_total_gb / len(self.peers)
        overhead_per_peer = activation_overhead_gb / len(self.peers)

        # Calculate how many layers each peer can handle
        import math
        peer_layer_capacity = []
        
        logger.info("Per-peer layer capacity calculation:")
        for peer in self.peers:
            # Apply safety margin to each peer
            available = peer.ram_available_gb * (1 - self.safety_margin)
            # Reserve space for KV cache and overhead
            available_for_layers = available - kv_cache_per_peer - overhead_per_peer
            # Calculate max layers
            # Use round instead of floor to be less conservative when very close
            max_layers_float = available_for_layers / layer_memory_gb
            max_layers = max(1, round(max_layers_float))
            peer_layer_capacity.append(max_layers)

            logger.info(
                f"  Peer {peer.id[:8]} ({peer.ram_available_gb:.1f}GB): "
                f"{available:.1f}GB usable → {available_for_layers:.1f}GB for layers → "
                f"{max_layers} layers ({max_layers_float:.2f} exact)"
            )
        
        logger.info(f"  Total capacity: {sum(peer_layer_capacity)} layers (need {self.total_layers})")

        # Validate we can fit the model
        if sum(peer_layer_capacity) < self.total_layers:
            # If we're only 1-2 layers short, try reducing context slightly more
            layers_short = self.total_layers - sum(peer_layer_capacity)
            if layers_short <= 2 and self.context_length > 1024:
                logger.warning(f"Short by {layers_short} layer(s), attempting further context reduction...")
                # Reduce context by 10% and retry
                new_context = int(self.context_length * 0.9 // 1024) * 1024
                new_context = max(1024, new_context)
                
                logger.warning(f"Reducing context from {self.context_length:,} to {new_context:,} tokens")
                self.context_length = new_context
                
                # Re-estimate and recurse (but only once to avoid infinite loop)
                self.memory_estimates = SystemCapabilities.estimate_model_memory(
                    self.model_path_or_repo, self.context_length
                )
                # Recalculate from the beginning
                return self.calculate_sharding_plan()
            
            raise InsufficientMemoryError(
                f"Cannot fit {self.total_layers} layers across {len(self.peers)} peer(s). "
                f"Total capacity: {sum(peer_layer_capacity)} layers. "
                f"Need {self.total_layers * layer_memory_gb:.1f}GB for layers, "
                f"have {sum(p.ram_available_gb for p in self.peers):.1f}GB total."
            )

        # Distribute layers across peers based on strategy
        if self.resource_strategy == "proportionally":
            shards = self._distribute_proportionally(
                peer_layer_capacity,
                layer_memory_gb,
                kv_cache_per_peer,
                overhead_per_peer,
            )
        else:  # fewest-nodes (default)
            shards = self._distribute_fewest_nodes(
                peer_layer_capacity,
                layer_memory_gb,
                kv_cache_per_peer,
                overhead_per_peer,
            )

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

    def _distribute_fewest_nodes(
        self,
        peer_layer_capacity: List[int],
        layer_memory_gb: float,
        kv_cache_per_peer: float,
        overhead_per_peer: float,
    ) -> List[ShardAssignment]:
        """
        Distribute layers using fewest nodes strategy (greedy).
        Uses minimum number of peers needed to fit the model.
        """
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
            shard_memory = (
                layer_memory_gb * num_layers + kv_cache_per_peer + overhead_per_peer
            )

            # Determine optimal gRPC address (use discovery address for now)
            grpc_address = f"{peer.host}:{peer.grpc_port}"

            # Model loading treats end_layer as EXCLUSIVE (like Python range)
            is_last_peer = end_layer == self.total_layers

            shard = ShardAssignment(
                peer_id=peer.id,
                peer_address=peer.address,
                grpc_address=grpc_address,
                start_layer=current_layer,
                end_layer=end_layer,
                estimated_memory_gb=shard_memory,
                has_embedding=(current_layer == 0),
                has_lm_head=is_last_peer,
            )

            shards.append(shard)

            # Log the actual layers being loaded
            actual_last_layer = end_layer - 1
            num_layers_in_shard = end_layer - current_layer
            lm_head_note = " [HAS LM HEAD]" if is_last_peer else ""
            logger.info(
                f"Assigned layers {current_layer}-{actual_last_layer} ({num_layers_in_shard} layers) to peer {peer.id[:8]} "
                f"({shard_memory:.1f}GB){lm_head_note}"
            )

            current_layer = end_layer

            if current_layer >= self.total_layers:
                break

        return shards

    def _distribute_proportionally(
        self,
        peer_layer_capacity: List[int],
        layer_memory_gb: float,
        kv_cache_per_peer: float,
        overhead_per_peer: float,
    ) -> List[ShardAssignment]:
        """
        Distribute layers proportionally across ALL available peers.
        Each peer gets layers proportional to its share of total RAM.
        """
        import math

        # Calculate total usable RAM across all peers
        total_usable_ram = sum(
            peer.ram_available_gb * (1 - self.safety_margin)
            - kv_cache_per_peer
            - overhead_per_peer
            for peer in self.peers
        )

        shards = []
        current_layer = 0

        for i, peer in enumerate(self.peers):
            # Calculate this peer's share of total RAM
            peer_usable_ram = (
                peer.ram_available_gb * (1 - self.safety_margin)
                - kv_cache_per_peer
                - overhead_per_peer
            )
            ram_proportion = peer_usable_ram / total_usable_ram

            # Calculate layers for this peer based on proportion
            if i == len(self.peers) - 1:
                # Last peer gets all remaining layers
                end_layer = self.total_layers
            else:
                # Assign layers proportionally
                layers_for_peer = math.floor(self.total_layers * ram_proportion)
                # Ensure at least 1 layer if there are layers remaining
                if layers_for_peer == 0 and current_layer < self.total_layers:
                    layers_for_peer = 1
                end_layer = min(current_layer + layers_for_peer, self.total_layers)

            # Skip if no layers to assign
            if current_layer >= end_layer:
                continue

            # Calculate memory for this shard
            num_layers = end_layer - current_layer
            shard_memory = (
                layer_memory_gb * num_layers + kv_cache_per_peer + overhead_per_peer
            )

            # Determine optimal gRPC address
            grpc_address = f"{peer.host}:{peer.grpc_port}"

            is_last_peer = end_layer == self.total_layers

            shard = ShardAssignment(
                peer_id=peer.id,
                peer_address=peer.address,
                grpc_address=grpc_address,
                start_layer=current_layer,
                end_layer=end_layer,
                estimated_memory_gb=shard_memory,
                has_embedding=(current_layer == 0),
                has_lm_head=is_last_peer,
            )

            shards.append(shard)

            # Log the assignment
            actual_last_layer = end_layer - 1
            num_layers_in_shard = end_layer - current_layer
            lm_head_note = " [HAS LM HEAD]" if is_last_peer else ""
            logger.info(
                f"Assigned layers {current_layer}-{actual_last_layer} ({num_layers_in_shard} layers) to peer {peer.id[:8]} "
                f"({shard_memory:.1f}GB, {ram_proportion*100:.1f}% of RAM){lm_head_note}"
            )

            current_layer = end_layer

            if current_layer >= self.total_layers:
                break

        return shards

    def optimize_network_paths(
        self, plan: ShardingPlan, coordinator_interfaces: List
    ) -> ShardingPlan:
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
            from shard.zeroconf.network import NetworkInterface

            peer_interfaces = [
                NetworkInterface.from_dict(iface_data)
                for iface_data in peer.network_interfaces.values()
            ]

            # Find best path
            best_path = NetworkDiscovery.select_best_network(
                coordinator_interfaces, peer_interfaces, peer.grpc_port
            )

            if best_path:
                local_ip, remote_ip = best_path
                shard.grpc_address = f"{remote_ip}:{peer.grpc_port}"
                logger.info(
                    f"Optimized path for {shard.peer_id[:8]}: {shard.grpc_address}"
                )

        return plan
