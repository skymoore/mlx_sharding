import grpc
from shard.grpc import mlx_tensor_pb2_grpc
from typing import Tuple, List
from logging import getLogger

logger = getLogger(__name__)


class GRPCConnectionPool:
    """
    Connection pool for creating isolated gRPC stubs per request.
    Enables true concurrency by preventing shared state between requests.
    """

    def __init__(self, peer_addresses: List[Tuple[str, int]]):
        """
        Initialize connection pool with peer addresses.

        Args:
            peer_addresses: List of (host, port) tuples for each peer
        """
        self.peer_addresses = peer_addresses
        self.channel_options = [
            ("grpc.max_metadata_size", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
            ("grpc.http2.max_frame_size", 4 * 1024 * 1024),
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 300000),
        ]
        logger.info(f"✓ Connection pool initialized with {len(peer_addresses)} peers")

    def create_stubs_for_request(self) -> Tuple[List, List]:
        """
        Create fresh gRPC stubs for a single request.
        Each request gets isolated channels to prevent concurrent interference.

        Returns:
            Tuple of (stubs, channels) - caller must close channels after use
        """
        stubs = []
        channels = []
        for host, port in self.peer_addresses:
            channel = grpc.insecure_channel(
                f"{host}:{port}", options=self.channel_options
            )
            stub = mlx_tensor_pb2_grpc.MLXTensorServiceStub(channel)
            stubs.append(stub)
            channels.append(channel)
        return stubs, channels
