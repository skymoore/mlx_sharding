from pyarrow import flight
from typing import List, Tuple
from logging import getLogger

logger = getLogger(__name__)


class FlightConnectionPool:
    """
    Connection pool for creating isolated Arrow Flight clients per request.
    Enables true concurrency by preventing shared state between requests.
    """

    def __init__(self, peer_addresses: List[Tuple[str, int]]):
        """
        Initialize connection pool with peer addresses.

        Args:
            peer_addresses: List of (host, port) tuples for each peer
        """
        self.peer_addresses = peer_addresses
        logger.info(f"✓ Flight connection pool initialized with {len(peer_addresses)} peers")

    def create_clients_for_request(self) -> List[flight.FlightClient]:
        """
        Create fresh Flight clients for a single request.

        Returns:
            List of FlightClient instances - caller should close them after use
        """
        clients = []
        for host, port in self.peer_addresses:
            client = flight.FlightClient(f"grpc://{host}:{port}")
            clients.append(client)
        return clients