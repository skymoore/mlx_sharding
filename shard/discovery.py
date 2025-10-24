"""
Peer discovery system using mDNS/Zeroconf and UDP broadcast.
Enables automatic discovery of MLX sharding peers on the local network.
"""

import socket
import threading
import time
import logging
import uuid
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser, ServiceListener
import json

logger = logging.getLogger(__name__)

# Service configuration
SERVICE_TYPE = "_mlx-shard._tcp.local."
UDP_PORT = 50052
BROADCAST_INTERVAL = 5.0  # seconds
DISCOVERY_TIMEOUT = 10.0  # seconds


@dataclass
class PeerInfo:
    """Information about a discovered peer."""
    id: str
    address: str  # host:port format
    host: str
    grpc_port: int
    http_port: int
    role: str  # "peer" or "coordinator"
    ram_total_gb: float
    ram_available_gb: float
    cpu_cores: int
    platform: str
    status: str
    model_loaded: str
    layers_loaded: str
    version: str
    last_seen: float
    network_interfaces: Optional[Dict[str, Dict]] = None  # Network interface info
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PeerInfo":
        """Create from dictionary."""
        return cls(**data)


class PeerDiscovery:
    """
    Handles peer discovery via mDNS/Zeroconf and UDP broadcast.
    
    Uses a hybrid approach:
    - mDNS/Zeroconf for standard service discovery
    - UDP broadcast as fallback for networks where mDNS is blocked
    """
    
    def __init__(self, role: str = "peer"):
        """
        Initialize discovery service.
        
        Args:
            role: Role of this node ("peer" or "coordinator")
        """
        self.role = role
        self.peer_id = str(uuid.uuid4())
        self.peers: Dict[str, PeerInfo] = {}
        self.peers_lock = threading.Lock()
        
        # mDNS/Zeroconf
        self.zeroconf: Optional[Zeroconf] = None
        self.service_info: Optional[ServiceInfo] = None
        self.browser: Optional[ServiceBrowser] = None
        
        # UDP broadcast
        self.udp_socket: Optional[socket.socket] = None
        self.udp_thread: Optional[threading.Thread] = None
        self.udp_running = False
        
        # Callbacks
        self.on_peer_discovered: Optional[Callable[[PeerInfo], None]] = None
        self.on_peer_lost: Optional[Callable[[str], None]] = None
        
        logger.info(f"Initialized discovery service (role={role}, id={self.peer_id[:8]})")
    
    def announce(self, grpc_port: int, http_port: int, capabilities: Dict):
        """
        Announce this peer on the network.
        
        Args:
            grpc_port: gRPC server port
            http_port: HTTP control server port
            capabilities: System capabilities dictionary
        """
        logger.info(f"Announcing peer on network (gRPC={grpc_port}, HTTP={http_port})")
        
        # Start mDNS announcement
        self._start_mdns_announcement(grpc_port, http_port, capabilities)
        
        # Start UDP broadcast
        self._start_udp_broadcast(grpc_port, http_port, capabilities)
        
        logger.info("✓ Peer announced via mDNS and UDP")
    
    def discover_peers(self, timeout: float = DISCOVERY_TIMEOUT) -> List[PeerInfo]:
        """
        Discover available peers on the network.
        
        Args:
            timeout: Discovery timeout in seconds
            
        Returns:
            List of discovered peers
        """
        logger.info(f"Discovering peers (timeout={timeout}s)...")
        
        # Start mDNS browser
        self._start_mdns_browser()
        
        # Start UDP listener
        self._start_udp_listener()
        
        # Wait for discovery
        time.sleep(timeout)
        
        # Get discovered peers
        with self.peers_lock:
            peers = list(self.peers.values())
        
        logger.info(f"✓ Discovered {len(peers)} peer(s)")
        for peer in peers:
            logger.info(f"  - {peer.id[:8]} @ {peer.address} ({peer.ram_available_gb:.1f}GB RAM)")
        
        return peers
    
    def update_status(self, status: str, **kwargs):
        """
        Update this peer's advertised status.
        
        Args:
            status: New status
            **kwargs: Additional properties to update
        """
        if self.service_info:
            # Update mDNS properties
            properties = self.service_info.properties
            properties[b'status'] = status.encode('utf-8')
            
            for key, value in kwargs.items():
                properties[key.encode('utf-8')] = str(value).encode('utf-8')
            
            # Re-register service with updated properties
            if self.zeroconf:
                self.zeroconf.update_service(self.service_info)
        
        logger.debug(f"Updated status: {status}")
    
    def stop(self):
        """Stop discovery service and cleanup."""
        logger.info("Stopping discovery service...")
        
        # Stop UDP
        self.udp_running = False
        if self.udp_socket:
            self.udp_socket.close()
        if self.udp_thread:
            self.udp_thread.join(timeout=2.0)
        
        # Stop mDNS
        if self.browser:
            self.browser.cancel()
        if self.zeroconf:
            if self.service_info:
                self.zeroconf.unregister_service(self.service_info)
            self.zeroconf.close()
        
        logger.info("✓ Discovery service stopped")
    
    # mDNS/Zeroconf implementation
    
    def _start_mdns_announcement(self, grpc_port: int, http_port: int, capabilities: Dict):
        """Start mDNS service announcement."""
        try:
            self.zeroconf = Zeroconf()
            
            # Get local IP
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            # Create service info
            properties = {
                'version': '2.0.0',
                'role': self.role,
                'peer_id': self.peer_id,
                'http_port': str(http_port),
                'ram_total_gb': str(capabilities.get('ram_total_gb', 0)),
                'ram_available_gb': str(capabilities.get('ram_available_gb', 0)),
                'cpu_cores': str(capabilities.get('cpu_cores', 0)),
                'platform': capabilities.get('platform', 'unknown'),
                'status': 'idle',
                'model_loaded': '',
                'layers_loaded': '',
            }
            
            service_name = f"mlx-shard-{hostname}-{self.peer_id[:8]}.{SERVICE_TYPE}"
            
            self.service_info = ServiceInfo(
                SERVICE_TYPE,
                service_name,
                addresses=[socket.inet_aton(local_ip)],
                port=grpc_port,
                properties=properties,
            )
            
            self.zeroconf.register_service(self.service_info)
            logger.info(f"✓ mDNS service registered: {service_name}")
            
        except Exception as e:
            logger.warning(f"Failed to start mDNS announcement: {e}")
    
    def _start_mdns_browser(self):
        """Start mDNS service browser."""
        try:
            if not self.zeroconf:
                self.zeroconf = Zeroconf()
            
            listener = _MDNSListener(self)
            self.browser = ServiceBrowser(self.zeroconf, SERVICE_TYPE, listener)
            logger.info("✓ mDNS browser started")
            
        except Exception as e:
            logger.warning(f"Failed to start mDNS browser: {e}")
    
    # UDP broadcast implementation
    
    def _start_udp_broadcast(self, grpc_port: int, http_port: int, capabilities: Dict):
        """Start UDP broadcast announcements."""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.udp_running = True
            
            # Broadcast message
            message = {
                'type': 'announcement',
                'peer_id': self.peer_id,
                'role': self.role,
                'grpc_port': grpc_port,
                'http_port': http_port,
                'version': '2.0.0',
                'status': 'idle',
                'model_loaded': '',
                'layers_loaded': '',
                **capabilities
            }
            
            def broadcast_loop():
                while self.udp_running:
                    try:
                        data = json.dumps(message).encode('utf-8')
                        self.udp_socket.sendto(data, ('<broadcast>', UDP_PORT))
                        time.sleep(BROADCAST_INTERVAL)
                    except OSError as e:
                        # Network unreachable or no route to host - this is expected
                        # when broadcasting across subnets. Just rely on mDNS instead.
                        if self.udp_running:
                            logger.debug(f"UDP broadcast not available: {e}")
                            # Stop trying to broadcast if network is unreachable
                            self.udp_running = False
                            break
                    except Exception as e:
                        if self.udp_running:
                            logger.warning(f"UDP broadcast error: {e}")
                            time.sleep(1)  # Back off on errors
            
            self.udp_thread = threading.Thread(target=broadcast_loop, daemon=True)
            self.udp_thread.start()
            logger.info(f"✓ UDP broadcast started on port {UDP_PORT}")
            
        except Exception as e:
            logger.warning(f"Failed to start UDP broadcast: {e}")
    
    def _start_udp_listener(self):
        """Start UDP broadcast listener."""
        try:
            listener_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener_socket.bind(('', UDP_PORT))
            listener_socket.settimeout(1.0)
            
            def listen_loop():
                while self.udp_running:
                    try:
                        data, addr = listener_socket.recvfrom(4096)
                        message = json.loads(data.decode('utf-8'))
                        
                        # Ignore own messages
                        if message.get('peer_id') == self.peer_id:
                            continue
                        
                        # Process announcement
                        if message.get('type') == 'announcement':
                            self._process_udp_announcement(message, addr[0])
                    
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.udp_running:
                            logger.debug(f"UDP listener error: {e}")
                
                listener_socket.close()
            
            thread = threading.Thread(target=listen_loop, daemon=True)
            thread.start()
            logger.info(f"✓ UDP listener started on port {UDP_PORT}")
            
        except Exception as e:
            logger.warning(f"Failed to start UDP listener: {e}")
    
    def _process_udp_announcement(self, message: Dict, host: str):
        """Process a UDP announcement message."""
        try:
            peer_id = message.get('peer_id')
            if not peer_id:
                return
            
            peer_info = PeerInfo(
                id=peer_id,
                address=f"{host}:{message.get('grpc_port')}",
                host=host,
                grpc_port=message.get('grpc_port'),
                http_port=message.get('http_port'),
                role=message.get('role', 'peer'),
                ram_total_gb=float(message.get('ram_total_gb', 0)),
                ram_available_gb=float(message.get('ram_available_gb', 0)),
                cpu_cores=int(message.get('cpu_cores', 0)),
                platform=message.get('platform', 'unknown'),
                status=message.get('status', 'idle'),
                model_loaded=message.get('model_loaded', ''),
                layers_loaded=message.get('layers_loaded', ''),
                version=message.get('version', '1.0.0'),
                last_seen=time.time()
            )
            
            with self.peers_lock:
                is_new = peer_id not in self.peers
                self.peers[peer_id] = peer_info
            
            if is_new:
                logger.debug(f"Discovered peer via UDP: {peer_id[:8]} @ {peer_info.address}")
                if self.on_peer_discovered:
                    self.on_peer_discovered(peer_info)
        
        except Exception as e:
            logger.debug(f"Error processing UDP announcement: {e}")
    
    def _add_peer(self, peer_info: PeerInfo):
        """Add or update a peer."""
        with self.peers_lock:
            is_new = peer_info.id not in self.peers
            self.peers[peer_info.id] = peer_info
        
        if is_new and self.on_peer_discovered:
            self.on_peer_discovered(peer_info)
    
    def _remove_peer(self, peer_id: str):
        """Remove a peer."""
        with self.peers_lock:
            if peer_id in self.peers:
                del self.peers[peer_id]
        
        if self.on_peer_lost:
            self.on_peer_lost(peer_id)


class _MDNSListener(ServiceListener):
    """Listener for mDNS service events."""
    
    def __init__(self, discovery: PeerDiscovery):
        self.discovery = discovery
    
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is discovered."""
        info = zc.get_service_info(type_, name)
        if info:
            self._process_service_info(info)
    
    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is removed."""
        # Extract peer_id from name if possible
        logger.debug(f"Service removed: {name}")
    
    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is updated."""
        info = zc.get_service_info(type_, name)
        if info:
            self._process_service_info(info)
    
    def _process_service_info(self, info: ServiceInfo):
        """Process discovered service info."""
        try:
            # Extract properties
            props = {}
            for key, value in info.properties.items():
                props[key.decode('utf-8')] = value.decode('utf-8')
            
            peer_id = props.get('peer_id')
            if not peer_id or peer_id == self.discovery.peer_id:
                return
            
            # Get address
            if info.addresses:
                host = socket.inet_ntoa(info.addresses[0])
            else:
                return
            
            peer_info = PeerInfo(
                id=peer_id,
                address=f"{host}:{info.port}",
                host=host,
                grpc_port=info.port,
                http_port=int(props.get('http_port', 8081)),
                role=props.get('role', 'peer'),
                ram_total_gb=float(props.get('ram_total_gb', 0)),
                ram_available_gb=float(props.get('ram_available_gb', 0)),
                cpu_cores=int(props.get('cpu_cores', 0)),
                platform=props.get('platform', 'unknown'),
                status=props.get('status', 'idle'),
                model_loaded=props.get('model_loaded', ''),
                layers_loaded=props.get('layers_loaded', ''),
                version=props.get('version', '1.0.0'),
                last_seen=time.time()
            )
            
            logger.debug(f"Discovered peer via mDNS: {peer_id[:8]} @ {peer_info.address}")
            self.discovery._add_peer(peer_info)
        
        except Exception as e:
            logger.debug(f"Error processing mDNS service: {e}")
