"""
Network interface detection and optimal path selection.
Finds the fastest network path between peers for gRPC communication.
"""

import socket
import logging
import subprocess
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import ipaddress

logger = logging.getLogger(__name__)


@dataclass
class NetworkInterface:
    """Information about a network interface."""
    name: str
    ip_address: str
    netmask: str
    network: str  # CIDR notation
    speed_mbps: Optional[int]  # Link speed in Mbps
    is_up: bool
    is_loopback: bool
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "ip_address": self.ip_address,
            "netmask": self.netmask,
            "network": self.network,
            "speed_mbps": self.speed_mbps,
            "is_up": self.is_up,
            "is_loopback": self.is_loopback,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "NetworkInterface":
        return cls(**data)


class NetworkDiscovery:
    """Discover and analyze network interfaces."""
    
    @staticmethod
    def get_interfaces() -> List[NetworkInterface]:
        """
        Get all network interfaces with their properties.
        
        Returns:
            List of NetworkInterface objects
        """
        import psutil
        
        interfaces = []
        
        # Get network interface addresses
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        
        for iface_name, iface_addrs in addrs.items():
            # Get interface stats
            iface_stats = stats.get(iface_name)
            if not iface_stats:
                continue
            
            # Find IPv4 address
            ipv4_addr = None
            netmask = None
            for addr in iface_addrs:
                if addr.family == socket.AF_INET:
                    ipv4_addr = addr.address
                    netmask = addr.netmask
                    break
            
            if not ipv4_addr:
                continue
            
            # Calculate network CIDR
            try:
                network = ipaddress.IPv4Network(f"{ipv4_addr}/{netmask}", strict=False)
                network_cidr = str(network)
            except Exception as e:
                logger.debug(f"Failed to calculate network for {iface_name}: {e}")
                continue
            
            # Get link speed
            speed_mbps = iface_stats.speed if iface_stats.speed > 0 else None
            
            # Detect if loopback
            is_loopback = ipv4_addr.startswith("127.")
            
            interface = NetworkInterface(
                name=iface_name,
                ip_address=ipv4_addr,
                netmask=netmask,
                network=network_cidr,
                speed_mbps=speed_mbps,
                is_up=iface_stats.isup,
                is_loopback=is_loopback,
            )
            
            interfaces.append(interface)
            logger.debug(f"Found interface: {iface_name} @ {ipv4_addr} "
                        f"({speed_mbps}Mbps, {network_cidr})")
        
        return interfaces
    
    @staticmethod
    def find_common_networks(local_interfaces: List[NetworkInterface],
                            remote_interfaces: List[NetworkInterface]) -> List[Tuple[NetworkInterface, NetworkInterface]]:
        """
        Find interfaces on common networks between local and remote.
        
        Args:
            local_interfaces: Local network interfaces
            remote_interfaces: Remote peer's network interfaces
            
        Returns:
            List of (local_interface, remote_interface) tuples on same network
        """
        common = []
        
        for local_iface in local_interfaces:
            if local_iface.is_loopback or not local_iface.is_up:
                continue
            
            local_net = ipaddress.IPv4Network(local_iface.network)
            
            for remote_iface in remote_interfaces:
                if remote_iface.is_loopback or not remote_iface.is_up:
                    continue
                
                remote_net = ipaddress.IPv4Network(remote_iface.network)
                
                # Check if networks overlap
                if local_net.overlaps(remote_net):
                    common.append((local_iface, remote_iface))
                    logger.debug(f"Common network found: {local_iface.name} <-> {remote_iface.name} "
                               f"on {local_iface.network}")
        
        return common
    
    @staticmethod
    def test_connectivity(local_ip: str, remote_ip: str, port: int, timeout: float = 2.0) -> Tuple[bool, float]:
        """
        Test if we can connect to remote IP from local IP.
        
        Args:
            local_ip: Local IP to bind to
            remote_ip: Remote IP to connect to
            port: Port to test
            timeout: Connection timeout in seconds
            
        Returns:
            (success, latency_ms) tuple
        """
        import time
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            # Bind to specific local interface
            try:
                sock.bind((local_ip, 0))
            except OSError as e:
                logger.debug(f"Cannot bind to {local_ip}: {e}")
                sock.close()
                return False, 0.0
            
            # Try to connect
            start = time.time()
            try:
                sock.connect((remote_ip, port))
                latency_ms = (time.time() - start) * 1000
                sock.close()
                return True, latency_ms
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                logger.debug(f"Connection failed {local_ip} -> {remote_ip}:{port}: {e}")
                sock.close()
                return False, 0.0
        
        except Exception as e:
            logger.debug(f"Connectivity test error: {e}")
            return False, 0.0
    
    @staticmethod
    def select_best_network(local_interfaces: List[NetworkInterface],
                           remote_interfaces: List[NetworkInterface],
                           remote_port: int) -> Optional[Tuple[str, str]]:
        """
        Select the best network path between local and remote.
        
        Args:
            local_interfaces: Local network interfaces
            remote_interfaces: Remote peer's network interfaces
            remote_port: Port to test connectivity on
            
        Returns:
            (local_ip, remote_ip) tuple for best path, or None if no path found
        """
        # Find common networks
        common_networks = NetworkDiscovery.find_common_networks(
            local_interfaces, remote_interfaces
        )
        
        if not common_networks:
            logger.warning("No common networks found between peers")
            return None
        
        # Score each path
        paths = []
        for local_iface, remote_iface in common_networks:
            # Test connectivity
            reachable, latency_ms = NetworkDiscovery.test_connectivity(
                local_iface.ip_address,
                remote_iface.ip_address,
                remote_port
            )
            
            if not reachable:
                logger.debug(f"Path not reachable: {local_iface.ip_address} -> {remote_iface.ip_address}")
                continue
            
            # Calculate score based on speed and latency
            # Prefer: higher speed, lower latency
            speed = min(local_iface.speed_mbps or 100, remote_iface.speed_mbps or 100)
            score = speed * 1000 - latency_ms  # Speed in Mbps * 1000 - latency in ms
            
            paths.append({
                "local_ip": local_iface.ip_address,
                "remote_ip": remote_iface.ip_address,
                "local_iface": local_iface.name,
                "remote_iface": remote_iface.name,
                "speed_mbps": speed,
                "latency_ms": latency_ms,
                "score": score,
            })
            
            logger.info(f"Path found: {local_iface.name}({local_iface.ip_address}) -> "
                       f"{remote_iface.name}({remote_iface.ip_address}) "
                       f"[{speed}Mbps, {latency_ms:.1f}ms latency, score={score:.0f}]")
        
        if not paths:
            logger.warning("No reachable paths found")
            return None
        
        # Sort by score (highest first)
        paths.sort(key=lambda p: p["score"], reverse=True)
        best = paths[0]
        
        logger.info(f"✓ Best path selected: {best['local_iface']}({best['local_ip']}) -> "
                   f"{best['remote_iface']}({best['remote_ip']}) "
                   f"[{best['speed_mbps']}Mbps, {best['latency_ms']:.1f}ms latency]")
        
        return best["local_ip"], best["remote_ip"]


def get_network_interfaces_dict() -> Dict[str, Dict]:
    """
    Get network interfaces as a dictionary for serialization.
    
    Returns:
        Dictionary mapping interface names to their properties
    """
    interfaces = NetworkDiscovery.get_interfaces()
    return {iface.name: iface.to_dict() for iface in interfaces}
