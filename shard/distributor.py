"""
Model file distribution system.
Handles transferring model files to peers with chunking, caching, and verification.
"""

import asyncio
import aiohttp
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

from .discovery import PeerInfo
from .planner import ShardAssignment

logger = logging.getLogger(__name__)


class FileTransferError(Exception):
    """Raised when file transfer fails."""
    pass


@dataclass
class FileTransferProgress:
    """Progress information for a file transfer."""
    peer_id: str
    file_name: str
    bytes_sent: int
    total_bytes: int
    status: str  # "pending", "transferring", "complete", "cached", "error"
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_bytes == 0:
            return 100.0
        return (self.bytes_sent / self.total_bytes) * 100.0


class ModelFileDistributor:
    """
    Handles distributing model files to peers.
    
    Features:
    - Chunked file transfer (1MB chunks)
    - Hash-based cache verification
    - Parallel transfers to multiple peers
    - Progress callbacks
    - Resume capability for interrupted transfers
    """
    
    def __init__(self, model_path: str, cache_dir: str = "~/.cache/mlx-sharding"):
        """
        Initialize file distributor.
        
        Args:
            model_path: Path to model directory
            cache_dir: Cache directory on peers
        """
        self.model_path = Path(model_path)
        self.cache_dir = cache_dir
        self.chunk_size = 1024 * 1024  # 1MB chunks
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        logger.info(f"Initialized ModelFileDistributor for {model_path}")
    
    async def distribute_to_peers(self, 
                                   peers: List[PeerInfo],
                                   shards: List[ShardAssignment],
                                   progress_callback: Optional[Callable] = None) -> Dict[str, bool]:
        """
        Distribute model files to all peers in parallel.
        
        Args:
            peers: List of peers to distribute to
            shards: Shard assignments (to get peer IDs)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping peer_id to success status
        """
        logger.info(f"Starting file distribution to {len(peers)} peer(s)")
        
        # Create tasks for each peer
        tasks = []
        for shard in shards:
            peer = next((p for p in peers if p.id == shard.peer_id), None)
            if not peer:
                logger.warning(f"Peer {shard.peer_id} not found, skipping")
                continue
            
            task = self.distribute_to_peer(peer, shard, progress_callback)
            tasks.append((peer.id, task))
        
        # Execute all transfers in parallel
        results = {}
        for peer_id, task in tasks:
            try:
                success = await task
                results[peer_id] = success
            except Exception as e:
                logger.error(f"Failed to distribute to peer {peer_id}: {e}")
                results[peer_id] = False
        
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Distribution complete: {successful}/{len(results)} peers successful")
        
        return results
    
    async def distribute_to_peer(self,
                                  peer: PeerInfo,
                                  shard: ShardAssignment,
                                  progress_callback: Optional[Callable] = None) -> bool:
        """
        Distribute necessary model files to a single peer.
        
        Args:
            peer: Target peer
            shard: Shard assignment for this peer
            progress_callback: Optional callback(peer_id, file_name, status, progress)
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Distributing files to peer {peer.id[:8]} @ {peer.host}:{peer.http_port}")
        
        try:
            # Check if peer is on localhost - if so, skip file transfer
            if self._is_localhost(peer.host):
                logger.info(f"  ✓ Peer is on localhost - using local model files (no transfer needed)")
                # Still need to tell the peer about the local path
                await self._set_local_model_path(peer, self.model_path)
                return True
            
            # Get list of required files
            required_files = self._get_required_files()
            logger.info(f"  Required files: {len(required_files)}")
            
            # Check what peer already has cached
            cached_files = await self._check_peer_cache(peer, shard.peer_address.split(':')[0])
            logger.info(f"  Cached files: {len(cached_files)}")
            
            # Transfer missing files
            for file_name in required_files:
                file_path = self.model_path / file_name
                
                # Check if already cached
                if file_name in cached_files:
                    local_hash = self._compute_file_hash(file_path)
                    if cached_files[file_name] == local_hash:
                        logger.info(f"  ✓ {file_name} (cached)")
                        if progress_callback:
                            progress_callback(peer.id, file_name, "cached", 100.0)
                        continue
                
                # Transfer file
                logger.info(f"  → {file_name} ({file_path.stat().st_size / (1024**2):.1f}MB)")
                success = await self._transfer_file(
                    peer, 
                    file_path, 
                    file_name,
                    shard.peer_address.split(':')[0],
                    progress_callback
                )
                
                if not success:
                    logger.error(f"  ✗ Failed to transfer {file_name}")
                    return False
                
                logger.info(f"  ✓ {file_name} (transferred)")
            
            logger.info(f"✓ All files distributed to peer {peer.id[:8]}")
            return True
        
        except Exception as e:
            logger.error(f"Error distributing to peer {peer.id[:8]}: {e}", exc_info=True)
            return False
    
    def _get_required_files(self) -> List[str]:
        """
        Get list of required model files.
        
        Returns:
            List of file names to transfer
        """
        required = []
        
        # Config and tokenizer files
        for pattern in ["*.json", "*.model", "tokenizer.json", "*.txt"]:
            for file_path in self.model_path.glob(pattern):
                if file_path.is_file():
                    required.append(file_path.name)
        
        # Model weight files
        for pattern in ["*.safetensors", "*.bin"]:
            for file_path in self.model_path.glob(pattern):
                if file_path.is_file():
                    required.append(file_path.name)
        
        # Remove duplicates and sort
        required = sorted(set(required))
        
        return required
    
    async def _check_peer_cache(self, peer: PeerInfo, host: str) -> Dict[str, str]:
        """
        Check which files peer already has cached.
        
        Args:
            peer: Target peer
            host: Host address to connect to
            
        Returns:
            Dictionary mapping file_name to hash
        """
        try:
            url = f"http://{host}:{peer.http_port}/api/cache/check"
            params = {"model": self.model_path.name}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.warning(f"Cache check failed: HTTP {resp.status}")
                        return {}
        
        except Exception as e:
            logger.warning(f"Failed to check peer cache: {e}")
            return {}
    
    async def _transfer_file(self,
                             peer: PeerInfo,
                             file_path: Path,
                             file_name: str,
                             host: str,
                             progress_callback: Optional[Callable] = None) -> bool:
        """
        Transfer a file to peer in chunks.
        
        Args:
            peer: Target peer
            file_path: Local file path
            file_name: Name of file
            host: Host address to connect to
            progress_callback: Optional progress callback
            
        Returns:
            True if successful
        """
        file_size = file_path.stat().st_size
        total_chunks = (file_size + self.chunk_size - 1) // self.chunk_size
        
        try:
            async with aiohttp.ClientSession() as session:
                with open(file_path, 'rb') as f:
                    for chunk_num in range(total_chunks):
                        # Read chunk
                        chunk_data = f.read(self.chunk_size)
                        if not chunk_data:
                            break
                        
                        # Prepare form data
                        data = aiohttp.FormData()
                        data.add_field('model_name', self.model_path.name)
                        data.add_field('file_name', file_name)
                        data.add_field('chunk_num', str(chunk_num))
                        data.add_field('total_chunks', str(total_chunks))
                        data.add_field('chunk_data', chunk_data, 
                                     filename=f'{file_name}.chunk{chunk_num}',
                                     content_type='application/octet-stream')
                        
                        # Send chunk
                        url = f"http://{host}:{peer.http_port}/api/receive_file_chunk"
                        async with session.post(url, data=data, 
                                              timeout=aiohttp.ClientTimeout(total=60)) as resp:
                            if resp.status != 200:
                                result = await resp.json()
                                logger.error(f"Chunk {chunk_num} failed: {result}")
                                return False
                        
                        # Update progress
                        bytes_sent = min((chunk_num + 1) * self.chunk_size, file_size)
                        progress = (bytes_sent / file_size) * 100.0
                        
                        if progress_callback:
                            progress_callback(peer.id, file_name, "transferring", progress)
            
            # Mark as complete
            if progress_callback:
                progress_callback(peer.id, file_name, "complete", 100.0)
            
            return True
        
        except Exception as e:
            logger.error(f"File transfer failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(peer.id, file_name, "error", 0.0)
            return False
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of hash
        """
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _is_localhost(self, host: str) -> bool:
        """
        Check if a host is localhost by comparing against all local network interfaces.
        Uses psutil for reliable cross-platform interface enumeration.
        
        Args:
            host: Host address to check
            
        Returns:
            True if host is localhost
        """
        import socket
        import psutil
        
        localhost_names = ['localhost', '127.0.0.1', '::1', '0.0.0.0', '::']
        
        # Check direct localhost names
        if host in localhost_names:
            return True
        
        try:
            # Get all local IP addresses from all network interfaces using psutil
            local_ips = set(localhost_names)
            
            # Use psutil to get all network interface addresses
            net_if_addrs = psutil.net_if_addrs()
            for interface_name, addr_list in net_if_addrs.items():
                for addr in addr_list:
                    # addr.family can be AF_INET (IPv4) or AF_INET6 (IPv6)
                    if addr.family == socket.AF_INET or addr.family == socket.AF_INET6:
                        local_ips.add(addr.address)
            
            # Also try hostname resolution as fallback
            try:
                local_hostname = socket.gethostname()
                local_fqdn = socket.getfqdn()
                
                # Add hostname IPs
                try:
                    for ip in socket.gethostbyname_ex(local_hostname)[2]:
                        local_ips.add(ip)
                except Exception:
                    pass
                
                # Add FQDN IPs
                try:
                    for ip in socket.gethostbyname_ex(local_fqdn)[2]:
                        local_ips.add(ip)
                except Exception:
                    pass
            except Exception:
                pass
            
            # Check if peer host matches any local IP
            if host in local_ips:
                return True
            
            # Try to resolve peer hostname to IP and check intersection
            try:
                peer_ips = set()
                for info in socket.getaddrinfo(host, None):
                    ip_addr = info[4][0]
                    if isinstance(ip_addr, str):
                        peer_ips.add(ip_addr)
                if peer_ips & local_ips:  # Intersection
                    return True
            except socket.gaierror:
                pass
        
        except Exception as e:
            logger.warning(f"Error checking localhost: {e}")
        
        return False
    
    async def _set_local_model_path(self, peer: PeerInfo, model_path: Path):
        """
        Tell a localhost peer to use the local model path instead of cache.
        
        Args:
            peer: Target peer (must be localhost)
            model_path: Local model path to use
        """
        try:
            url = f"http://{peer.host}:{peer.http_port}/api/set_local_model_path"
            data = {"model_path": str(model_path)}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"  ✓ Peer configured to use local path: {model_path}")
                        return True
                    else:
                        logger.warning(f"Failed to set local model path: HTTP {resp.status}")
                        return False
        
        except Exception as e:
            logger.warning(f"Failed to set local model path: {e}")
            # Not critical - peer can still use cache
            return False
