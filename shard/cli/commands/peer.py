"""Peer server command."""
import click
import logging
import signal
import sys


@click.command()
@click.option(
    "--grpc-port",
    type=int,
    default=50052,
    help="gRPC server port",
    show_default=True,
)
@click.option(
    "--http-port",
    type=int,
    default=8081,
    help="HTTP control server port",
    show_default=True,
)
@click.option(
    "--cache-dir",
    type=str,
    default="~/.cache/mlx-sharding",
    help="Cache directory for model files",
    show_default=True,
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level",
    show_default=True,
)
@click.option(
    "--bind-ip",
    type=str,
    default=None,
    help="Specific IP address to bind to (auto-detect if not specified)",
)
def peer(grpc_port, http_port, cache_dir, log_level, bind_ip):
    """Start an MLX Shard Peer (zero-configuration worker node)."""
    from shard.server.peer import PeerServer

    logger = logging.getLogger(__name__)

    # Setup logging
    logging.getLogger().setLevel(getattr(logging, log_level))

    # Create and start server
    server = PeerServer(
        grpc_port=grpc_port,
        http_port=http_port,
        cache_dir=cache_dir,
        bind_ip=bind_ip,
    )

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("\nReceived shutdown signal")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        server.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
