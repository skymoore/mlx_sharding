"""API server command."""
import click
import logging
import asyncio
import signal
import sys
import mlx.core as mx
import uvicorn


@click.command()
@click.option(
    "--model",
    required=True,
    help="Path to MLX model or HuggingFace repo",
)
@click.option(
    "--grpc-port",
    type=int,
    default=50051,
    help="gRPC port for local peer",
    show_default=True,
)
@click.option(
    "--http-port",
    type=int,
    default=8080,
    help="HTTP API port",
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
    "--cache-limit-gb",
    type=int,
    default=None,
    help="MLX cache limit in GB",
)
def api(model, grpc_port, http_port, log_level, cache_limit_gb):
    """Start the MLX Sharding API server (coordinator-only mode)."""
    from shard.api.fastapi import (
        app,
        load_api_keys,
        run_orchestrator_setup,
        shutdown_coordinator,
        api_keys as global_api_keys,
    )

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Set cache limit
    if cache_limit_gb:
        mx.metal.set_cache_limit(cache_limit_gb * 1024 * 1024 * 1024)

    # Load API keys
    api_keys = load_api_keys()
    global_api_keys.clear()
    global_api_keys.update(api_keys)
    
    if api_keys:
        logging.info(f"✓ API key authentication enabled ({len(api_keys)} key(s) loaded)")
    else:
        logging.warning("⚠ No API keys configured - authentication disabled!")

    # Run orchestrator setup in background
    async def startup():
        try:
            await run_orchestrator_setup(
                model_path=model,
                grpc_port=grpc_port,
                http_port=http_port,
            )
        except Exception as e:
            logging.error(f"Fatal setup error: {e}")
            sys.exit(1)

    # Run setup before starting server
    asyncio.run(startup())

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logging.info("\n🛑 Received shutdown signal...")
        # Run async shutdown
        try:
            asyncio.run(shutdown_coordinator())
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            logging.info("Exiting...")
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start FastAPI server
    logging.info(f"🚀 Starting API server on 0.0.0.0:{http_port}")
    logging.info(f"   OpenAI API: http://0.0.0.0:{http_port}/v1")
    logging.info(f"   Health: http://0.0.0.0:{http_port}/health")
    logging.info(f"   Setup Status: http://0.0.0.0:{http_port}/v1/setup/status")
    logging.info("Press Ctrl+C to stop")

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=http_port,
            log_level=log_level.lower(),
            access_log=False,
        )
    except KeyboardInterrupt:
        logging.info("\n🛑 Keyboard interrupt received...")
        try:
            asyncio.run(shutdown_coordinator())
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            logging.info("Server stopped")
            sys.exit(0)
