"""API server command."""
import click
import logging
import asyncio
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
@click.option(
    "--chat-template",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom Jinja chat template file",
)
@click.option(
    "--chat-template-string",
    type=str,
    default=None,
    help="Custom chat template as inline string",
)
def api(model, grpc_port, http_port, log_level, cache_limit_gb, chat_template, chat_template_string):
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

    # Load custom chat template if provided
    custom_template = None
    if chat_template:
        with open(chat_template) as f:
            custom_template = f.read()
        logging.info(f"✓ Loaded custom chat template from: {chat_template}")
    elif chat_template_string:
        custom_template = chat_template_string
        logging.info("✓ Using inline custom chat template")

    # Run orchestrator setup in background
    async def startup():
        try:
            await run_orchestrator_setup(
                model_path=model,
                grpc_port=grpc_port,
                http_port=http_port,
                custom_chat_template=custom_template,
            )
        except Exception as e:
            logging.error(f"Fatal setup error: {e}")
            sys.exit(1)

    # Register startup and shutdown events with FastAPI
    @app.on_event("startup")
    async def on_startup():
        await startup()

    @app.on_event("shutdown")
    async def on_shutdown():
        logging.info("\n🛑 Shutting down server...")
        try:
            await shutdown_coordinator()
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

    # Start FastAPI server
    logging.info(f"🚀 Starting API server on 0.0.0.0:{http_port}")
    logging.info(f"   OpenAI API: http://0.0.0.0:{http_port}/v1")
    logging.info(f"   Health: http://0.0.0.0:{http_port}/health")
    logging.info(f"   Setup Status: http://0.0.0.0:{http_port}/v1/setup/status")
    logging.info("Press Ctrl+C to stop")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=http_port,
        log_level=log_level.lower(),
        access_log=False,
    )
