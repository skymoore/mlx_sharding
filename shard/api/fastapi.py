"""
FastAPI V2 - Zero-Configuration Distributed Inference API Server

This server:
1. Runs the orchestrator to discover peers and distribute model
2. Starts the FastAPI server after setup is complete
3. Routes inference requests through the distributed system
"""

import logging
import time
from typing import Optional, Dict, Any

import mlx.core as mx
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from shard.orchestrator.orchestrator import APIServerOrchestrator as Orchestrator
from shard.api.models import (
    ModelList,
    ModelInfo,
    ChatCompletionRequest,
    CompletionRequest,
)
from shard.api.mlx_model_provider import MLXModelProvider
from shard.api.completions import (
    stream_chat_completion,
    stream_completion,
    generate_chat_completion,
    generate_completion,
)
from shard.api.grpc import FlightConnectionPool

# Setup logging
logger = logging.getLogger(__name__)


# Application state (stored in app.state to avoid globals)
app = FastAPI(title="MLX Sharding V2 API", version="2.0.0")

# Initialize app state
app.state.model_provider = None
app.state.api_keys = set()
app.state.setup_complete = False
app.state.setup_info = {}
app.state.orchestrator_instance = None
app.state.discovered_peers = []

# Security
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> bool:
    """Verify API key from Authorization header."""
    # Skip authentication for health check and setup status
    if request.url.path in ["/health", "/v1/setup/status"]:
        return True

    # If no API keys configured, allow all requests
    if not request.app.state.api_keys:
        logging.warning("No API keys configured - authentication disabled")
        return True

    # Check for valid credentials
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify the API key
    if credentials.credentials not in request.app.state.api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    return {"status": "ok", "setup_complete": request.app.state.setup_complete, "version": "2.0.0"}


@app.get("/v1/setup/status")
async def get_setup_status(request: Request):
    """Get current setup status and distributed system info."""
    return {
        "setup_complete": request.app.state.setup_complete,
        "setup_info": request.app.state.setup_info,
        "timestamp": time.time(),
    }


@app.get("/v1/models")
async def list_models(
    request: Request, authenticated: bool = Security(verify_api_key)
) -> ModelList:
    """List available models (OpenAI format)."""
    if not request.app.state.setup_complete:
        raise HTTPException(status_code=503, detail="Setup not complete")

    model_provider = request.app.state.model_provider
    logger.info(f"📋 /v1/models called - returning model: {model_provider.model_name}")
    return ModelList(
        data=[
            ModelInfo(
                id=model_provider.model_name,
                created=model_provider.created,
            )
        ]
    )


@app.get("/api/models")
async def list_models_openwebui(
    request: Request, authenticated: bool = Security(verify_api_key)
) -> ModelList:
    """List available models (Open WebUI format)."""
    if not request.app.state.setup_complete:
        raise HTTPException(status_code=503, detail="Setup not complete")

    model_provider = request.app.state.model_provider
    return ModelList(
        data=[
            ModelInfo(
                id=model_provider.model_name,
                created=model_provider.created,
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    authenticated: bool = Security(verify_api_key),
):
    """Handle chat completion requests with tool calling support."""

    if not http_request.app.state.setup_complete:
        raise HTTPException(status_code=503, detail="Setup not complete")

    model_provider = http_request.app.state.model_provider

    try:
        # Prepare messages
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Add tool descriptions to system message if tools provided
        if request.tools:
            tool_prompt = model_provider.tool_manager.format_tools_for_prompt(
                request.tools
            )
            # Find or create system message
            system_msg_idx = next(
                (i for i, m in enumerate(messages) if m["role"] == "system"), None
            )
            if system_msg_idx is not None:
                messages[system_msg_idx]["content"] += tool_prompt
            else:
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant.{tool_prompt}",
                    },
                )

        # Apply chat template using tokenizer's built-in support
        # The tokenizer handles system message compatibility automatically
        prompt = model_provider.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        # 🔍 DEBUG: Log prompt details
        logger.info("=" * 80)
        logger.info("📝 PROMPT ENCODING")
        logger.info("=" * 80)
        logger.info(
            f"Prompt token IDs: {prompt if len(prompt) < 100 else f'{prompt[:20]}...'}"
        )
        logger.info(f"Prompt length: {len(prompt)} tokens")
        try:
            decoded = model_provider.tokenizer.decode(prompt)
            logger.info(f"Decoded prompt: {repr(decoded)}")
        except:
            pass

        # 🔍 DEBUG: Check if EOS tokens are in prompt
        if hasattr(model_provider.tokenizer, "eos_token_ids"):
            for eos_id in model_provider.tokenizer.eos_token_ids:
                if eos_id in prompt:
                    positions = [i for i, t in enumerate(prompt) if t == eos_id]
                    logger.warning(
                        f"⚠️  EOS token {eos_id} found in prompt at positions: {positions}"
                    )
        logger.info("=" * 80)

        prompt_array = mx.array(prompt)

        # Handle streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(request, model_provider, prompt_array),
                media_type="text/event-stream",
            )
        else:
            return await generate_chat_completion(request, model_provider, prompt_array)

    except Exception as e:
        logging.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    http_request: Request,
    authenticated: bool = Security(verify_api_key),
):
    """Handle text completion requests."""
    if not http_request.app.state.setup_complete:
        raise HTTPException(status_code=503, detail="Setup not complete")

    model_provider = http_request.app.state.model_provider

    try:
        prompt = model_provider.tokenizer.encode(request.prompt)
        prompt_array = mx.array(prompt)

        if request.stream:
            return StreamingResponse(
                stream_completion(request, model_provider, prompt_array),
                media_type="text/event-stream",
            )
        else:
            return await generate_completion(request, model_provider, prompt_array)

    except Exception as e:
        logging.error(f"Error in completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def run_orchestrator_setup(
    model_path: str,
    grpc_port: int,
    http_port: int,
    custom_chat_template: Optional[str] = None,
    resource_strategy: str = "fewest-nodes",
) -> Dict[str, Any]:
    """Run the orchestrator setup process (coordinator-only, no local layers)."""

    logger.info("=" * 80)
    logger.info("🚀 MLX SHARDING V2 - ZERO-CONFIG SETUP (COORDINATOR-ONLY)")
    logger.info("=" * 80)
    logger.info("Note: API server does NOT load model layers")
    logger.info("      Start peer processes separately with: mlx-shard-peer")
    logger.info("=" * 80)

    # Create orchestrator (coordinator-only mode - no local layers)
    orchestrator = Orchestrator(
        model_name=model_path,
        grpc_port=grpc_port,
        http_port=http_port,
        resource_strategy=resource_strategy,
    )

    # Store orchestrator in app state for cleanup
    app.state.orchestrator_instance = orchestrator

    # Run setup
    try:
        plan = await orchestrator.setup()

        # Store discovered peers for unclaiming on shutdown
        app.state.discovered_peers = list(orchestrator.discovery.peers.values())

        logger.info("=" * 80)
        logger.info("✓ SETUP COMPLETE")
        logger.info(f"  Total layers: {plan.total_layers}")
        logger.info(f"  Shards: {len(plan.shards)}")
        logger.info(f"  Memory required: {plan.total_memory_required_gb:.1f}GB")
        logger.info("=" * 80)

        # Store setup info
        app.state.setup_info = {
            "plan": plan.to_dict(),
            "model_path": model_path,
            "coordinator_only": True,
        }

        # Extract peer addresses for connection pool
        peer_addresses = []
        for shard in plan.shards:
            # Get peer info from orchestrator's discovered peers
            peer = next(
                (
                    p
                    for p in orchestrator.discovery.peers.values()
                    if p.id == shard.peer_id
                ),
                None,
            )
            if peer:
                peer_addresses.append((peer.host, peer.grpc_port))
                logger.info(
                    f"✓ Registered peer: {peer.host}:{peer.grpc_port} (layers {shard.start_layer}-{shard.end_layer})"
                )

        # Create connection pool for per-request stub creation
        connection_pool = FlightConnectionPool(peer_addresses)

        # Initialize model provider (coordinator loads NO layers, only coordinates)
        app.state.model_provider = MLXModelProvider(
            model_path=model_path,
            start_layer=None,  # No local layers
            end_layer=None,  # No local layers
            connection_pool=connection_pool,
        )

        app.state.setup_complete = True
        logger.info("✓ API server ready - all inference will be distributed to peers")
        return {"plan": plan.to_dict()}

    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        raise


async def shutdown_coordinator():
    """Gracefully shutdown the coordinator and unclaim peers."""

    logger.info("=" * 80)
    logger.info("🛑 SHUTTING DOWN COORDINATOR")
    logger.info("=" * 80)

    if app.state.orchestrator_instance and app.state.discovered_peers:
        try:
            await app.state.orchestrator_instance.unclaim_peers(app.state.discovered_peers)
            logger.info("✓ Peers unclaimed successfully")
        except Exception as e:
            logger.error(f"Error unclaiming peers: {e}", exc_info=True)

    if app.state.orchestrator_instance:
        try:
            app.state.orchestrator_instance.cleanup()
            logger.info("✓ Orchestrator cleanup complete")
        except Exception as e:
            logger.error(f"Error during orchestrator cleanup: {e}", exc_info=True)

    logger.info("=" * 80)
    logger.info("✓ SHUTDOWN COMPLETE")
    logger.info("=" * 80)


# Entry point moved to shard.cli.commands.api
