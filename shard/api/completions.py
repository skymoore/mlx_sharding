from shard.api.models import ChatCompletionRequest, CompletionRequest
import mlx.core as mx
from typing import Any, Dict
import time
import uuid
import json
import asyncio
from logging import getLogger
from shard.api.mlx_model_provider import MLXModelProvider

log = getLogger(__name__)


async def generate_chat_completion(
    request: ChatCompletionRequest, model_provider: MLXModelProvider, prompt: mx.array
) -> Dict[str, Any]:
    """Generate non-streaming chat completion."""
    tokens = []
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()

    finish_reason = "length"
    start_time = time.time()

    for (token, _), n in zip(
        model_provider.generate(
            prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
            max_tokens=request.max_tokens,
        ),
        range(request.max_tokens),
    ):
        tokens.append(token)
        detokenizer.add_token(token)

        if token == model_provider.tokenizer.eos_token_id:
            finish_reason = "stop"
            break

        if n % 256 == 0:
            mx.clear_cache()

    generation_time = time.time() - start_time
    tokens_per_second = len(tokens) / generation_time if generation_time > 0 else 0

    log.info("=" * 80)
    log.info(f"🏁 GENERATION COMPLETE")
    log.info(f"   Prompt tokens: {len(prompt)}")
    log.info(f"   Completion tokens: {len(tokens)}")
    log.info(f"   Total tokens: {len(prompt) + len(tokens)}")
    log.info(f"   Generation time: {generation_time:.2f}s")
    log.info(f"   Tokens/second: {tokens_per_second:.2f}")
    log.info("=" * 80)

    detokenizer.finalize()
    text = detokenizer.text

    # Build response
    response = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt) + len(tokens),
        },
    }

    return response


async def stream_chat_completion(
    request: ChatCompletionRequest, model_provider: MLXModelProvider, prompt: mx.array
):
    """Generate streaming chat completion."""
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()

    finish_reason = "length"
    token_count = 0
    start_time = time.time()

    try:
        for (token, _), n in zip(
            model_provider.generate(
                prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
                max_tokens=request.max_tokens,
            ),
            range(request.max_tokens),
        ):
            token_count += 1
            detokenizer.add_token(token)

            if token == model_provider.tokenizer.eos_token_id:
                finish_reason = "stop"
                break

            # Get the segment to send
            text = detokenizer.last_segment

            # Periodic cache clearing to prevent memory accumulation
            if n % 256 == 0:
                mx.clear_cache()

            if text:
                # Send text directly
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                await asyncio.sleep(0.001)  # Small delay to prevent socket overflow

        # Log generation statistics
        generation_time = time.time() - start_time
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0

        log.info("=" * 80)
        log.info("GENERATION COMPLETE")
        log.info(f"   Prompt tokens: {len(prompt)}")
        log.info(f"   Completion tokens: {token_count}")
        log.info(f"   Total tokens: {len(prompt) + token_count}")
        log.info(f"   Generation time: {generation_time:.2f}s")
        log.info(f"   Tokens/second: {tokens_per_second:.2f}")
        log.info("=" * 80)

        # Send final chunk
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except GeneratorExit:
        # Client disconnected
        log.info(f"Client disconnected during streaming (request {request_id})")
        raise

    except Exception as e:
        log.error(f"Error in streaming: {e}", exc_info=True)
        try:
            error_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            pass


async def generate_completion(
    request: CompletionRequest, model_provider: MLXModelProvider, prompt: mx.array
) -> Dict[str, Any]:
    """Generate non-streaming completion."""
    tokens = []
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()
    start_time = time.time()

    for (token, _), _ in zip(
        model_provider.generate(
            prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
        ),
        range(request.max_tokens),
    ):
        tokens.append(token)
        detokenizer.add_token(token)

        if token == model_provider.tokenizer.eos_token_id:
            break

    generation_time = time.time() - start_time
    tokens_per_second = len(tokens) / generation_time if generation_time > 0 else 0

    log.info("=" * 80)
    log.info("GENERATION COMPLETE")
    log.info(f"   Prompt tokens: {len(prompt)}")
    log.info(f"   Completion tokens: {len(tokens)}")
    log.info(f"   Total tokens: {len(prompt) + len(tokens)}")
    log.info(f"   Generation time: {generation_time:.2f}s")
    log.info(f"   Tokens/second: {tokens_per_second:.2f}")
    log.info("=" * 80)

    detokenizer.finalize()
    text = detokenizer.text

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt) + len(tokens),
        },
    }


async def stream_completion(
    request: CompletionRequest, model_provider: MLXModelProvider, prompt: mx.array
):
    """Generate streaming completion."""

    request_id = f"cmpl-{uuid.uuid4()}"
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()
    token_count = 0
    start_time = time.time()

    try:
        for (token, _), _ in zip(
            model_provider.generate(
                prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
            ),
            range(request.max_tokens),
        ):
            token_count += 1
            detokenizer.add_token(token)
            text = detokenizer.last_segment

            if text:
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "finish_reason": None,
                        }
                    ],
                }

                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.001)

            if token == model_provider.tokenizer.eos_token_id:
                break

        # Log generation statistics
        generation_time = time.time() - start_time
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0

        log.info("=" * 80)
        log.info(f"🏁 GENERATION COMPLETE")
        log.info(f"   Prompt tokens: {len(prompt)}")
        log.info(f"   Completion tokens: {token_count}")
        log.info(f"   Total tokens: {len(prompt) + token_count}")
        log.info(f"   Generation time: {generation_time:.2f}s")
        log.info(f"   Tokens/second: {tokens_per_second:.2f}")
        log.info("=" * 80)

        yield "data: [DONE]\n\n"

    except GeneratorExit:
        log.info(f"Client disconnected during streaming (request {request_id})")
        raise

    except Exception as e:
        log.error(f"Error in streaming: {e}", exc_info=True)
        try:
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            pass
