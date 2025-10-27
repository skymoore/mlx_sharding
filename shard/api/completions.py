from shard.api.models import ChatCompletionRequest, CompletionRequest
import mlx.core as mx
from typing import Any, Dict
import time
import uuid
import json
import asyncio
from logging import getLogger
from shard.api.util import check_stop_sequences
from shard.api.mlx_model_provider import MLXModelProvider

log = getLogger(__name__)


async def generate_chat_completion(
    request: ChatCompletionRequest, model_provider: MLXModelProvider, prompt: mx.array
) -> Dict[str, Any]:
    """Generate non-streaming chat completion."""
    tokens = []
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()

    # Prepare stop sequences (user-provided strings only)
    stop_sequences = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_sequences = [request.stop]
        else:
            stop_sequences = request.stop

    # Get stop token IDs from tokenizer (this is what mlx_lm uses)
    stop_token_ids = model_provider.get_stop_token_ids()

    finish_reason = "length"

    for (token, _), n in zip(
        model_provider.generate(
            prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
            max_tokens=request.max_tokens,
        ),
        range(request.max_tokens),
    ):
        tokens.append(token)
        detokenizer.add_token(token)

        # Check for stop tokens by ID (faster and more reliable)
        # Convert MLX array to Python int for comparison
        token_id = int(token.item()) if hasattr(token, "item") else int(token)
        if token_id in stop_token_ids:
            log.debug(f"Stop token detected: {token_id} in {stop_token_ids}")
            finish_reason = "stop"
            break

        # Also check for stop sequences in generated text (for multi-token stops)
        current_text = detokenizer.text
        stop_found, trimmed_text = check_stop_sequences(current_text, stop_sequences)
        if stop_found:
            finish_reason = "stop"
            break

        if n % 256 == 0:
            mx.clear_cache()

    detokenizer.finalize()
    text = detokenizer.text

    # Final check and trim stop sequences
    _, text = check_stop_sequences(text, stop_sequences)

    # Parse tool calls if tools were provided
    cleaned_content = text
    tool_calls = []
    if request.tools:
        cleaned_content, tool_calls = model_provider.tool_manager.parse(text)

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
                    "content": cleaned_content,
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

    # Add tool_calls if any were found
    if tool_calls:
        response["choices"][0]["message"]["tool_calls"] = [
            {"id": tc.id, "type": tc.type, "function": tc.function} for tc in tool_calls
        ]
        # When tool calls present, finish_reason should be "tool_calls"
        response["choices"][0]["finish_reason"] = "tool_calls"

    return response


async def stream_chat_completion(
    request: ChatCompletionRequest, model_provider: MLXModelProvider, prompt: mx.array
):
    """Generate streaming chat completion with tool call support."""
    import json
    import asyncio

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()

    # Create streaming parser if tools provided
    streaming_parser = None
    if request.tools:
        streaming_parser = model_provider.tool_manager.create_streaming_parser()

    # Prepare stop sequences (user-provided strings only)
    stop_sequences = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_sequences = [request.stop]
        else:
            stop_sequences = request.stop

    # Get stop token IDs from tokenizer (this is what mlx_lm uses)
    stop_token_ids = model_provider.get_stop_token_ids()

    finish_reason = "length"

    try:
        for (token, _), n in zip(
            model_provider.generate(
                prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
                max_tokens=request.max_tokens,
            ),
            range(request.max_tokens),
        ):
            detokenizer.add_token(token)

            # Check for stop tokens by ID (faster and more reliable)
            # Convert MLX array to Python int for comparison
            token_id = int(token.item()) if hasattr(token, "item") else int(token)
            if token_id in stop_token_ids:
                finish_reason = "stop"
                break

            # Also check for stop sequences in full text (for multi-token stops)
            current_text = detokenizer.text
            stop_found, trimmed_text = check_stop_sequences(
                current_text, stop_sequences
            )
            if stop_found:
                finish_reason = "stop"
                break

            # Get the segment to send
            text = detokenizer.last_segment

            # 🔥 NEW: Periodic cache clearing to prevent memory accumulation
            if n % 256 == 0:
                mx.clear_cache()

            # Check if this segment contains a stop sequence
            if text:
                segment_stop_found, trimmed_segment = check_stop_sequences(
                    text, stop_sequences
                )
                if segment_stop_found:
                    finish_reason = "stop"
                    text = trimmed_segment

                # Parse for tool calls if enabled
                if streaming_parser and text:
                    content_to_emit, new_tool_calls = streaming_parser.add_chunk(text)

                    # Emit content chunk if any
                    if content_to_emit:
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content_to_emit},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # Emit tool call chunks if any
                    for tool_call in new_tool_calls:
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": tool_call.id,
                                                "type": "function",
                                                "function": {
                                                    "name": tool_call.function["name"],
                                                    "arguments": tool_call.function[
                                                        "arguments"
                                                    ],
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                elif text:
                    # No tool parsing, send text directly
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

                if segment_stop_found:
                    break

        # Finalize - get any remaining content
        if streaming_parser:
            remaining_content, remaining_tool_calls = streaming_parser.finalize()

            if remaining_content:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": remaining_content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            for tool_call in remaining_tool_calls:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.function["name"],
                                            "arguments": tool_call.function[
                                                "arguments"
                                            ],
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Update finish_reason if tool calls were emitted
            if streaming_parser.emitted_tool_calls:
                finish_reason = "tool_calls"

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

    for (token, _), _ in zip(
        model_provider.generate(
            prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
        ),
        range(request.max_tokens),
    ):
        tokens.append(token)
        detokenizer.add_token(token)

        if token == model_provider.tokenizer.eos_token_id:
            break

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

    try:
        for (token, _), _ in zip(
            model_provider.generate(
                prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
            ),
            range(request.max_tokens),
        ):
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
