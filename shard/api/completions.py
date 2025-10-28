from shard.api.models import ChatCompletionRequest, CompletionRequest
import mlx.core as mx
from typing import Any, Dict, List, Union, Optional
import time
import uuid
import json
import asyncio
from logging import getLogger
from shard.api.mlx_model_provider import MLXModelProvider
from shard.api.tool_parsing import parse_glm4_tool_calls

log = getLogger(__name__)


def normalize_stop_sequences(stop: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize stop sequences to a list of strings."""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return stop


def check_stop_sequence(text: str, stop_sequences: List[str]) -> Optional[str]:
    """Check if text contains any stop sequence. Returns the matched sequence or None."""
    for stop_seq in stop_sequences:
        if stop_seq in text:
            return stop_seq
    return None


async def generate_chat_completion(
    request: ChatCompletionRequest, model_provider: MLXModelProvider, prompt: mx.array
) -> Dict[str, Any]:
    """Generate non-streaming chat completion."""
    tokens = []
    detokenizer = model_provider.tokenizer.detokenizer
    detokenizer.reset()

    finish_reason = "length"
    start_time = time.time()
    stop_sequences = normalize_stop_sequences(request.stop)

    # Get all EOS token IDs
    eos_token_ids = set()
    if hasattr(model_provider.tokenizer, "eos_token_ids"):
        eos_token_ids = set(model_provider.tokenizer.eos_token_ids)
    elif model_provider.tokenizer.eos_token_id is not None:
        eos_token_ids = {model_provider.tokenizer.eos_token_id}

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
        # Check if token is an EOS token before adding to detokenizer
        if token in eos_token_ids:
            finish_reason = "stop"
            break

        tokens.append(token)
        detokenizer.add_token(token)

        # Check for stop sequences
        if stop_sequences:
            current_text = detokenizer.text
            matched_stop = check_stop_sequence(current_text, stop_sequences)
            if matched_stop:
                finish_reason = "stop"
                log.info(f"Stop sequence detected: {repr(matched_stop)}")
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

    # Trim stop sequence from output if present
    if stop_sequences:
        for stop_seq in stop_sequences:
            if stop_seq in text:
                text = text.split(stop_seq)[0]
                break

    # Also trim EOS tokens that may have been decoded into text
    eos_token = model_provider.tokenizer.eos_token
    if eos_token and eos_token in text:
        text = text.split(eos_token)[0]

    # Trim any additional EOS tokens from the tokenizer
    if hasattr(model_provider.tokenizer, "eos_token_ids"):
        for eos_id in model_provider.tokenizer.eos_token_ids:
            eos_str = model_provider.tokenizer.decode([eos_id])
            if eos_str in text:
                text = text.split(eos_str)[0]

    # Parse tool calls if tools were provided
    message = {"role": "assistant", "content": text}
    
    if request.tools:
        # Only parse tool calls for GLM-4 models
        model_type = getattr(model_provider, "model_type", "unknown")
        if model_type in ("chatglm", "glm4_moe"):
            parsed = parse_glm4_tool_calls(text, request.tools)
            
            # Update message with parsed content
            if "reasoning_content" in parsed:
                message["reasoning_content"] = parsed["reasoning_content"]
            
            if "content" in parsed:
                message["content"] = parsed["content"]
            else:
                message["content"] = None
            
            if "tool_calls" in parsed:
                message["tool_calls"] = parsed["tool_calls"]
                # When tool calls are present, finish_reason should be "tool_calls"
                finish_reason = "tool_calls"
        else:
            # For other models, assume they return tool calls in a standard format
            # or implement additional parsers as needed
            log.warning(f"Tool calling not yet implemented for model_type: {model_type}")

    # Build response
    response = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
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
    stop_sequences = normalize_stop_sequences(request.stop)

    # Get all EOS token IDs
    eos_token_ids = set()
    if hasattr(model_provider.tokenizer, "eos_token_ids"):
        eos_token_ids = set(model_provider.tokenizer.eos_token_ids)
    elif model_provider.tokenizer.eos_token_id is not None:
        eos_token_ids = {model_provider.tokenizer.eos_token_id}

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
            # Check if token is an EOS token before adding to detokenizer
            if token in eos_token_ids:
                finish_reason = "stop"
                break

            token_count += 1
            detokenizer.add_token(token)

            # Check for stop sequences
            if stop_sequences:
                current_text = detokenizer.text
                matched_stop = check_stop_sequence(current_text, stop_sequences)
                if matched_stop:
                    finish_reason = "stop"
                    log.info(f"Stop sequence detected: {repr(matched_stop)}")
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

        # Parse tool calls if tools were provided
        if request.tools:
            # Only parse tool calls for GLM-4 models
            model_type = getattr(model_provider, "model_type", "unknown")
            if model_type in ("chatglm", "glm4_moe"):
                detokenizer.finalize()
                full_text = detokenizer.text
                parsed = parse_glm4_tool_calls(full_text, request.tools)
                
                # If tool calls were found, send them
                if "tool_calls" in parsed:
                    finish_reason = "tool_calls"
                    for tool_call in parsed["tool_calls"]:
                        tool_chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": [tool_call]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(tool_chunk)}\n\n"
            else:
                log.warning(f"Tool calling not yet implemented for model_type: {model_type}")

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
    finish_reason = "length"
    stop_sequences = normalize_stop_sequences(request.stop)

    # Get all EOS token IDs
    eos_token_ids = set()
    if hasattr(model_provider.tokenizer, "eos_token_ids"):
        eos_token_ids = set(model_provider.tokenizer.eos_token_ids)
    elif model_provider.tokenizer.eos_token_id is not None:
        eos_token_ids = {model_provider.tokenizer.eos_token_id}

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
        # Check if token is an EOS token before adding to detokenizer
        if token in eos_token_ids:
            finish_reason = "stop"
            break

        tokens.append(token)
        detokenizer.add_token(token)

        # Check for stop sequences
        if stop_sequences:
            current_text = detokenizer.text
            matched_stop = check_stop_sequence(current_text, stop_sequences)
            if matched_stop:
                finish_reason = "stop"
                log.info(f"Stop sequence detected: {repr(matched_stop)}")
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

    # Trim stop sequence from output if present
    if stop_sequences:
        for stop_seq in stop_sequences:
            if stop_seq in text:
                text = text.split(stop_seq)[0]
                break

    # Also trim EOS tokens that may have been decoded into text
    eos_token = model_provider.tokenizer.eos_token
    if eos_token and eos_token in text:
        text = text.split(eos_token)[0]

    # Trim any additional EOS tokens from the tokenizer
    if hasattr(model_provider.tokenizer, "eos_token_ids"):
        for eos_id in model_provider.tokenizer.eos_token_ids:
            eos_str = model_provider.tokenizer.decode([eos_id])
            if eos_str in text:
                text = text.split(eos_str)[0]

    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
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
    finish_reason = "length"
    stop_sequences = normalize_stop_sequences(request.stop)

    # Get all EOS token IDs
    eos_token_ids = set()
    if hasattr(model_provider.tokenizer, "eos_token_ids"):
        eos_token_ids = set(model_provider.tokenizer.eos_token_ids)
    elif model_provider.tokenizer.eos_token_id is not None:
        eos_token_ids = {model_provider.tokenizer.eos_token_id}

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
            # Check if token is an EOS token before adding to detokenizer
            if token in eos_token_ids:
                finish_reason = "stop"
                break

            token_count += 1
            detokenizer.add_token(token)

            # Check for stop sequences
            if stop_sequences:
                current_text = detokenizer.text
                matched_stop = check_stop_sequence(current_text, stop_sequences)
                if matched_stop:
                    finish_reason = "stop"
                    log.info(f"Stop sequence detected: {repr(matched_stop)}")
                    break

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

        # Send final chunk with finish_reason
        final_chunk = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": finish_reason,
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
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
