"""Generate text using distributed inference."""
import click
import time
from typing import List, Tuple
import grpc
import mlx.core as mx
from transformers import AutoTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper


@click.command()
@click.option(
    "--model",
    type=str,
    default="shard_0",
    help="Path or name of the model to use",
    show_default=True,
)
@click.option(
    "--prompt",
    type=str,
    default="how to write quicksort in python",
    help="Prompt for text generation",
    show_default=True,
)
@click.option(
    "--max-tokens",
    type=int,
    default=512,
    help="Maximum number of tokens to generate",
    show_default=True,
)
@click.option(
    "--server-address",
    type=str,
    default="localhost:50051",
    help="Comma-separated addresses of the gRPC servers",
    show_default=True,
)
@click.option(
    "--start-layer",
    type=int,
    default=None,
    help="Start layer for dynamic sharding",
)
@click.option(
    "--end-layer",
    type=int,
    default=None,
    help="End layer for dynamic sharding",
)
def generate(model, prompt, max_tokens, server_address, start_layer, end_layer):
    """Generate text using a distributed model."""
    from shard.grpc import mlx_tensor_pb2, mlx_tensor_pb2_grpc
    from shard.server.utils import load_model, response_to_mlx_array, send_tensor

    tokenizer = AutoTokenizer.from_pretrained(model)
    loaded_model = load_model(model, start_layer=start_layer, end_layer=end_layer)

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    channel_options = [
        ("grpc.max_metadata_size", 64 * 1024 * 1024),  # 64MB metadata
        ("grpc.max_send_message_length", -1),  # Unlimited send
        ("grpc.max_receive_message_length", -1),  # Unlimited receive
        ("grpc.http2.max_frame_size", 16 * 1024 * 1024),  # 16MB frames
    ]

    server_addresses = server_address.split(",")
    stubs = []
    for address in server_addresses:
        channel = grpc.insecure_channel(address.strip(), options=channel_options)
        stub = mlx_tensor_pb2_grpc.MLXTensorServiceStub(channel)
        stubs.append(stub)

    for stub in stubs:
        reset_response = stub.ResetCache(mlx_tensor_pb2.ResetCacheRequest())
        click.echo(f"ResetCache Response: {reset_response.message}")

    for t in stream_generate(
        loaded_model, tokenizer, prompt_text, max_tokens=max_tokens, stubs=stubs
    ):
        click.echo(t, nl=False)
    click.echo()


def generate_step(prompt, model, stubs: List):
    """Generate tokens step by step."""
    from shard.server.utils import response_to_mlx_array, send_tensor

    def sample(logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1)

    y = prompt
    if hasattr(model, "make_cache"):
        cache = model.make_cache()
    else:
        raise ValueError(
            "Model does not have make_cache() method. Please use a compatible model."
        )

    def _step(y):
        output = model(y[None], cache=cache)
        if output.dtype == mx.bfloat16:
            output = output.astype(mx.float16)

        for i, stub in enumerate(stubs):
            response = send_tensor(stub, output)
            output = response_to_mlx_array(response)
            if output is None:
                raise ValueError(f"Shard {i+1} returned None")
            if i == len(stubs) - 1:  # Last stub
                logits = output[:, -1, :]
                y = sample(logits)
                return y

        raise ValueError("No valid response from any stub")

    y = _step(y)
    mx.async_eval(y)
    while True:
        next_y = _step(y)
        mx.async_eval(next_y)
        yield y.item()
        y = next_y


def stream_generate(model, tokenizer, prompt: str, max_tokens: int = 100, stubs=None):
    """Stream generated tokens."""
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    tic = time.perf_counter()
    detokenizer.reset()
    
    token_count = 0
    prompt_time = 0
    
    for token, n in zip(
        generate_step(prompt_tokens, model, stubs),
        range(max_tokens),
    ):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)
        yield detokenizer.last_segment
        token_count = n + 1

    detokenizer.finalize()
    yield detokenizer.last_segment
    gen_time = time.perf_counter() - tic
    
    click.echo("=" * 10)
    if token_count == 0:
        click.echo("No tokens generated for this prompt")
        return
    prompt_tps = prompt_tokens.size / prompt_time if prompt_time > 0 else 0
    gen_tps = (token_count - 1) / gen_time if gen_time > 0 else 0
    click.echo(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
    click.echo(f"Generation: {gen_tps:.3f} tokens-per-sec")
