"""Chat with the model via the API server."""

import click
import requests
import json
import sys
from typing import Optional


@click.command()
@click.option(
    "--api-url",
    type=str,
    default="http://localhost",
    help="Base URL of the API server",
    show_default=True,
)
@click.option(
    "--prompt",
    "-p",
    type=str,
    help="Prompt for text generation (use '-' to read from stdin)",
)
@click.option(
    "--max-tokens",
    "-m",
    type=int,
    default=512,
    help="Maximum number of tokens to generate",
    show_default=True,
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.2,
    help="Sampling temperature",
    show_default=True,
)
@click.option(
    "--top-p",
    type=float,
    default=1.0,
    help="Top-p sampling parameter",
    show_default=True,
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Stream the response",
    show_default=True,
)
@click.option(
    "--system",
    type=str,
    default=None,
    help="System message to prepend",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    envvar="MLX_API_KEY",
    help="API key for authentication (or set MLX_API_KEY env var)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name (defaults to server's model)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show verbose output including timing",
)
def chat(
    api_url: str,
    prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    system: Optional[str],
    api_key: Optional[str],
    model: Optional[str],
    verbose: bool,
):
    """Chat with the model via the API server.

    Examples:
        # Simple chat
        mlx-shard chat -p "What is Python?"

        # Read from stdin
        echo "Explain quantum computing" | mlx-shard chat -p -

        # With system message
        mlx-shard chat --system "You are a helpful coding assistant" -p "Write a function"

        # Non-streaming
        mlx-shard chat --no-stream -p "Hello"

        # With API key
        mlx-shard chat --api-key sk-xxx -p "Hello"
    """
    # Read prompt from stdin if '-'
    if prompt == "-":
        prompt = sys.stdin.read().strip()

    if not prompt:
        click.echo("Error: No prompt provided. Use --prompt or -p", err=True)
        sys.exit(1)

    # Get model name from server if not specified
    if model is None:
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = requests.get(f"{api_url}/v1/models", headers=headers)
            response.raise_for_status()
            models = response.json()
            if models.get("data") and len(models["data"]) > 0:
                model = models["data"][0]["id"]
                if verbose:
                    click.echo(f"Using model: {model}", err=True)
            else:
                click.echo("Error: No models available on server", err=True)
                sys.exit(1)
        except Exception as e:
            click.echo(f"Error fetching models: {e}", err=True)
            sys.exit(1)

    # Build messages
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Build request
    request_data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }

    # Set up headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        if stream:
            # Streaming response
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                headers=headers,
                json=request_data,
                stream=True,
            )
            response.raise_for_status()

            if verbose:
                click.echo("=" * 60, err=True)

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    click.echo(content, nl=False)
                        except json.JSONDecodeError:
                            pass

            click.echo()  # Final newline

            if verbose:
                click.echo("=" * 60, err=True)

        else:
            # Non-streaming response
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                headers=headers,
                json=request_data,
            )
            response.raise_for_status()

            result = response.json()

            if verbose:
                click.echo("=" * 60, err=True)

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                click.echo(content)

            if verbose:
                click.echo("=" * 60, err=True)
                usage = result.get("usage", {})
                click.echo(f"Prompt tokens: {usage.get('prompt_tokens', 0)}", err=True)
                click.echo(
                    f"Completion tokens: {usage.get('completion_tokens', 0)}", err=True
                )
                click.echo(f"Total tokens: {usage.get('total_tokens', 0)}", err=True)

    except requests.exceptions.RequestException as e:
        click.echo(f"Error: {e}", err=True)
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json()
                click.echo(f"Details: {json.dumps(error_detail, indent=2)}", err=True)
            except:
                click.echo(f"Response: {e.response.text}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nInterrupted", err=True)
        sys.exit(130)
