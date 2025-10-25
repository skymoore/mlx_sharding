"""API key generation command."""
import click
import secrets


@click.command(name="generate-api-key")
@click.option(
    "-n",
    "--count",
    type=int,
    default=1,
    help="Number of keys to generate",
    show_default=True,
)
@click.option(
    "-l",
    "--length",
    type=int,
    default=32,
    help="Length of each key in bytes",
    show_default=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file (default: print to stdout)",
)
@click.option(
    "-a",
    "--append",
    is_flag=True,
    help="Append to output file instead of overwriting",
)
def api_key(count, length, output, append):
    """Generate secure API keys for MLX Sharding FastAPI server."""
    keys = [generate_key(length) for _ in range(count)]

    if output:
        mode = "a" if append else "w"
        with open(output, mode) as f:
            for key in keys:
                f.write(f"{key}\n")
        click.echo(f"✓ Generated {count} key(s) to {output}")
    else:
        for key in keys:
            click.echo(key)


def generate_key(length: int = 32) -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(length)
