"""
MLX Sharding CLI - Main entry point for all commands.
"""

import click

# Import subcommands
from shard.cli.commands.api import api
from shard.cli.commands.peer import peer
from shard.cli.commands.generate import generate
from shard.cli.commands.shard_weights import shard_weights
from shard.cli.commands.api_key import api_key


@click.group()
@click.version_option(version="0.1.3", prog_name="mlx-sharding")
def cli():
    """MLX Sharding - Distributed inference for MLX models."""
    pass


@cli.group()
def utils():
    """Utility commands."""
    pass


# Register commands
cli.add_command(api)
cli.add_command(peer)
utils.add_command(generate)
utils.add_command(shard_weights)
utils.add_command(api_key)


if __name__ == "__main__":
    cli()
