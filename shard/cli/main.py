"""
MLX Sharding CLI - Main entry point for all commands.
"""

# Suppress transformers warnings - must be before any imports that use transformers
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import warnings
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")

import click

# Import subcommands
from shard.cli.commands.api import api
from shard.cli.commands.peer import peer
from shard.cli.commands.generate import generate
from shard.cli.commands.chat import chat
from shard.cli.commands.shard_weights import shard_weights
from shard.cli.commands.api_key import api_key
from shard.cli.commands.template import template


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
cli.add_command(chat)
utils.add_command(generate)
utils.add_command(shard_weights)
utils.add_command(api_key)
utils.add_command(template)


if __name__ == "__main__":
    cli()
