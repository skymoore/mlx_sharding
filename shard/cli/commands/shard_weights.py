"""Shard model weights command."""
import click
import os
import shutil
import glob
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import hf_repo_to_path


@click.command(name="shard-weights")
@click.option(
    "--model",
    required=True,
    help="Name or path of the model to shard",
)
@click.option(
    "--output-dir",
    required=True,
    help="Directory to save sharded weights and other files",
)
@click.option(
    "--start-layer",
    type=int,
    required=True,
    help="Start layer for this shard (inclusive)",
)
@click.option(
    "--end-layer",
    type=int,
    required=True,
    help="End layer for this shard (exclusive)",
)
@click.option(
    "--total-layers",
    type=int,
    required=True,
    help="Total number of layers in the model",
)
def shard_weights(model, output_dir, start_layer, end_layer, total_layers):
    """Shard model weights and copy other files."""
    os.makedirs(output_dir, exist_ok=True)
    save_sharded_weights(model, output_dir, start_layer, end_layer, total_layers)
    model_path = Path(model) if Path(model).exists() else hf_repo_to_path(model)
    copy_other_files(str(model_path), output_dir)
    click.echo(f"✓ Sharding complete: {output_dir}")


def save_sharded_weights(model, output_dir, start_layer, end_layer, total_layers):
    """Save sharded model weights."""
    model_dir = Path(model) if Path(model).exists() else hf_repo_to_path(model)
    shard_state_dict = {}
    safetensor_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    
    for file in safetensor_files:
        state_dict = mx.load(file)
        for key, value in state_dict.items():
            if key.startswith("model.layers."):
                layer_num = int(key.split(".")[2])
                if start_layer <= layer_num < end_layer:
                    shard_state_dict[key] = value
            elif start_layer == 0 and key.startswith("model.embed_tokens"):
                shard_state_dict[key] = value
            elif end_layer == total_layers and (
                key.startswith("model.norm") or key.startswith("lm_head")
            ):
                shard_state_dict[key] = value

    output_path = os.path.join(
        output_dir, f"model-{start_layer:05d}-{end_layer:05d}.safetensors"
    )
    mx.save_safetensors(output_path, shard_state_dict, metadata={"format": "mlx"})
    click.echo(f"Saved shard with layers {start_layer}-{end_layer} to {output_path}")

    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index_data = json.load(f)
        new_index_data = {"weight_map": {}}
        for key, filename in index_data["weight_map"].items():
            if key in shard_state_dict:
                new_index_data["weight_map"][
                    key
                ] = f"model-{start_layer:05d}-{end_layer:05d}.safetensors"
        new_index_file = os.path.join(
            output_dir,
            f"model-{start_layer:05d}-{end_layer:05d}.safetensors.index.json",
        )
        with open(new_index_file, "w") as f:
            json.dump(new_index_data, f, indent=2)
        click.echo(f"Updated index file saved to {new_index_file}")

    input_config_file = os.path.join(model_dir, "config.json")
    output_config_file = os.path.join(output_dir, "config.json")
    if os.path.exists(input_config_file):
        with open(input_config_file, "r") as f:
            config_data = json.load(f)
    else:
        config_data = {}
    config_data["start_layer"] = start_layer
    config_data["end_layer"] = end_layer
    with open(output_config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    click.echo(
        f"Updated config.json with start_layer and end_layer, saved to {output_config_file}"
    )


def copy_other_files(src_dir, dst_dir):
    """Copy non-weight files from source to destination."""
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks=False, ignore=None)
        else:
            if (
                not item.endswith(".safetensors")
                and item != "model.safetensors.index.json"
                and item != "config.json"
            ):
                shutil.copy2(s, d)
