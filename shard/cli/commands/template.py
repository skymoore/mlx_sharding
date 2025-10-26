"""Template management CLI commands."""
import click
import logging
import json
from pathlib import Path
from typing import Optional

from jinja2 import Template, TemplateError


# Popular models with known chat templates
POPULAR_MODELS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama 3 8B Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B Instruct v0.3",
    "Qwen/Qwen2-7B-Instruct": "Qwen 2 7B Instruct",
    "google/gemma-2-9b-it": "Gemma 2 9B IT",
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3 Mini 4K Instruct",
}


@click.group()
def template():
    """Manage chat templates for MLX Sharding."""
    pass


@template.command()
@click.option(
    "--model",
    required=True,
    help="HuggingFace model ID (e.g., meta-llama/Meta-Llama-3-8B-Instruct)",
)
@click.option(
    "--output",
    default="chat_template.jinja",
    help="Output file path",
    show_default=True,
)
@click.option(
    "--show",
    is_flag=True,
    help="Print template to stdout instead of saving to file",
)
def fetch(model: str, output: str, show: bool):
    """Fetch chat template from HuggingFace model repository."""
    try:
        from huggingface_hub import hf_hub_download
        
        click.echo(f"📥 Fetching chat template for: {model}")
        
        # Try to download chat_template.jinja first
        chat_template = None
        template_source = None
        
        try:
            template_path = hf_hub_download(
                repo_id=model,
                filename="chat_template.jinja",
                repo_type="model"
            )
            with open(template_path) as f:
                chat_template = f.read()
            template_source = "chat_template.jinja"
            click.echo(f"✓ Found chat_template.jinja in repository")
        except Exception:
            # If chat_template.jinja doesn't exist, try tokenizer_config.json
            click.echo(f"  No chat_template.jinja found, checking tokenizer_config.json...")
            try:
                config_path = hf_hub_download(
                    repo_id=model,
                    filename="tokenizer_config.json",
                    repo_type="model"
                )
                with open(config_path) as f:
                    config = json.load(f)
                
                chat_template = config.get("chat_template")
                if chat_template:
                    template_source = "tokenizer_config.json"
                    click.echo(f"✓ Found chat template in tokenizer_config.json")
            except Exception as e:
                pass
        
        if not chat_template:
            click.echo(f"❌ Error: No chat template found in {model}", err=True)
            click.echo(f"   Checked: chat_template.jinja and tokenizer_config.json", err=True)
            click.echo(f"   This model may not have a chat template defined", err=True)
            return
        
        # Show or save
        if show:
            click.echo("\n" + "=" * 80)
            click.echo(f"Chat Template for {model} (from {template_source}):")
            click.echo("=" * 80)
            click.echo(chat_template)
            click.echo("=" * 80)
        else:
            with open(output, "w") as f:
                f.write(chat_template)
            click.echo(f"✓ Chat template saved to: {output}")
            click.echo(f"  Source: {template_source}")
            click.echo(f"  Use with: mlx-shard-api --model {model} --chat-template {output}")
    
    except ImportError:
        click.echo("❌ Error: huggingface_hub is required for this command", err=True)
        click.echo("   Install with: pip install huggingface-hub", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        logging.error(f"Error fetching template: {e}", exc_info=True)


@template.command()
def list():
    """List built-in chat templates."""
    # Get the templates directory
    templates_dir = Path(__file__).parent.parent.parent / "api" / "chat_templates"
    
    if not templates_dir.exists():
        click.echo("❌ Error: Templates directory not found", err=True)
        return
    
    # List all .jinja files
    templates = sorted(templates_dir.glob("*.jinja"))
    
    if not templates:
        click.echo("No built-in templates found")
        return
    
    click.echo("\n📋 Built-in Chat Templates:")
    click.echo("=" * 80)
    
    for template_path in templates:
        template_name = template_path.stem
        
        # Map template names to model families
        model_families = {
            "llama-3-instruct": "Llama 3.x (Instruct)",
            "mistral-instruct": "Mistral (Instruct)",
            "chatml": "Qwen, GLM (ChatML format)",
            "gemma-it": "Gemma 2 (IT)",
            "phi-3": "Phi-3 (Instruct)",
        }
        
        description = model_families.get(template_name, "Unknown")
        click.echo(f"  • {template_name:20s} - {description}")
    
    click.echo("=" * 80)
    click.echo(f"\nUse 'mlx-shard template show <name>' to view a template")


@template.command()
@click.argument("name")
def show(name: str):
    """Display a built-in chat template."""
    # Get the templates directory
    templates_dir = Path(__file__).parent.parent.parent / "api" / "chat_templates"
    
    # Add .jinja extension if not provided
    if not name.endswith(".jinja"):
        name = f"{name}.jinja"
    
    template_path = templates_dir / name
    
    if not template_path.exists():
        click.echo(f"❌ Error: Template '{name}' not found", err=True)
        click.echo(f"   Use 'mlx-shard template list' to see available templates", err=True)
        return
    
    # Read and display the template
    with open(template_path) as f:
        content = f.read()
    
    click.echo("\n" + "=" * 80)
    click.echo(f"Template: {name}")
    click.echo("=" * 80)
    click.echo(content)
    click.echo("=" * 80)


@template.command()
@click.argument("file", type=click.Path(exists=True))
def validate(file: str):
    """Validate a custom chat template."""
    click.echo(f"🔍 Validating template: {file}")
    
    try:
        # Read the template
        with open(file) as f:
            template_content = f.read()
        
        # Check if it's valid Jinja2
        try:
            Template(template_content)
            click.echo("✓ Template syntax is valid")
        except TemplateError as e:
            click.echo(f"❌ Template syntax error: {e}", err=True)
            return
        
        # Check for recommended variables
        recommended_vars = ["messages", "add_generation_prompt"]
        missing_vars = []
        
        for var in recommended_vars:
            # Check if variable is referenced in template
            if f"{{{{{var}" not in template_content and f"{{% for" not in template_content:
                missing_vars.append(var)
        
        if missing_vars:
            click.echo(f"⚠ Warning: Template may be missing recommended variables:")
            for var in missing_vars:
                click.echo(f"  - {var}")
        else:
            click.echo("✓ Template contains recommended variables")
        
        # Check for common special tokens
        special_tokens = ["bos_token", "eos_token"]
        found_tokens = []
        
        for token in special_tokens:
            if token in template_content:
                found_tokens.append(token)
        
        if found_tokens:
            click.echo(f"✓ Template uses special tokens: {', '.join(found_tokens)}")
        
        # Try to render with sample data
        try:
            jinja_template = Template(template_content)
            sample_messages = [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            
            rendered = jinja_template.render(
                messages=sample_messages,
                add_generation_prompt=True,
                bos_token="<s>",
                eos_token="</s>",
            )
            
            click.echo("✓ Template renders successfully with sample data")
            
            if click.confirm("\nShow rendered output?", default=False):
                click.echo("\n" + "=" * 80)
                click.echo("Rendered Output:")
                click.echo("=" * 80)
                click.echo(rendered)
                click.echo("=" * 80)
        
        except Exception as e:
            click.echo(f"⚠ Warning: Template failed to render with sample data: {e}", err=True)
        
        click.echo("\n✓ Validation complete")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        logging.error(f"Error validating template: {e}", exc_info=True)


@template.command(name="list-popular")
def list_popular():
    """Show popular models with chat templates."""
    click.echo("\n🌟 Popular Models with Chat Templates:")
    click.echo("=" * 80)
    
    for model_id, description in POPULAR_MODELS.items():
        click.echo(f"  • {description}")
        click.echo(f"    {model_id}")
        click.echo()
    
    click.echo("=" * 80)
    click.echo("\nFetch a template with:")
    click.echo("  mlx-shard template fetch --model <model-id>")
