"""
Chat Template Manager - Handles loading, validation, and application of chat templates.

This module provides a centralized way to manage chat templates for different model types,
ensuring that messages are formatted correctly for each model's training format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Template, TemplateError

logger = logging.getLogger(__name__)


class ChatTemplateManager:
    """
    Manages chat templates for different model types.
    
    Handles loading templates from multiple sources with priority:
    1. Custom template (provided by user)
    2. Template from tokenizer_config.json
    3. Built-in template for model type
    4. Generic fallback
    """
    
    # Built-in templates for common models
    BUILTIN_TEMPLATES = {
        "llama": "llama-3-instruct.jinja",
        "mistral": "mistral-instruct.jinja",
        "qwen": "chatml.jinja",
        "qwen2": "chatml.jinja",
        "qwen3_moe": "chatml.jinja",
        "glm": "chatml.jinja",
        "glm4_moe": "chatml.jinja",
        "gemma": "gemma-it.jinja",
        "gemma2": "gemma-it.jinja",
        "phi": "phi-3.jinja",
        "phi3": "phi-3.jinja",
    }
    
    # Models that don't natively support system messages
    NO_SYSTEM_SUPPORT = ["mistral", "gemma", "gemma2"]
    
    def __init__(
        self,
        model_type: str,
        model_path: Path,
        custom_template: Optional[str] = None
    ):
        """
        Initialize chat template manager.
        
        Args:
            model_type: Model type from config.json (e.g., "llama", "mistral")
            model_path: Path to model directory
            custom_template: Optional custom Jinja template string
        """
        self.model_type = model_type
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # Load template
        self.template = self._load_template(custom_template)
        
        if self.template:
            self.logger.info(f"✓ Chat template loaded for model type: {model_type}")
        else:
            self.logger.warning(f"⚠ No chat template found for {model_type}, using fallback")
    
    def _load_template(self, custom_template: Optional[str]) -> Optional[str]:
        """
        Load chat template from various sources with priority.
        
        Priority order:
        1. Custom template provided by user
        2. Template from tokenizer_config.json
        3. Built-in template for model type
        4. None (will use fallback)
        
        Args:
            custom_template: Optional custom Jinja template string
            
        Returns:
            Template string or None if no template found
        """
        # Priority 1: Custom template provided
        if custom_template:
            self.logger.info("Using custom chat template")
            return custom_template
        
        # Priority 2: Template from tokenizer_config.json
        tokenizer_config = self.model_path / "tokenizer_config.json"
        if tokenizer_config.exists():
            try:
                with open(tokenizer_config) as f:
                    config = json.load(f)
                    if "chat_template" in config:
                        self.logger.info("Using chat template from tokenizer_config.json")
                        return config["chat_template"]
            except Exception as e:
                self.logger.warning(f"Failed to load template from tokenizer_config.json: {e}")
        
        # Priority 3: Built-in template for model type
        template_name = self.BUILTIN_TEMPLATES.get(self.model_type)
        if template_name:
            template_path = Path(__file__).parent / "chat_templates" / template_name
            if template_path.exists():
                try:
                    with open(template_path) as f:
                        self.logger.info(f"Using built-in template: {template_name}")
                        return f.read()
                except Exception as e:
                    self.logger.warning(f"Failed to load built-in template {template_name}: {e}")
        
        # Priority 4: No template found, will use fallback
        return None
    
    def apply_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True
    ) -> str:
        """
        Apply chat template to messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add generation prompt at the end
            
        Returns:
            Formatted prompt string ready for tokenization
        """
        if not self.template:
            return self._fallback_format(messages, add_generation_prompt)
        
        try:
            # Use Jinja2 to render template
            jinja_template = Template(self.template)
            
            # Render with common variables
            rendered = jinja_template.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                bos_token="<s>",
                eos_token="</s>",
            )
            
            return rendered
            
        except TemplateError as e:
            self.logger.error(f"Error applying chat template: {e}")
            self.logger.warning("Falling back to simple formatting")
            return self._fallback_format(messages, add_generation_prompt)
        except Exception as e:
            self.logger.error(f"Unexpected error applying chat template: {e}", exc_info=True)
            return self._fallback_format(messages, add_generation_prompt)
    
    def _fallback_format(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool
    ) -> str:
        """
        Improved fallback formatting when no template is available.
        
        Args:
            messages: List of message dicts
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            Simple formatted string
        """
        formatted = []
        
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        
        if add_generation_prompt:
            formatted.append("ASSISTANT:")
        
        return "\n\n".join(formatted)
    
    def supports_system_message(self) -> bool:
        """
        Check if model natively supports system messages.
        
        Some models (like Mistral and Gemma) don't have native system message
        support and require system messages to be merged into user messages.
        
        Returns:
            True if model supports system messages, False otherwise
        """
        return self.model_type not in self.NO_SYSTEM_SUPPORT
    
    def merge_system_into_user(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Merge system message into first user message for models that don't support it.
        
        This is necessary for models like Mistral and Gemma that don't have
        native system message support in their training format.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Modified message list with system message merged
        """
        if not messages:
            return messages
        
        # Find system message
        system_msg = None
        other_msgs = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                other_msgs.append(msg)
        
        # If we have a system message and other messages, merge it
        if system_msg and other_msgs:
            # Find first user message and prepend system message
            for i, msg in enumerate(other_msgs):
                if msg["role"] == "user":
                    other_msgs[i] = {
                        "role": "user",
                        "content": f"{system_msg}\n\n{msg['content']}"
                    }
                    self.logger.debug(f"Merged system message into first user message for {self.model_type}")
                    break
            return other_msgs
        
        # No system message or no user messages, return as-is
        return other_msgs if system_msg else messages
    
    def validate_template(self, template: str) -> bool:
        """
        Validate that a template is well-formed and usable.
        
        Args:
            template: Jinja2 template string to validate
            
        Returns:
            True if template is valid, False otherwise
        """
        try:
            # Check Jinja syntax by creating a Template object
            Template(template)
            
            # Check for recommended variables
            recommended_vars = ["messages", "add_generation_prompt"]
            for var in recommended_vars:
                if f"{{{{{var}" not in template:  # Check for {{var or {%var
                    self.logger.warning(
                        f"Template missing recommended variable: {var}"
                    )
            
            return True
            
        except TemplateError as e:
            self.logger.error(f"Invalid template syntax: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error validating template: {e}")
            return False
