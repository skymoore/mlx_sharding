"""
Tool call parsing utilities for different model formats.

Each model family may have its own tool calling format that needs to be parsed
into the OpenAI-compatible format expected by clients.

Currently supported:
- GLM-4 / GLM-4-MoE: Uses <tool_call>, <arg_key>, <arg_value> XML-style tags

To add support for a new model:
1. Create a new parser module (e.g., qwen_moe.py)
2. Implement a parse function that returns OpenAI-compatible format
3. Import and export it here
4. Update completions.py to use it based on model_type
"""

from .glm4_moe import parse_glm4_tool_calls

__all__ = ["parse_glm4_tool_calls"]
