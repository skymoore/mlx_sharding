"""
Tool calling support for MLX Sharding API.
Handles both standard OpenAI tool calling format and model-specific formats (e.g., GLM).
"""
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


# Pydantic models for OpenAI tool calling spec
class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]  # {"name": str, "arguments": str (JSON)}


class ToolCallParser:
    """Base class for parsing tool calls from model output."""
    
    def can_handle(self, model_type: str) -> bool:
        """Check if this parser can handle the given model type."""
        raise NotImplementedError
    
    def parse(self, content: str) -> Tuple[str, List[ToolCall]]:
        """
        Parse tool calls from model output.
        
        Returns:
            Tuple of (cleaned_content, tool_calls)
            - cleaned_content: Content with tool call markers removed
            - tool_calls: List of parsed tool calls
        """
        raise NotImplementedError


class StandardToolCallParser(ToolCallParser):
    """
    Parser for standard OpenAI tool calling format.
    Expects model to output tool_calls in the response directly.
    """
    
    def can_handle(self, model_type: str) -> bool:
        # This is the fallback parser for models that follow OpenAI spec
        return True
    
    def parse(self, content: str) -> Tuple[str, List[ToolCall]]:
        # Standard format doesn't embed tool calls in content
        # They should be in a separate field in the response
        return content, []


class GLMToolCallParser(ToolCallParser):
    """
    Parser for GLM model tool calling format.
    GLM models output tool calls inside <think> tags as JSON.
    
    Format: <think>{"name": "function_name", "arguments": {...}}</think>
    """
    
    # Regex to match <think>...</think> blocks
    THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    
    def can_handle(self, model_type: str) -> bool:
        return model_type.startswith('glm') or 'glm' in model_type.lower()
    
    def parse(self, content: str) -> Tuple[str, List[ToolCall]]:
        """
        Parse tool calls from GLM <think> tags.
        
        Handles:
        - Single tool call: <think>{"name": "func", "arguments": {...}}</think>
        - Multiple tool calls: Multiple <think> blocks
        - Malformed JSON: Logs error and skips
        - Partial blocks: Returns content as-is (for streaming)
        """
        tool_calls = []
        cleaned_content = content
        
        # Find all <think> blocks
        matches = self.THINK_PATTERN.finditer(content)
        
        for match in matches:
            think_content = match.group(1).strip()
            
            # Try to parse as JSON
            try:
                # GLM may output JSON directly or with extra text
                # Try to extract JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', think_content)
                if json_match:
                    json_str = json_match.group(0)
                    tool_data = json.loads(json_str)
                    
                    # Validate it looks like a tool call
                    if isinstance(tool_data, dict) and 'name' in tool_data:
                        # Convert to OpenAI format
                        tool_call = ToolCall(
                            id=f"call_{len(tool_calls)}_{hash(json_str) & 0xFFFFFFFF:08x}",
                            type="function",
                            function={
                                "name": tool_data.get("name", "unknown"),
                                "arguments": json.dumps(tool_data.get("arguments", {}))
                            }
                        )
                        tool_calls.append(tool_call)
                        
                        logging.info(f"Parsed GLM tool call: {tool_call.function['name']}")
                    else:
                        logging.debug(f"JSON in <think> doesn't look like tool call: {tool_data}")
                else:
                    logging.debug(f"No JSON found in <think> block: {think_content[:100]}")
                    
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON in <think> block: {e}")
                logging.debug(f"Content was: {think_content[:200]}")
            except Exception as e:
                logging.error(f"Unexpected error parsing <think> block: {e}")
        
        # Remove <think> blocks from content if we found tool calls
        if tool_calls:
            cleaned_content = self.THINK_PATTERN.sub('', content).strip()
            # Clean up extra whitespace
            cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content)
        
        return cleaned_content, tool_calls


class StreamingToolCallParser:
    """
    Handles tool call parsing for streaming responses.
    Buffers content until complete <think> blocks are received.
    """
    
    def __init__(self, parser: ToolCallParser):
        self.parser = parser
        self.buffer = ""
        self.emitted_tool_calls = []
    
    def add_chunk(self, chunk: str) -> Tuple[str, List[ToolCall]]:
        """
        Add a chunk of streaming content and parse any complete tool calls.
        
        Returns:
            Tuple of (content_to_emit, new_tool_calls)
        """
        self.buffer += chunk
        
        # For GLM parser, check if we have complete <think> blocks
        if isinstance(self.parser, GLMToolCallParser):
            return self._parse_glm_streaming()
        else:
            # Standard parser doesn't need buffering
            return chunk, []
    
    def _parse_glm_streaming(self) -> Tuple[str, List[ToolCall]]:
        """Parse GLM streaming content with buffering."""
        content_to_emit = ""
        new_tool_calls = []
        
        # Check for complete <think>...</think> blocks
        while True:
            think_start = self.buffer.find('<think>')
            if think_start == -1:
                # No <think> tag, emit everything before potential partial tag
                # Keep last 10 chars in case we're in the middle of "<think>"
                if len(self.buffer) > 10:
                    content_to_emit = self.buffer[:-10]
                    self.buffer = self.buffer[-10:]
                break
            
            think_end = self.buffer.find('</think>', think_start)
            if think_end == -1:
                # Incomplete block, emit content before <think> and wait
                if think_start > 0:
                    content_to_emit = self.buffer[:think_start]
                    self.buffer = self.buffer[think_start:]
                break
            
            # Complete block found
            # Emit content before <think>
            if think_start > 0:
                content_to_emit += self.buffer[:think_start]
            
            # Parse the complete block
            block_end = think_end + len('</think>')
            think_block = self.buffer[think_start:block_end]
            _, tool_calls = self.parser.parse(think_block)
            
            # Track new tool calls (avoid duplicates)
            for tc in tool_calls:
                if tc.id not in [existing.id for existing in self.emitted_tool_calls]:
                    new_tool_calls.append(tc)
                    self.emitted_tool_calls.append(tc)
            
            # Remove processed block from buffer
            self.buffer = self.buffer[block_end:]
        
        return content_to_emit, new_tool_calls
    
    def finalize(self) -> Tuple[str, List[ToolCall]]:
        """
        Finalize parsing and return any remaining content.
        Call this when streaming is complete.
        """
        remaining_content = self.buffer
        self.buffer = ""
        
        # Try to parse any remaining content
        if isinstance(self.parser, GLMToolCallParser):
            cleaned, tool_calls = self.parser.parse(remaining_content)
            new_tool_calls = [
                tc for tc in tool_calls 
                if tc.id not in [existing.id for existing in self.emitted_tool_calls]
            ]
            self.emitted_tool_calls.extend(new_tool_calls)
            return cleaned, new_tool_calls
        
        return remaining_content, []


class ToolCallManager:
    """
    Manages tool call parsing for different model types.
    Automatically selects the appropriate parser based on model type.
    """
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.parsers = [
            GLMToolCallParser(),
            StandardToolCallParser(),  # Fallback
        ]
        
        # Select parser
        self.parser = self._select_parser()
        logging.info(f"Selected tool call parser: {self.parser.__class__.__name__} for model type: {model_type}")
    
    def _select_parser(self) -> ToolCallParser:
        """Select the appropriate parser for the model type."""
        for parser in self.parsers:
            if parser.can_handle(self.model_type):
                return parser
        # Should never reach here due to StandardToolCallParser fallback
        return StandardToolCallParser()
    
    def parse(self, content: str) -> Tuple[str, List[ToolCall]]:
        """Parse tool calls from model output."""
        return self.parser.parse(content)
    
    def create_streaming_parser(self) -> StreamingToolCallParser:
        """Create a streaming parser for this model type."""
        return StreamingToolCallParser(self.parser)
    
    def format_tools_for_prompt(self, tools: List[ToolDefinition]) -> str:
        """
        Format tools for inclusion in the prompt.
        Different models may need different formats.
        """
        if isinstance(self.parser, GLMToolCallParser):
            return self._format_tools_glm(tools)
        else:
            return self._format_tools_standard(tools)
    
    def _format_tools_glm(self, tools: List[ToolDefinition]) -> str:
        """Format tools for GLM models (Chinese + JSON format)."""
        if not tools:
            return ""
        
        tool_descriptions = []
        for tool in tools:
            func = tool.function
            desc = f"\n## {func.name}\n\n{func.description or '无描述'}\n"
            
            if func.parameters:
                desc += "\n参数:\n"
                props = func.parameters.get('properties', {})
                required = func.parameters.get('required', [])
                
                for param_name, param_info in props.items():
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', '')
                    is_required = 'required' if param_name in required else 'optional'
                    desc += f"- {param_name} ({param_type}, {is_required}): {param_desc}\n"
            
            tool_descriptions.append(desc)
        
        tools_text = "".join(tool_descriptions)
        
        return f"""
# 可用工具
{tools_text}
在调用工具时，请在 <think> 标签内使用 JSON 格式: {{"name": "function_name", "arguments": {{"param": "value"}}}}
"""
    
    def _format_tools_standard(self, tools: List[ToolDefinition]) -> str:
        """Format tools for standard OpenAI-compatible models."""
        if not tools:
            return ""
        
        # Standard models should receive tools in the API request, not the prompt
        # But we can add a hint in the system message
        tool_names = [tool.function.name for tool in tools]
        return f"\n\nYou have access to the following tools: {', '.join(tool_names)}"


def create_tool_call_manager(model_type: str) -> ToolCallManager:
    """Factory function to create a tool call manager for a model type."""
    return ToolCallManager(model_type)
