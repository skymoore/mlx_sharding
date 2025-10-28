"""
GLM-4.6 Tool Call Parser

Based on the official GLM-4.6 Tool-Integrated Reasoning Guide:
https://github.com/zai-org/GLM-4.5/blob/main/resources/glm_4.6_tir_guide.md

GLM-4.6 format:
- Reasoning content is wrapped with <think> and </think>
- Each tool call is wrapped with <tool_call> and </tool_call>
- Function name immediately follows <tool_call>
- Parameters are wrapped with <arg_key> and <arg_value>

Example:
<think>I need to calculate the Fibonacci sequence</think>
<tool_call>python
<arg_key>code</arg_key>
<arg_value>def fib(n): ...</arg_value>
</tool_call>
"""

import re
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple


def parse_arguments(json_value: str) -> Tuple[Any, bool]:
    """
    Try to parse a value as JSON.
    
    Returns:
        Tuple of (parsed_value, is_valid_json)
    """
    try:
        parsed_value = json.loads(json_value)
        return parsed_value, isinstance(parsed_value, dict)
    except:
        return json_value, False


def get_argument_type(func_name: str, arg_key: str, defined_tools: List[Dict]) -> Optional[str]:
    """
    Get the expected type of an argument from tool definitions.
    
    Args:
        func_name: Name of the function
        arg_key: Name of the argument
        defined_tools: List of tool definitions
        
    Returns:
        The type string or None if not found
    """
    name2tool = {tool["function"]["name"]: tool["function"] for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    if "parameters" not in tool or "properties" not in tool["parameters"]:
        return None
    if arg_key not in tool["parameters"]["properties"]:
        return None
    return tool["parameters"]["properties"][arg_key].get("type")


def parse_glm4_tool_calls(
    response: str, 
    defined_tools: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Parse GLM-4.6 model response to extract reasoning content, text content, and tool calls.
    
    Args:
        response: Raw model output text
        defined_tools: List of tool definitions in OpenAI format (optional, used for type checking)
        
    Returns:
        Dictionary with:
        - role: "assistant"
        - reasoning_content: Optional reasoning text
        - content: Optional response text
        - tool_calls: Optional list of tool calls in OpenAI format
    """
    text = response.strip()
    reasoning_content = None
    content = None
    tool_calls = []
    
    if defined_tools is None:
        defined_tools = []
    
    # Extract reasoning content
    if text.startswith('<think>'):
        if '</think>' in text:
            reasoning_content, text = text.split('</think>', 1)
            reasoning_content = reasoning_content.removeprefix('<think>').strip()
            text = text.strip()
        else:
            # Incomplete thinking block
            reasoning_content = text.removeprefix('<think>').strip()
            text = ""
    
    # Extract content (text before tool calls)
    if '<tool_call>' in text:
        index = text.find('<tool_call>')
        content = text[:index].strip()
        text = text[index:].strip()
    else:
        content = text.strip()
        text = ""
    
    # Extract tool calls
    tool_call_strs = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    for call in tool_call_strs:
        # Extract function name (first line after <tool_call>)
        func_name_match = re.match(r'([^\n<]+)', call.strip())
        func_name = func_name_match.group(1).strip() if func_name_match else None
        
        if func_name:
            # Extract arguments
            pairs = re.findall(
                r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', 
                call, 
                re.DOTALL
            )
            arguments = {}
            
            for arg_key, arg_value in pairs:
                arg_key = arg_key.strip()
                arg_value = arg_value.strip()
                
                # Get expected type from tool definition
                arg_type = get_argument_type(func_name, arg_key, defined_tools)
                
                # Parse non-string types as JSON
                if arg_type != 'string':
                    arg_value, is_good_json = parse_arguments(arg_value)
                
                arguments[arg_key] = arg_value
            
            # Create tool call in OpenAI format
            tool_calls.append({
                'id': "call_" + str(uuid.uuid4()).replace('-', '')[:24],
                'type': 'function',
                'function': {
                    'name': func_name,
                    'arguments': json.dumps(arguments, ensure_ascii=False)
                }
            })
    
    # Build message
    message = {'role': 'assistant'}
    
    if reasoning_content:
        message['reasoning_content'] = reasoning_content
    
    if content:
        message['content'] = content
    
    if tool_calls:
        message['tool_calls'] = tool_calls
    
    return message
