import logging
import json
import os
import random
import sys
from mcp.server.fastmcp import FastMCP
from transformers.utils import get_json_schema

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

# Create an MCP server
mcp = FastMCP(
    name="Knowledge Base"
)

@mcp.tool()
def get_random_number_between(min: int, max: int) -> int:
    """
    Gets a random number between min and max.

    Args:
        min: The minimum number.
        max: The maximum number.

    Returns:
        A random number between min and max.
    """
    return random.randint(min, max)

@mcp.tool()
def get_knowledge_base() -> str:
    """Retrieve the entire knowledge base as a formatted string.

    Returns:
        A formatted string containing all Q&A pairs from the knowledge base.
    """
    try:
        kb_path = os.path.join(os.path.dirname(__file__), "data", "kb.json")
        with open(kb_path, "r") as f:
            kb_data = json.load(f)

        # Format the knowledge base as a string
        kb_text = "Here is the retrieved knowledge base:\n\n"

        if isinstance(kb_data, list):
            for i, item in enumerate(kb_data, 1):
                if isinstance(item, dict):
                    question = item.get("question", "Unknown question")
                    answer = item.get("answer", "Unknown answer")
                else:
                    question = f"Item {i}"
                    answer = str(item)

                kb_text += f"Q{i}: {question}\n"
                kb_text += f"A{i}: {answer}\n\n"
        else:
            kb_text += f"Knowledge base content: {json.dumps(kb_data, indent=2)}\n\n"

        return kb_text
    except FileNotFoundError:
        return "Error: Knowledge base file not found"
    except json.JSONDecodeError:
        return "Error: Invalid JSON in knowledge base file"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_HF_json_schema() -> dict:
    """Retrieve properly formatted tools schema for Huggingface.

    Returns:
        A json dict containing all tools on the server.  Except this one.
    """
    tools_schema = [get_json_schema(get_random_number_between), get_json_schema(get_knowledge_base)]

    return tools_schema

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")

# view with npx @modelcontextprotocol/inspector
# /home/jonathon/workspace/HuggingFaceLLMCourse/.venv/bin/python
# mcp_server_HF.py
