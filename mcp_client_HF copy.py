import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_complete_path_server(server_file):
    cwd = os.getcwd()
    sfp = os.path.join(cwd, server_file)
    return sfp


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

class MCPHFClient:
    """Client for interacting with Huggingface models using MCP tools."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        self.model.to(device)

        self.session: Optional[ClientSession] = None
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None

    async def get_tools_coroutine_object(self):
        tools_coroutine_object = await self.session.list_tools()
        return tools_coroutine_object

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script.
        """
        
        async def _connect():
            server_params = StdioServerParameters(
                command = sys.executable,
                args=[server_script_path],
                env={"PYTHONUNBUFFERED": "1"}
            )

            # async context manager
            async with stdio_client(server_params) as (stdio, write):
                async with ClientSession(stdio, write) as session:
                    # await asyncio.wait_for(session.initialize(), timeout=5.0)
                    await session.initialize()
                
                return stdio, write, session

        # run once, returns fully initialized session
        self.stdio, self.write, self.session = await _connect()

        # List available tools
        # tools_coroutine_object = self.get_tools_coroutine_object()
        tools_coroutine_object = await self.session.list_tools()
        print("\nConnected to server with tools:")
        for tool in tools_coroutine_object.tools:
            print(f"  - {tool.name}: {tool.description}")


    def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in OpenAI format.

        Returns:
            A list of tools in OpenAI format.
            WILL NEED TO CHANGE TO HUGGINGFACE SmolB FORMAT
        """
        tools_result = self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    def process_query(self, query: str) -> str:
        """Process a query using HF and available MCP tools.

        Args:
            query: The user query.

        Returns:
            The response from HF.
        """
        # Get available tools
        tools = self.get_mcp_tools()

        # Initial HF call
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}],
            tools=tools,
            tool_choice="auto",
        )

        # Get assistant's response
        assistant_message = response.choices[0].message

        # Initialize conversation with user query and assistant response
        messages = [
            {"role": "user", "content": query},
            assistant_message,
        ]

        # Handle tool calls if present
        if assistant_message.tool_calls:
            # Process each tool call
            for tool_call in assistant_message.tool_calls:
                # Execute tool call
                result = self.session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                # Add tool response to conversation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.content[0].text,
                    }
                )

            # Get final response from OpenAI with tool results
            final_response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="none",  # Don't allow more tool calls
            )

            return final_response.choices[0].message.content

        # No tool calls, just return the direct response
        return assistant_message.content

async def main():
    """Main entry point for the client."""
    client = MCPHFClient(model_id)
    # client.connect_to_server(get_complete_path_server("mcp_server_HF.py"))
    await client.connect_to_server("mcp_server_HF.py")

    # Example: Ask about company vacation policy
    query = "What is our company's vacation policy?"
    print(f"\nQuery: {query}")

    response = client.process_query(query)
    print(f"\nResponse: {response}")


if __name__ == "__main__":
    asyncio.run(main())

