import asyncio
import inspect
import json
import os
import pprint
import re
import sys
from typing import Any, Callable, Dict, List, Optional

from jinja2 import Template
import mcp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
mcp_server_script = "mcp_server_HF.py"

system_prompt = Template("""You are a helpful assistant. 
You may answer questions directly OR decide to use one or more tools if that is the best way to help.

You have access to the following tools:
<tools>{{ tools }}</tools>
                         
Before using a tool, STOP and ask:
- Is there a tool that matches my goal exactly?
- Are all required parameters available?

Only if both are true, proceed to the tool call.  Otherwise, answer the query directly.

If a tool is appropriate and all required parameters are available, respond in the following format:
                           
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>

Only use functions that are listed above in the <tools> block. 
Do not invent or guess function names.

If none of the above tools are appropriate, return: <tool_call>[]</tool_call> and then answer directly.

""")


class MCPHFClient:
    """Client for interacting with Huggingface models using MCP tools."""

    def __init__(self, model_id, mcp_server_script):
        self.model_id = model_id
        self.mcp_server_script = mcp_server_script
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        self.model.to(device)
        self.mcp_tools = None
        self.HF_tools = None
        self.system_prompt = system_prompt

        # these cannot be made to persist.  so.
        # self.session: Optional[ClientSession] = None
        # self.stdio: Optional[Any] = None
        # self.write: Optional[Any] = None

    def mcp_session():
        def decorator(fn):
            async def wrapper(self, *args, **kwargs):
                server_params = StdioServerParameters(
                    command=sys.executable,
                    args=[self.mcp_server_script],
                )
                async with stdio_client(server_params) as (stdio, write):
                    async with ClientSession(stdio, write) as session:
                        await session.initialize()
                        return await fn(self, session, *args, **kwargs)
            return wrapper
        return decorator

    @mcp_session()
    async def get_mcp_tools(self, session) -> list:
        """Get available tools from the MCP server in json rpc format.

        Returns:
            A list of tools in json rpc format.
        """
        tools_result = await session.list_tools()
        self.mcp_tools = tools_result.tools

        return tools_result.tools

    @mcp_session()
    async def get_json_HF_tools(self, session) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in Huggingface format.

        Returns:
            A list of tools in Huggingface format.
        """
        result = await session.call_tool(
            "get_HF_json_schema",
        )


        converted_result = [json.loads(c.text) for c in result.content if isinstance(c, mcp.types.TextContent)]
        self.HF_tools = converted_result
        
        return converted_result

    @mcp_session()
    async def process_query(self, session, query: str) -> str:
        """Process a query using HF and available MCP tools.

        Args:
            query: The user query.

        Returns:
            The response from HF.
        """
        messages = self.prepare_messages(query)

        def query_LLM(messages):
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(device)

            outputs = self.model.generate(
                inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                num_return_sequences=1, 
            )
            result = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

            return result

        result = query_LLM(messages=messages)
        parsed_result = self.parse_response(result)

        if parsed_result != result:
            history = messages
            for tool_call in parsed_result:
                # Execute tool call
                tc_result = await session.call_tool(
                    tool_call['name'],
                    arguments=tool_call['arguments'],
                )
                # Add tool calls response to conversation
                history.append({"role": "assistant", "content": str(tool_call)})
                
                # Add tool response to conversation
                result = tc_result.content[0].text
                history.append({"role": "tool", "name": tool_call['name'], "content": result})

            print(history)
            final_response = query_LLM(messages=history)

        else:
            final_response = parsed_result

        return final_response
    

    def prepare_messages(
        self,
        query: str,
        tools: Optional[dict[str, any]] = None,
        history: Optional[list[dict[str, str]]] = None
    ) -> list[dict[str, str]]:
        """Prepare the system and user messages for the given query and tools.
        
        Args:
            query: The query to be answered.
            tools: The tools available to the user. Defaults to None, in which case if a
                list without content will be passed to the model.
            history: Exchange of messages, including the system_prompt from
                the first query. Defaults to None, the first message in a conversation.
        """
        tools = self.HF_tools
        system_prompt = self.system_prompt

        if tools is None:
            tools = []
        if history:
            messages = history.copy()
            messages.append({"role": "user", "content": query})
        else:
            messages = [
                {"role": "system", "content": system_prompt.render(tools=json.dumps(tools))},
                {"role": "user", "content": query}
            ]
        return messages

    def parse_response(self, text: str) -> str | dict[str, any]:
        """Parses a response from the model, returning either the
        parsed list with the tool calls parsed, or the
        model thought or response if couldn't generate one.

        Args:
            text: Response from the model.
        """
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        return text


async def main():
    """Main entry point for the client."""
    client = MCPHFClient(model_id, mcp_server_script)
    tools = await client.get_mcp_tools()
    # pprint.pprint(tools)
    HF_tools = await client.get_json_HF_tools()
    # pprint.pprint(HF_tools)

    # Example: Ask about company vacation policy
    query = "What is our company's vacation policy?"
    # query = "Please provide me a random number between 2 and 8."
    print(f"\nQuery: {query}")
    response = await client.process_query(query)
    print(f"\nResponse: {response}")



if __name__ == "__main__":
    asyncio.run(main())
