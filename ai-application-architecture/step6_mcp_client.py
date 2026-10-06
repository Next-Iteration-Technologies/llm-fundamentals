"""
Step 6, the client: the chatbot gets its tools from the LoDA MCP server.

The chatbot no longer holds any tool code. At start-up it starts the server,
asks it which tools it has, and hands their names, descriptions and
parameters to the model, exactly like in step 2. When the model asks for a
tool, the chatbot forwards the call to the server. Same answers, less code.

Try it:
    uv run step6_mcp_client.py --demo
    uv run step6_mcp_client.py --replay

    You: Hi, I'm Priya. I can't find my invoice 1234.
    You: My upload failed, it says the file is too large.
"""

import asyncio
import concurrent.futures
import sys
import threading
from pathlib import Path

from mcp import Client, StdioServerParameters

import llm_client
from step1_bare_model_call import run_safely
from step2_agent_loop import Tool
from step3_retrieval import SYSTEM_PROMPT
from step4_conversation_store import ConversationStore, chat_with_memory

HERE = Path(__file__).parent
STEP6_SERVER = StdioServerParameters(command=sys.executable, args=[str(HERE / "step6_mcp_server.py")])
CONNECT_TIMEOUT_SECONDS = 30
CALL_TIMEOUT_SECONDS = 60
DEMO_LINES = [
    "Hi, I'm Priya. I can't find my invoice 1234.",
    "My upload failed, it says the file is too large.",
]


class McpConnection:
    """One open connection to an MCP server, usable from ordinary (not async) code.

    The MCP client is async; the agent loop is not. So the connection lives in a
    background thread with its own event loop, and every call waits for its answer.
    Use it with `with`, so the server is stopped at the end.
    """

    def __init__(self, server: StdioServerParameters | str):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        connected = concurrent.futures.Future()
        self._lifetime = asyncio.run_coroutine_threadsafe(self._stay_connected(server, connected), self._loop)
        try:
            self._client = connected.result(timeout=CONNECT_TIMEOUT_SECONDS)
        except Exception as error:
            raise llm_client.SetupProblem(f"Could not connect to the MCP server ({error}).") from error

    async def _stay_connected(self, server, connected: concurrent.futures.Future):
        """Open the connection, keep it open until close(), and close it in the same task."""
        self._stop = asyncio.Event()
        try:
            async with Client(server) as client:
                connected.set_result(client)
                await self._stop.wait()
        except Exception as error:
            if not connected.done():
                connected.set_exception(error)
            raise

    def _wait_for(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=CALL_TIMEOUT_SECONDS)

    def list_tools(self) -> list:
        return self._wait_for(self._client.list_tools()).tools

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._wait_for(self._client.call_tool(name, arguments))
        return "\n".join(block.text for block in result.content if block.type == "text")

    def close(self):
        self._loop.call_soon_threadsafe(self._stop.set)
        self._lifetime.result(timeout=CONNECT_TIMEOUT_SECONDS)
        self._loop.call_soon_threadsafe(self._loop.stop)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def remote_tool(connection: McpConnection, listed) -> Tool:
    """A tool the server described. Running it means asking the server to run it."""
    return Tool(
        name=listed.name,
        description=listed.description or "",
        parameters=listed.input_schema,
        function=lambda **arguments: connection.call_tool(listed.name, arguments),
    )


def tools_from_server(connection: McpConnection) -> list[Tool]:
    return [remote_tool(connection, listed) for listed in connection.list_tools()]


def run_chat():
    llm = llm_client.Session("step6_mcp_client", DEMO_LINES)
    with McpConnection(STEP6_SERVER) as connection:
        tools = tools_from_server(connection)
        print(f"The LoDA MCP server offers: {', '.join(tool.name for tool in tools)}\n")
        chat_with_memory(llm, ConversationStore(), "priya-mcp", tools, SYSTEM_PROMPT)


if __name__ == "__main__":
    run_safely(run_chat)
