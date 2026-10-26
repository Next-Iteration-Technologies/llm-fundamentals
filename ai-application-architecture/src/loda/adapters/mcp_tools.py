"""MCP connection adapter: discovers tools from a running MCP server."""

import asyncio
import concurrent.futures
import sys
import threading
from pathlib import Path

from mcp import Client, StdioServerParameters

import llm_client
from loda.core.tools import Tool

CONNECT_TIMEOUT_SECONDS = 30
CALL_TIMEOUT_SECONDS = 60

HERE = Path(__file__).parents[4]
DEFAULT_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(HERE / "step9_mcp_server_secure.py")],
)


class McpConnection:
    """Synchronous facade over the async MCP client.

    The MCP client is async; the agent loop is not. The connection lives in a
    background thread with its own event loop; every call blocks until the
    server replies. Use as a context manager so the server shuts down cleanly.
    """

    def __init__(self, server: StdioServerParameters | str):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        connected: concurrent.futures.Future = concurrent.futures.Future()
        self._lifetime = asyncio.run_coroutine_threadsafe(
            self._stay_connected(server, connected), self._loop
        )
        try:
            self._client = connected.result(timeout=CONNECT_TIMEOUT_SECONDS)
        except Exception as error:
            raise llm_client.SetupProblem(
                f"Could not connect to the MCP server ({error})."
            ) from error

    async def _stay_connected(self, server, connected: concurrent.futures.Future):
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
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(
            timeout=CALL_TIMEOUT_SECONDS
        )

    def list_tools(self) -> list:
        return self._wait_for(self._client.list_tools()).tools

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._wait_for(self._client.call_tool(name, arguments))
        return "\n".join(block.text for block in result.content if block.type == "text")

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._lifetime.result(timeout=CONNECT_TIMEOUT_SECONDS)
        self._loop.call_soon_threadsafe(self._loop.stop)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def tools_from_server(connection: McpConnection) -> list[Tool]:
    return [_remote_tool(connection, listed) for listed in connection.list_tools()]


def _remote_tool(connection: McpConnection, listed) -> Tool:
    return Tool(
        name=listed.name,
        description=listed.description or "",
        parameters=listed.input_schema,
        function=lambda **arguments, _name=listed.name: connection.call_tool(_name, arguments),
    )
