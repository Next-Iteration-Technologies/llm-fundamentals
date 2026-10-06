"""A stand-in for llm_client.Session that answers from a script: no key, no network."""

from llm_client import Reply


class ScriptedLLM:
    provider_name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def chat(self, messages, system=None, tools=None):
        self.requests.append({"messages": [dict(message) for message in messages], "system": system, "tools": tools})
        return self.replies.pop(0)

    def run_tool(self, call, function):
        return function(**call["arguments"])


def answers(text):
    return Reply(text, "scripted", 10, 5)


def asks_for(tool_name, **arguments):
    call = {"id": f"call_{tool_name}", "name": tool_name, "arguments": arguments}
    return Reply("", "scripted", 10, 5, tool_calls=[call])
