"""
The printout after every call: what we sent to the model.

The model's answer is only half the story. This shows the other half, the
request, so you can see where "memory" and "rules" really come from.
"""

LINE_WIDTH = 60


def shorten(text):
    text = " ".join(text.split())
    return text if len(text) <= LINE_WIDTH else text[: LINE_WIDTH - 3] + "..."


def tokens(count):
    return "?" if count is None else str(count)


def header(call_number, reply):
    replay_note = "  (replay)" if reply.replayed else ""
    return f"--- what we sent to the model | call {call_number} | {reply.model}{replay_note} ---"


def call_text(call):
    arguments = ", ".join(f"{name}={value!r}" for name, value in call["arguments"].items())
    return f"{call['name']}({arguments})"


def message_text(message):
    """One line per message. Tool calls and tool results get their own look."""
    if message.get("tool_calls"):
        calls = ", ".join(call_text(call) for call in message["tool_calls"])
        before = f"{message['content']} " if message["content"] else ""
        return shorten(f"{before}-> please run {calls}")
    if message["role"] == "tool":
        return shorten(f"{message['name']} returned {message['content']}")
    return shorten(message["content"])


def show_request(call_number, messages, reply, system=None, dropped=0, tools=None):
    """Print the conversation we sent, one line per message, and its size in tokens."""
    print(header(call_number, reply))
    print(f"system prompt   {shorten(system) if system else '(none)'}")
    if tools is not None:
        print(f"tools offered   {', '.join(tool['name'] for tool in tools) if tools else '(none)'}")
    dropped_note = f"   ({dropped} older messages not sent)" if dropped else ""
    print(f"conversation    {len(messages)} message{'s' if len(messages) != 1 else ''}{dropped_note}")
    for position, message in enumerate(messages):
        print(f"   [{position}] {message['role']:<9}  {message_text(message)}")
    print(f"input tokens    {tokens(reply.input_tokens)}")
    print()


def show_text_request(text, reply):
    """Print the plain-text request of round 0 once: one string, no roles."""
    replay_note = "  (replay)" if reply.replayed else ""
    print(f"--- what we sent to the model | {reply.model}{replay_note} ---")
    print("roles           (none: no system, no user, no assistant)")
    print(f"text            {shorten(text)}")
    print(f"input tokens    {tokens(reply.input_tokens)}")
    print()


def show_continuations(runs):
    """Print what the model added, one line per run. runs = [(temperature, run, added_text)]."""
    print("The model added:")
    earlier = {}
    for temperature, run, added in runs:
        one_line = " ".join(added.split())
        same_note = "   (same as run 1)" if earlier.get(temperature) == one_line else ""
        earlier.setdefault(temperature, one_line)
        print(f"  temperature {temperature:g}, run {run}:  \033[1m{one_line}\033[0m{same_note}")
    print()
