"""
Round 4: a tool call.

The model cannot know today's date or the weather: neither is in the request.
So we offer it tools. A tool is a plain Python function on this laptop, plus a
description the model reads. The model never runs anything. It answers
"please run get_weather(city='Stuttgart')", our code runs it, and we send the
result back in the next call. One question, two calls to the model.

Try it:
    uv run r4_tool_call.py --demo                       # Nexus, the default
    uv run r4_tool_call.py --demo --provider anthropic
    uv run r4_tool_call.py --demo --provider ollama-chat
    uv run r4_tool_call.py --replay                     # the trainer's recorded run
    uv run r4_tool_call.py --demo --no-tools            # the same questions, no tools offered

    You: What's today's date?
    You: What's the weather in Stuttgart right now?
    You: Is it warmer in Berlin or in Munich?

The weather comes from Open-Meteo (open-meteo.com), free and without a key.
"""

import argparse
import json
import sys
from datetime import date

import httpx

import llm_client
from request_view import call_text, show_request

DEMO_LINES = [
    "What's today's date?",
    "What's the weather in Stuttgart right now?",
    "Is it warmer in Berlin or in Munich?",
]
EXIT_WORDS = {"exit", "quit", "bye"}
MAX_CALLS_PER_QUESTION = 5       # stop if the model keeps asking for tools


# ---------- the tools: plain functions ----------

def get_today():
    return date.today().strftime("%A, %d %B %Y")


WEATHER_WORDS = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers",
    82: "heavy rain showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def get_weather(city):
    try:
        places = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": city, "count": 1}, timeout=10).json().get("results")
        if not places:
            return json.dumps({"error": f"No place called {city} found."})
        place = places[0]
        current = httpx.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m",
        }, timeout=10).json()["current"]
    except httpx.HTTPError as error:
        return json.dumps({"error": f"The weather service could not be reached ({error})."})
    return json.dumps({
        "city": place["name"],
        "country": place.get("country"),
        "temperature_c": current["temperature_2m"],
        "conditions": WEATHER_WORDS.get(current["weather_code"], f"weather code {current['weather_code']}"),
        "wind_kmh": current["wind_speed_10m"],
    })


TOOL_FUNCTIONS = {"get_today": get_today, "get_weather": get_weather}

# ---------- what the model sees: name, description, parameters ----------

TOOLS = [  # THE TOOLS: sent with every call, like the system prompt
    {
        "name": "get_today",
        "description": "Returns today's date and weekday.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_weather",
        "description": "Returns the current weather in a city: temperature in °C, conditions and wind.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, for example Stuttgart"}},
            "required": ["city"],
        },
    },
]


def tools_wanted():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-tools", action="store_true")
    flags, _ = parser.parse_known_args()
    return not flags.no_tools


def run_chat():
    tools = TOOLS if tools_wanted() else []
    llm = llm_client.Session("r4_tool_call", DEMO_LINES, default_provider="nexus", replay_provider="anthropic")
    offered = ", ".join(tool["name"] for tool in tools) or "none"
    print(f"Chat with tools ({offered}) on {llm.provider_name}. Type 'exit' to stop.\n")

    chat_history = []
    call_number = 0

    while True:
        question = llm.read_question()

        if question is None or question.lower() in EXIT_WORDS:
            print(f"Goodbye! We exchanged {len(chat_history)} messages.")
            break
        if not question:
            continue

        chat_history.append({"role": "user", "content": question})

        # THE LOOP: call the model; while it asks for a tool, run it and call again
        for _ in range(MAX_CALLS_PER_QUESTION):
            call_number += 1
            reply = llm.chat(chat_history, tools=tools)
            chat_history.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})

            if not reply.tool_calls:
                print(f"LLM: {reply.text}\n")
                show_request(call_number, chat_history[:-1], reply, tools=tools)
                break

            for call in reply.tool_calls:
                print(f"LLM asks us to run: {call_text(call)}")
            show_request(call_number, chat_history[:-1], reply, tools=tools)

            for call in reply.tool_calls:
                function = TOOL_FUNCTIONS.get(call["name"], lambda **_: f"There is no tool called {call['name']}.")
                result = llm.run_tool(call, function)
                where = "in the recording" if llm.replay else "on this laptop"
                print(f"We ran {call['name']} {where}. It returned: {result}\n")
                chat_history.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result})
        else:
            print(f"(Stopped after {MAX_CALLS_PER_QUESTION} calls for one question.)\n")


if __name__ == "__main__":
    try:
        run_chat()
    except llm_client.SetupProblem as problem:
        sys.exit(f"\n{problem}")
