"""
Shared setup for the browser UI: which step talks to which system prompt
and tools, and the state that lives for as long as the server runs.

This does not change step1/2/3: it imports their system prompts and tools
and reuses them as-is. The step files stay exactly as the course teaches them.
"""

from pathlib import Path

import llm_client
import step1_bare_model_call as step1
import step2_agent_loop as step2
import step3_retrieval as step3

HERE = Path(__file__).parent.parent
STATIC_DIR = HERE / "web_static"

STEPS = {
    "1": {"system": step1.SYSTEM_PROMPT, "tools": None},
    "2": {"system": step2.SYSTEM_PROMPT, "tools": step2.ARCHIVE_TOOLS},
    "3": {"system": step3.SYSTEM_PROMPT, "tools": step3.TOOLS},
}

llm = llm_client.Session("web_app", demo_lines=[])
histories: dict[str, list[dict]] = {step: [] for step in STEPS}
call_numbers: dict[str, int] = {step: 0 for step in STEPS}
