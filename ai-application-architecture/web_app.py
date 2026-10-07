"""
A browser UI for the LoDA chatbot steps: chat on the left, the request/tool
printout on the right, live.

Run it: uv run web_app.py

The app itself lives in web/ (config, the agent-loop events, the FastAPI
routes). This file just starts it.
"""

import uvicorn

from web.routes import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
