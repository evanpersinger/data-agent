# server.py
# HTTP wrapper around the data agent so a browser can talk to it.
# The CLI in data_agent.py still works, this is just a second entry point.

import hmac
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from agents import Runner, SQLiteSession
from agents.stream_events import RawResponsesStreamEvent
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel, Field

# Importing the agent runs data_agent.py's module level setup (load_dotenv, tool
# registration) but not main(), which is guarded by __name__ == "__main__".
from data_agent import agent

# The CLI keeps history in memory, which is fine for one long lived REPL process.
# Over HTTP each request is separate, so history has to live on disk or the agent
# forgets the previous message.
SESSIONS_DB = Path(__file__).parent.parent / "sessions.db"

logger = logging.getLogger(__name__)

app = FastAPI(title="Data Agent")


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """Optional bearer token auth. Enabled when AGENT_API_KEY is set."""
    expected = os.getenv("AGENT_API_KEY")
    if not expected:
        return
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    # Restricted charset/length: the id becomes a key in sessions.db.
    session_id: str = Field(default="web", pattern=r"^[A-Za-z0-9_-]{1,64}$")


async def stream_reply(message: str, session_id: str) -> AsyncIterator[str]:
    """Yield the agent's response text as it is generated."""
    session = SQLiteSession(session_id=session_id, db_path=SESSIONS_DB)
    result = Runner.run_streamed(agent, message, session=session)

    try:
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent) and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                yield event.data.delta
    except Exception as exc:
        # Broad on purpose: this is the outermost handler for the request. A tool
        # or API failure here would otherwise just drop the connection and leave
        # the browser showing a half finished message with no explanation.
        logger.exception("agent run failed")
        yield "\n\n[agent error: the request failed, check the server logs]"


@app.post("/api/chat")
async def chat(
    request: ChatRequest, _: None = Depends(require_api_key)
) -> StreamingResponse:
    return StreamingResponse(
        stream_reply(request.message, request.session_id),
        media_type="text/plain; charset=utf-8",
    )
