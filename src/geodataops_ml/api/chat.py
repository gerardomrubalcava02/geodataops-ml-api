"""Endpoints del asistente: respuesta completa (JSON) y streaming (Server-Sent Events)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from geodataops_ml.api.errors import ServiceUnavailable, UpstreamError, UpstreamTimeout
from geodataops_ml.api.schemas import ErrorResponse
from geodataops_ml.llm.agent import Agent

router = APIRouter(prefix="/api/v1/chat", tags=["asistente"],
                   responses={503: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
                              504: {"model": ErrorResponse}})


class ChatRequest(BaseModel):
    pregunta: str = Field(min_length=1, max_length=2000, examples=["¿Cuál fue la población de 99001 en 2020?"])


def _agent(request: Request) -> Agent:
    agent = request.app.state.agent
    if agent is None:
        raise ServiceUnavailable(
            "Asistente LLM no configurado: defina GEODATAOPS_ML_LLM_PROVIDER=anthropic y una API key",
            code="llm_no_configurado",
        )
    return agent


@router.post("")
async def chat(body: ChatRequest, request: Request) -> dict[str, Any]:
    """Ejecuta la conversación completa y devuelve la respuesta final con las herramientas usadas."""
    agent = _agent(request)
    async for ev in agent.run(body.pregunta):
        if ev["event"] == "done":
            return ev["data"]
        if ev["event"] == "error":
            d = ev["data"]
            cls = UpstreamTimeout if d["codigo"] == "tiempo_agotado" else UpstreamError
            raise cls(d["mensaje"], code=d["codigo"])
    raise UpstreamError("La conversación terminó sin respuesta")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/stream", response_class=StreamingResponse,
             responses={200: {"content": {"text/event-stream": {}}}})
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """SSE: `text`, `tool_call`, `tool_result` y un evento final `done` o `error`."""
    agent = _agent(request)

    async def events() -> AsyncIterator[str]:
        async for ev in agent.run(body.pregunta):
            if await request.is_disconnected():  # el cliente se fue: dejar de consumir tokens
                break
            yield _sse(ev["event"], ev["data"])

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
