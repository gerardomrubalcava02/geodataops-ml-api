"""Bucle de tool calling con eventos para SSE.

Eventos emitidos (dicts {"event": str, "data": dict}):
  tool_call   {id, nombre, argumentos}
  tool_result {id, nombre, ok, resultado|error}
  text        {delta}
  done        {respuesta, modelo, herramientas, rondas, stop_reason, uso}
  error       {codigo, mensaje}

Controles: plazo total (deadline) aplicado a cada espera, timeout por herramienta,
límite de rondas de herramientas y manejo de `max_tokens` / `refusal`.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from geodataops_ml.llm.provider import (
    LLMProvider,
    ProviderError,
    ProviderTimeout,
    TextDelta,
    ToolCall,
    ToolOutcome,
    TurnResult,
)
from geodataops_ml.llm.tools import ToolError, run_tool, tool_specs
from geodataops_ml.services import GeoService

SYSTEM_PROMPT = """Eres el asistente de datos de GeoDataOps. Respondes en español sobre \
indicadores municipales de México y sobre el modelo predictivo de la plataforma.

Reglas:
- Toda cifra que menciones debe provenir de un resultado de herramienta de esta conversación. \
Si las herramientas no devuelven el dato, dilo explícitamente; no lo estimes ni lo inventes.
- Cita la procedencia (campo `fuente`) de los datos que uses y el periodo.
- Si un resultado indica `sintetico: true` o datos de entrenamiento sintéticos, advierte que \
son datos demostrativos y no estadísticas reales.
- Las predicciones son salidas de un modelo, no observaciones: indica la versión del modelo.
- Si una herramienta devuelve un error de argumentos, corrige los argumentos una vez; si el \
error es de datos insuficientes, explícalo al usuario.
- Sé breve: unas pocas oraciones o una tabla corta."""


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        service: GeoService,
        *,
        total_timeout_s: float,
        tool_timeout_s: float,
        max_tool_rounds: int,
    ):
        self.provider, self.service = provider, service
        self.total_timeout_s, self.tool_timeout_s = total_timeout_s, tool_timeout_s
        self.max_tool_rounds = max_tool_rounds
        self._tools = tool_specs()

    async def aclose(self) -> None:
        await self.provider.aclose()

    async def _exec(self, call: ToolCall) -> tuple[ToolOutcome, dict[str, Any]]:
        try:
            content = await asyncio.wait_for(
                run_tool(self.service, call.name, call.arguments), self.tool_timeout_s
            )
            return ToolOutcome(call.id, content), {"ok": True, "resultado": json.loads(content)}
        except ToolError as exc:
            msg = str(exc)
        except TimeoutError:
            msg = f"la herramienta excedió {self.tool_timeout_s}s"
        return ToolOutcome(call.id, msg, is_error=True), {"ok": False, "error": msg}

    async def run(self, question: str) -> AsyncIterator[dict[str, Any]]:
        deadline = time.monotonic() + self.total_timeout_s
        messages: list[Any] = [self.provider.user_message(question)]
        used: list[dict[str, Any]] = []
        usage = {"input_tokens": 0, "output_tokens": 0}

        def remaining() -> float:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ProviderTimeout("Se agotó el tiempo total de la conversación")
            return left

        try:
            for round_no in range(self.max_tool_rounds + 1):
                turn: TurnResult | None = None
                stream = self.provider.stream_turn(SYSTEM_PROMPT, messages, self._tools)
                try:
                    while True:
                        try:
                            item = await asyncio.wait_for(anext(stream), remaining())
                        except StopAsyncIteration:
                            break
                        except TimeoutError as exc:
                            raise ProviderTimeout("Se agotó el tiempo total de la conversación") from exc
                        if isinstance(item, TextDelta):
                            yield {"event": "text", "data": {"delta": item.text}}
                        else:
                            turn = item
                finally:
                    await stream.aclose()

                if turn is None:
                    raise ProviderError("El proveedor terminó sin respuesta", retryable=True)
                for k in usage:
                    usage[k] += turn.usage.get(k, 0)

                if turn.stop_reason == "refusal":
                    yield {"event": "error", "data": {"codigo": "rechazo_modelo",
                                                      "mensaje": "El modelo declinó responder esta solicitud"}}
                    return
                if turn.stop_reason == "max_tokens" and turn.tool_calls:
                    yield {"event": "error", "data": {"codigo": "respuesta_truncada",
                                                      "mensaje": "La llamada a herramienta quedó truncada"}}
                    return
                if not turn.tool_calls:
                    yield {"event": "done", "data": {
                        "respuesta": turn.text, "modelo": turn.model or self.provider.model,
                        "herramientas": used, "rondas": round_no, "stop_reason": turn.stop_reason,
                        "uso": usage,
                    }}
                    return
                if round_no == self.max_tool_rounds:
                    break

                for c in turn.tool_calls:
                    yield {"event": "tool_call", "data": {"id": c.id, "nombre": c.name, "argumentos": c.arguments}}
                results = await asyncio.gather(*(self._exec(c) for c in turn.tool_calls))
                for c, (_, summary) in zip(turn.tool_calls, results, strict=True):
                    used.append({"nombre": c.name, "argumentos": c.arguments, "ok": summary["ok"]})
                    yield {"event": "tool_result", "data": {"id": c.id, "nombre": c.name, **summary}}
                messages.append(self.provider.assistant_message(turn))
                messages.append(self.provider.tool_results_message([o for o, _ in results]))

            msg = f"Se alcanzó el máximo de {self.max_tool_rounds} rondas de herramientas"
            yield {"event": "error", "data": {"codigo": "limite_herramientas", "mensaje": msg}}
        except ProviderTimeout as exc:
            yield {"event": "error", "data": {"codigo": "tiempo_agotado", "mensaje": str(exc)}}
        except ProviderError as exc:
            yield {"event": "error", "data": {"codigo": "error_proveedor", "mensaje": str(exc),
                                              "reintentable": exc.retryable}}
