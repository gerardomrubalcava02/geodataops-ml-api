"""Interfaz de proveedor LLM y su implementación con el SDK oficial de Anthropic.

El agente solo conoce `LLMProvider`. Un proveedor:
- transmite un turno (`stream_turn`) emitiendo `TextDelta` y finalmente un `TurnResult`;
- sabe convertir ese turno y los resultados de herramientas a su formato de historial.
Las pruebas usan un proveedor guionizado que implementa la misma interfaz.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Any  # sin validar: lo valida el agente con Pydantic


@dataclass(frozen=True)
class ToolOutcome:
    call_id: str
    content: str  # JSON serializado
    is_error: bool = False


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass
class TurnResult:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # end_turn | tool_use | max_tokens | refusal | ...
    raw: Any = None  # contenido nativo del proveedor para reenviarlo sin cambios
    usage: dict[str, int] = field(default_factory=dict)
    model: str | None = None


class ProviderError(Exception):
    """Fallo del proveedor. `retryable` distingue 429/5xx/red de errores de configuración."""

    def __init__(self, message: str, *, retryable: bool, status: int | None = None):
        super().__init__(message)
        self.retryable, self.status = retryable, status


class ProviderTimeout(ProviderError):
    def __init__(self, message: str = "El proveedor LLM no respondió a tiempo"):
        super().__init__(message, retryable=True)


class LLMProvider(abc.ABC):
    name: str
    model: str

    @abc.abstractmethod
    def stream_turn(
        self, system: str, messages: list[Any], tools: list[ToolSpec]
    ) -> AsyncIterator[TextDelta | TurnResult]: ...

    def user_message(self, text: str) -> Any:
        return {"role": "user", "content": text}

    @abc.abstractmethod
    def assistant_message(self, turn: TurnResult) -> Any: ...

    @abc.abstractmethod
    def tool_results_message(self, outcomes: list[ToolOutcome]) -> Any: ...

    async def aclose(self) -> None:
        return None


class AnthropicProvider(LLMProvider):
    """Claude vía `anthropic.AsyncAnthropic` con streaming y tool use manual.

    - Se reenvía `message.content` completo (incluye bloques de thinking) en cada turno.
    - `fallbacks="default"` (beta) reintenta en el servidor si el modelo rechaza la solicitud.
    - El SDK reintenta 429/5xx/red (max_retries=2); el timeout por solicitud es configurable.
    """

    name = "anthropic"
    _FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        effort: str,
        max_tokens: int,
        request_timeout_s: float,
        server_fallbacks: bool = True,
    ):
        import anthropic

        self._anthropic = anthropic
        # api_key=None deja que el SDK resuelva ANTHROPIC_API_KEY / perfiles de `ant auth`.
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=request_timeout_s, max_retries=2
        )
        self.model, self._effort, self._max_tokens = model, effort, max_tokens
        self._fallbacks = server_fallbacks

    def _tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "eager_input_streaming": True,  # el agente valida cada entrada con Pydantic
            }
            for t in tools
        ]

    async def stream_turn(
        self, system: str, messages: list[Any], tools: list[ToolSpec]
    ) -> AsyncIterator[TextDelta | TurnResult]:
        a = self._anthropic
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": messages,
            "tools": self._tools(tools),
            "output_config": {"effort": self._effort},
        }
        if self._fallbacks:
            kwargs |= {"betas": [self._FALLBACK_BETA], "fallbacks": "default"}
        try:
            async with self._client.beta.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield TextDelta(event.text)
                message = await stream.get_final_message()
        except a.APITimeoutError as exc:
            raise ProviderTimeout() from exc
        except a.RateLimitError as exc:
            raise ProviderError("Límite de solicitudes del proveedor", retryable=True, status=429) from exc
        except a.AuthenticationError as exc:
            raise ProviderError("Credenciales del proveedor inválidas", retryable=False, status=401) from exc
        except a.APIStatusError as exc:
            raise ProviderError(
                f"Error del proveedor ({exc.status_code})",
                retryable=exc.status_code >= 500, status=exc.status_code,
            ) from exc
        except a.APIConnectionError as exc:
            raise ProviderError("No se pudo conectar con el proveedor", retryable=True) from exc
        except ValueError as exc:  # JSON de herramienta imposible de parsear (eager streaming)
            raise ProviderError("Entrada de herramienta ilegible", retryable=True) from exc

        calls = [
            ToolCall(b.id, b.name, b.input) for b in message.content if b.type == "tool_use"
        ]
        text = "".join(b.text for b in message.content if b.type == "text")
        yield TurnResult(
            text=text,
            tool_calls=calls,
            stop_reason=message.stop_reason or "end_turn",
            raw=message.content,
            usage={"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens},
            model=message.model,
        )

    def assistant_message(self, turn: TurnResult) -> Any:
        return {"role": "assistant", "content": turn.raw}

    def tool_results_message(self, outcomes: list[ToolOutcome]) -> Any:
        # Todos los resultados en un único mensaje de usuario (preserva llamadas paralelas).
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": o.call_id, "content": o.content, "is_error": o.is_error}
                for o in outcomes
            ],
        }

    async def aclose(self) -> None:
        await self._client.close()
