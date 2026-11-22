"""Pruebas del asistente con un proveedor guionizado (sin red ni credenciales).

Verifican el bucle de tool calling, la validación de argumentos, los eventos SSE y los
controles de tiempo/rondas. NO verifican el comportamiento real de Claude.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from geodataops_ml.api.app import create_app
from geodataops_ml.llm.provider import LLMProvider, TextDelta, ToolCall, ToolOutcome, TurnResult
from geodataops_ml.llm.tools import tool_specs


class ScriptedProvider(LLMProvider):
    """Reproduce turnos predefinidos. Un turno es una función (messages) -> TurnResult."""

    name, model = "guion", "guion-1"

    def __init__(self, turns: list, delay_s: float = 0.0):
        self.turns, self.delay_s = list(turns), delay_s
        self.seen: list[list[Any]] = []

    async def stream_turn(self, system, messages, tools):
        self.seen.append(list(messages))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        turn = self.turns.pop(0)(messages)
        for word in turn.text.split(" "):
            if word:
                yield TextDelta(word + " ")
        yield turn

    def assistant_message(self, turn: TurnResult) -> Any:
        return {"role": "assistant", "calls": [(c.id, c.name, c.arguments) for c in turn.tool_calls]}

    def tool_results_message(self, outcomes: list[ToolOutcome]) -> Any:
        return {"role": "tool", "results": outcomes}


def call(name: str, args: Any, id_: str = "c1"):
    return lambda _m: TurnResult(text="", tool_calls=[ToolCall(id_, name, args)], stop_reason="tool_use")


def answer_from_last_result(prefix: str):
    """Responde usando el resultado real de la última herramienta (como haría un LLM fundamentado)."""

    def _turn(messages):
        outcome = messages[-1]["results"][0]
        return TurnResult(text=f"{prefix} {outcome.content[:60]}", tool_calls=[], stop_reason="end_turn")

    return _turn


def app_with(settings, provider):
    return TestClient(create_app(settings.model_copy(update={"llm_provider": "anthropic"}), llm_provider=provider))


def parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_tool_schemas_come_from_pydantic():
    specs = {s.name: s for s in tool_specs()}
    assert set(specs) == {"listar_indicadores", "listar_municipios", "consultar_indicadores",
                          "predecir_crecimiento", "metricas_modelo"}
    schema = specs["predecir_crecimiento"].input_schema
    assert schema["required"] == ["cvegeo", "periodo"] and schema["additionalProperties"] is False


def test_chat_not_configured_returns_503(client):
    r = client.post("/api/v1/chat", json={"pregunta": "hola"})
    assert r.status_code == 503 and r.json()["error"]["codigo"] == "llm_no_configurado"


def test_chat_answers_with_real_tool_data(settings):
    provider = ScriptedProvider([
        call("consultar_indicadores", {"cvegeo": "99001", "indicadores": ["poblacion_total"], "periodo_desde": 2020}),
        answer_from_last_result("Según los datos:"),
    ])
    with app_with(settings, provider) as c:
        r = c.post("/api/v1/chat", json={"pregunta": "Población de 99001 en 2020"})
        expected = c.get("/api/v1/observaciones", params={
            "cvegeo": "99001", "indicador": "poblacion_total", "periodo_desde": 2020}).json()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["herramientas"] == [{"nombre": "consultar_indicadores", "argumentos": {
        "cvegeo": "99001", "indicadores": ["poblacion_total"], "periodo_desde": 2020}, "ok": True}]
    # El resultado entregado al modelo es exactamente el de la API REST.
    tool_payload = json.loads(provider.seen[1][-1]["results"][0].content)
    assert tool_payload["items"] == expected["items"]
    assert body["rondas"] == 1


def test_invalid_tool_arguments_are_rejected_and_reported(settings):
    provider = ScriptedProvider([
        call("predecir_crecimiento", {"cvegeo": "ABC", "periodo": 20}),
        call("predecir_crecimiento", {"cvegeo": "99001", "periodo": 2015}, id_="c2"),
        answer_from_last_result("Predicción:"),
    ])
    with app_with(settings, provider) as c:
        r = c.post("/api/v1/chat/stream", json={"pregunta": "predice 99001"})
    events = parse_sse(r.text)
    results = [d for e, d in events if e == "tool_result"]
    assert results[0]["ok"] is False and "argumentos_invalidos" in results[0]["error"]
    assert results[1]["ok"] is True and results[1]["resultado"]["datos_entrenamiento_sinteticos"] is True
    # el error de validación llega al modelo marcado como is_error
    assert provider.seen[1][-1]["results"][0].is_error is True


def test_sse_event_sequence(settings):
    provider = ScriptedProvider([call("metricas_modelo", {}), answer_from_last_result("Métricas:")])
    with app_with(settings, provider) as c, c.stream("POST", "/api/v1/chat/stream", json={"pregunta": "m"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    kinds = [e for e, _ in parse_sse(text)]
    assert kinds[0] == "tool_call" and kinds[1] == "tool_result"
    assert "text" in kinds and kinds[-1] == "done"


def test_unknown_tool_and_domain_errors_go_back_to_model(settings):
    provider = ScriptedProvider([
        lambda _m: TurnResult(text="", stop_reason="tool_use", tool_calls=[
            ToolCall("a", "borrar_tabla", {}),
            ToolCall("b", "predecir_crecimiento", {"cvegeo": "01001", "periodo": 2015}),
        ]),
        lambda _m: TurnResult(text="No hay datos para ese municipio.", tool_calls=[], stop_reason="end_turn"),
    ])
    with app_with(settings, provider) as c:
        body = c.post("/api/v1/chat", json={"pregunta": "x"}).json()
    outcomes = provider.seen[1][-1]["results"]
    assert all(o.is_error for o in outcomes)
    assert "desconocida" in outcomes[0].content and "no_encontrado" in outcomes[1].content
    assert [h["ok"] for h in body["herramientas"]] == [False, False]


def test_total_timeout(settings):
    s = settings.model_copy(update={"llm_total_timeout_s": 0.2})
    slow = lambda: ScriptedProvider([answer_from_last_result("x")], delay_s=2)  # noqa: E731
    with app_with(s, slow()) as c:
        r = c.post("/api/v1/chat", json={"pregunta": "lento"})
    assert r.status_code == 504 and r.json()["error"]["codigo"] == "tiempo_agotado"
    with app_with(s, slow()) as c:
        events = parse_sse(c.post("/api/v1/chat/stream", json={"pregunta": "lento"}).text)
    assert events == [("error", {"codigo": "tiempo_agotado", "mensaje": "Se agotó el tiempo total de la conversación"})]


def test_max_tool_rounds(settings):
    s = settings.model_copy(update={"llm_max_tool_rounds": 2})
    provider = ScriptedProvider([call("listar_indicadores", {}, id_=f"c{i}") for i in range(3)])
    with app_with(s, provider) as c:
        r = c.post("/api/v1/chat", json={"pregunta": "bucle"})
    assert r.status_code == 502 and r.json()["error"]["codigo"] == "limite_herramientas"


@pytest.mark.parametrize("stop,code", [("refusal", "rechazo_modelo"), ("max_tokens", "respuesta_truncada")])
def test_refusal_and_truncation(settings, stop, code):
    provider = ScriptedProvider([
        lambda _m: TurnResult(text="", stop_reason=stop, tool_calls=[ToolCall("x", "listar_indicadores", {})])
    ])
    with app_with(settings, provider) as c:
        r = c.post("/api/v1/chat", json={"pregunta": "x"})
    assert r.status_code == 502 and r.json()["error"]["codigo"] == code


def test_question_length_validated(settings):
    with app_with(settings, ScriptedProvider([])) as c:
        assert c.post("/api/v1/chat", json={"pregunta": ""}).status_code == 422
        assert c.post("/api/v1/chat", json={"pregunta": "x" * 2001}).status_code == 422
