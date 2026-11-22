"""Construye el agente a partir de la configuración (None si el LLM está desactivado)."""

from __future__ import annotations

from geodataops_ml.config import Settings
from geodataops_ml.llm.agent import Agent
from geodataops_ml.llm.provider import AnthropicProvider, LLMProvider
from geodataops_ml.services import GeoService


def build_provider(settings: Settings) -> LLMProvider | None:
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
            effort=settings.llm_effort,
            max_tokens=settings.llm_max_tokens,
            request_timeout_s=settings.llm_request_timeout_s,
            server_fallbacks=settings.llm_server_fallbacks,
        )
    return None


def build_agent(settings: Settings, service: GeoService, provider: LLMProvider | None = None) -> Agent | None:
    provider = provider or build_provider(settings)
    if provider is None:
        return None
    return Agent(
        provider, service,
        total_timeout_s=settings.llm_total_timeout_s,
        tool_timeout_s=settings.llm_tool_timeout_s,
        max_tool_rounds=settings.llm_max_tool_rounds,
    )
