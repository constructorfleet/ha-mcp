"""Settings endpoints for opt-in public web search."""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config import get_global_settings
from ..errors import ErrorCode, create_error_response
from ..web_search import (
    SearchSettings,
    WebSearchConfigurationError,
    credential_status,
    load_credentials,
    load_search_settings,
    save_credentials,
    save_search_settings,
)


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(
        create_error_response(ErrorCode.VALIDATION_INVALID_PARAMETER, message),
        status_code=status,
    )


def _settings_payload() -> dict[str, Any]:
    settings = load_search_settings()
    return {
        "enabled": get_global_settings().enable_web_search,
        "default_provider": settings.default_provider,
        "safe_search": settings.safe_search,
        "max_results": settings.max_results,
        "domain_allowlist": list(settings.domain_allowlist),
        "domain_blocklist": list(settings.domain_blocklist),
        "credentials": credential_status(),
        "unavailable_providers": {
            "bing": "The public Bing Search API was retired.",
            "duckduckgo": "DuckDuckGo has no supported general-search API.",
        },
        "retention": "disabled",
    }


async def _get_web_search(_: Request) -> JSONResponse:
    try:
        return JSONResponse(_settings_payload())
    except WebSearchConfigurationError as exc:
        return _error(str(exc), 409)


def _string_list(payload: Any, name: str) -> tuple[str, ...] | None:
    value = payload.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return tuple(item.strip().lower() for item in value if item.strip())


def _parse_settings(payload: dict[str, Any]) -> SearchSettings | None:
    enabled, maximum = payload.get("enabled"), payload.get("max_results")
    provider, safe = payload.get("default_provider"), payload.get("safe_search")
    allowlist, blocklist = (
        _string_list(payload, "domain_allowlist"),
        _string_list(payload, "domain_blocklist"),
    )
    if (
        not isinstance(enabled, bool)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 1 <= maximum <= 10
        or provider not in ("kagi", "google")
        or safe not in ("off", "moderate", "strict")
        or allowlist is None
        or blocklist is None
    ):
        return None
    return SearchSettings(False, provider, safe, maximum, allowlist, blocklist)


def _update_credentials(payload: dict[str, Any]) -> JSONResponse | None:
    credentials = payload.get("credentials")
    if credentials is None:
        return None
    if not isinstance(credentials, dict):
        return _error("credentials must be an object.")
    current = load_credentials()
    for provider_name in ("kagi", "google"):
        entry = credentials.get(provider_name)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            return _error(f"Invalid {provider_name} credentials.")
        if entry.get("clear") is True:
            current.pop(provider_name, None)
            continue
        api_key = entry.get("api_key")
        engine_id = entry.get("engine_id")
        if not isinstance(api_key, str) or not api_key.strip():
            return _error(
                f"{provider_name} API key is required when replacing credentials."
            )
        if provider_name == "google" and (
            not isinstance(engine_id, str) or not engine_id.strip()
        ):
            return _error("Google search-engine ID is required.")
        current[provider_name] = {"api_key": api_key.strip()}
        if provider_name == "google":
            assert isinstance(engine_id, str)
            current[provider_name]["engine_id"] = engine_id.strip()
    save_credentials(current)
    return None


async def _save_web_search(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except ValueError:
        return _error("Web-search settings request must be JSON.")
    if not isinstance(payload, dict):
        return _error("Web-search settings must be an object.")
    settings = _parse_settings(payload)
    if settings is None:
        return _error("Invalid web-search settings.")
    credential_error = _update_credentials(payload)
    if credential_error is not None:
        return credential_error
    save_search_settings(settings)
    return JSONResponse(
        {"success": True, "restart_required": True, **_settings_payload()}
    )


def build_web_search_handlers() -> dict[str, Any]:
    return {"get_web_search": _get_web_search, "save_web_search": _save_web_search}
