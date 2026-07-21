"""Provider adapters and encrypted local configuration for web search."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from cryptography.fernet import Fernet, InvalidToken

from .utils.data_paths import get_data_dir

ProviderName = Literal["kagi", "google"]
_CREDENTIALS_FILE = "web_search_credentials.enc"
_KEY_FILE = ".web_search_credentials.key"
_CONFIG_FILE = "web_search.json"
_PROVIDERS = ("kagi", "google")


class WebSearchConfigurationError(ValueError):
    """Configuration is unavailable, invalid, or lacks provider credentials."""


class WebSearchProviderError(RuntimeError):
    """A configured provider declined or failed a search request."""


@dataclass(frozen=True)
class SearchSettings:
    enabled: bool = False
    default_provider: ProviderName = "kagi"
    safe_search: Literal["off", "on"] = "on"
    max_results: int = 5
    domain_allowlist: tuple[str, ...] = ()
    domain_blocklist: tuple[str, ...] = ()


def _data_path(name: str) -> Path:
    return get_data_dir() / name


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _create_key(path: Path) -> bytes:
    """Create the credential key file, tolerating a concurrent creator.

    Uses ``O_EXCL`` so only the first writer persists a key; a racing writer
    that loses reads the winner's key instead of overwriting it — overwriting
    would silently orphan any credentials already encrypted under the first key.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = Fernet.generate_key()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_bytes()
    with os.fdopen(fd, "wb") as handle:
        handle.write(candidate)
    return candidate


def _fernet() -> Fernet:
    path = _data_path(_KEY_FILE)
    try:
        key = path.read_bytes()
    except FileNotFoundError:
        key = _create_key(path)
    if len(key) != 44:
        raise WebSearchConfigurationError("Web-search credential key is invalid.")
    os.chmod(path, 0o600)
    return Fernet(key)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WebSearchConfigurationError(
            f"Invalid web-search configuration: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise WebSearchConfigurationError("Web-search configuration must be an object.")
    return value


def normalize_domain(value: str) -> str:
    """Reduce a user-entered domain to a bare lowercase hostname.

    Accepts bare hosts, ``host:port``, or full URLs so that a value pasted as
    ``https://example.com/path`` still matches ``urlparse(url).hostname``.
    """
    text = value.strip().lower()
    if not text:
        return ""
    if "//" not in text:
        text = "//" + text
    return urlparse(text).hostname or ""


def _clean_domains(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise WebSearchConfigurationError("Domain lists must contain strings.")
    return tuple(host for host in (normalize_domain(item) for item in value) if host)


def _normalize_safe_search(value: Any) -> Literal["off", "on"]:
    # Kagi ignores safe-search and Google's API is on/off only, so the retired
    # three-tier model (off/moderate/strict) collapses to On/Off. Persisted
    # legacy values migrate to "on" — they meant "safe search enabled".
    coerced = {"moderate": "on", "strict": "on"}.get(value, value)
    if coerced == "off":
        return "off"
    if coerced == "on":
        return "on"
    raise WebSearchConfigurationError("Web-search safe-search setting is invalid.")


def load_search_settings() -> SearchSettings:
    raw = _read_json(_data_path(_CONFIG_FILE))
    provider = raw.get("default_provider", "kagi")
    safe = _normalize_safe_search(raw.get("safe_search", "on"))
    maximum = raw.get("max_results", 5)
    if provider not in _PROVIDERS:
        raise WebSearchConfigurationError("Web-search provider is invalid.")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 1 <= maximum <= 10
    ):
        raise WebSearchConfigurationError(
            "Web-search max_results must be between 1 and 10."
        )
    return SearchSettings(
        enabled=bool(raw.get("enabled", False)),
        default_provider=provider,
        safe_search=safe,
        max_results=maximum,
        domain_allowlist=_clean_domains(raw.get("domain_allowlist", [])),
        domain_blocklist=_clean_domains(raw.get("domain_blocklist", [])),
    )


def save_search_settings(settings: SearchSettings) -> None:
    _atomic_write(
        _data_path(_CONFIG_FILE),
        json.dumps(
            {
                "enabled": settings.enabled,
                "default_provider": settings.default_provider,
                "safe_search": settings.safe_search,
                "max_results": settings.max_results,
                "domain_allowlist": list(settings.domain_allowlist),
                "domain_blocklist": list(settings.domain_blocklist),
            },
            sort_keys=True,
        ).encode(),
    )


def load_credentials() -> dict[str, dict[str, str]]:
    path = _data_path(_CREDENTIALS_FILE)
    try:
        encrypted = path.read_bytes()
    except FileNotFoundError:
        return {}
    try:
        value = json.loads(_fernet().decrypt(encrypted))
    except (InvalidToken, json.JSONDecodeError) as exc:
        raise WebSearchConfigurationError(
            "Could not decrypt web-search credentials."
        ) from exc
    if not isinstance(value, dict):
        raise WebSearchConfigurationError("Web-search credentials are invalid.")
    return {name: data for name, data in value.items() if isinstance(data, dict)}


def save_credentials(credentials: dict[str, dict[str, str]]) -> None:
    encrypted = _fernet().encrypt(json.dumps(credentials, sort_keys=True).encode())
    _atomic_write(_data_path(_CREDENTIALS_FILE), encrypted)


def credential_status() -> dict[str, bool]:
    credentials = load_credentials()
    return {
        "kagi": bool(credentials.get("kagi", {}).get("api_key")),
        "google": bool(
            credentials.get("google", {}).get("api_key")
            and credentials.get("google", {}).get("engine_id")
        ),
    }


def _host_allowed(url: str, settings: SearchSettings) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False

    def matches(domains: tuple[str, ...]) -> bool:
        return any(host == domain or host.endswith(f".{domain}") for domain in domains)

    return not matches(settings.domain_blocklist) and (
        not settings.domain_allowlist or matches(settings.domain_allowlist)
    )


def _normalize_result(item: dict[str, Any], provider: ProviderName) -> dict[str, str]:
    if provider == "kagi":
        url = str(item.get("url", ""))
        result = {
            "title": str(item.get("title", "")),
            "url": url,
            "display_url": url,
            "snippet": str(item.get("snippet", "")),
            "provider": provider,
        }
        time = item.get("time")
        if time:
            result["published_date"] = str(time)
        return result
    url = str(item.get("link", ""))
    result = {
        "title": str(item.get("title", "")),
        "url": url,
        "display_url": str(item.get("displayLink", url)),
        "snippet": str(item.get("snippet", "")),
        "provider": provider,
    }
    # Google may return an explicit ``"pagemap": null`` (or omit metatags), so
    # ``.get("pagemap", {})`` alone can yield None — normalize before indexing.
    pagemap = item.get("pagemap")
    metatags = pagemap.get("metatags") if isinstance(pagemap, dict) else None
    if metatags:
        date = metatags[0].get("article:published_time")
        if date:
            result["published_date"] = str(date)
    return result


def _provider_items(
    response: httpx.Response, label: str, path: tuple[str, ...]
) -> list[dict[str, Any]]:
    try:
        payload: Any = response.json()
    except json.JSONDecodeError as exc:
        raise WebSearchProviderError(
            f"{label} returned a malformed response."
        ) from exc
    for key in path:
        payload = payload.get(key) if isinstance(payload, dict) else None
    return payload if isinstance(payload, list) else []


def _kagi_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return "no further detail available"
    if not isinstance(payload, dict):
        return "no further detail available"
    # Kagi's own docs say the field is ``error``; the live API has been
    # observed returning ``errors`` instead, so check both defensively.
    errors = payload.get("error")
    if not isinstance(errors, list):
        errors = payload.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        return str(errors[0].get("message") or "no further detail available")
    return "no further detail available"


async def search_web(
    query: str, provider: ProviderName | None, limit: int | None
) -> dict[str, Any]:
    settings = load_search_settings()
    from .config import get_global_settings

    if not get_global_settings().enable_web_search:
        raise WebSearchConfigurationError(
            "Web search is disabled. Enable it in Settings first."
        )
    chosen = provider or settings.default_provider
    if chosen not in _PROVIDERS:
        raise WebSearchConfigurationError("Unsupported web-search provider.")
    result_limit = min(limit or settings.max_results, settings.max_results)
    if result_limit < 1:
        raise WebSearchConfigurationError("result_limit must be at least 1.")
    credentials = load_credentials().get(chosen, {})
    if chosen == "kagi":
        response = await _search_kagi(query, result_limit, settings, credentials)
    else:
        response = await _search_google(query, result_limit, settings, credentials)
    results = [_normalize_result(item, chosen) for item in response]
    return {
        "success": True,
        "provider": chosen,
        "results": [item for item in results if _host_allowed(item["url"], settings)],
    }


async def _search_kagi(
    query: str, limit: int, settings: SearchSettings, credentials: dict[str, str]
) -> list[dict[str, Any]]:
    key = credentials.get("api_key")
    if not key:
        raise WebSearchConfigurationError("Kagi API key is not configured.")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://kagi.com/api/v1/search",
                json={
                    "query": query,
                    "workflow": "search",
                    "limit": limit,
                    "safe_search": settings.safe_search == "on",
                },
                headers={"Authorization": f"Bearer {key}"},
            )
    except httpx.HTTPError as exc:
        raise WebSearchProviderError(f"Kagi search request failed: {exc}") from exc
    if response.is_error:
        raise WebSearchProviderError(
            f"Kagi search failed (HTTP {response.status_code}): "
            f"{_kagi_error_detail(response)}"
        )
    return _provider_items(response, "Kagi", ("data", "search"))


async def _search_google(
    query: str, limit: int, settings: SearchSettings, credentials: dict[str, str]
) -> list[dict[str, Any]]:
    key, engine_id = credentials.get("api_key"), credentials.get("engine_id")
    if not key or not engine_id:
        raise WebSearchConfigurationError(
            "Google API key and search-engine ID are required."
        )
    # Google's Custom Search ``safe`` parameter only accepts ``active``/``off``.
    safe = "active" if settings.safe_search == "on" else "off"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://customsearch.googleapis.com/customsearch/v1",
                params={
                    "q": query,
                    "num": limit,
                    "safe": safe,
                    "key": key,
                    "cx": engine_id,
                },
            )
    except httpx.HTTPError as exc:
        raise WebSearchProviderError(f"Google search request failed: {exc}") from exc
    if response.is_error:
        raise WebSearchProviderError(
            f"Google search failed (HTTP {response.status_code})."
        )
    return _provider_items(response, "Google", ("items",))
