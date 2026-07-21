"""Regression coverage for web-search configuration and provider behavior."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest
from starlette.requests import Request

from ha_mcp import web_search
from ha_mcp.settings_ui._handlers_web_search import build_web_search_handlers
from ha_mcp.tools import tools_web_search


@pytest.fixture(autouse=True)
def isolated_web_search_dir(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(web_search, "get_data_dir", lambda: tmp_path)


def test_credentials_are_encrypted_and_never_returned(tmp_path) -> None:
    web_search.save_credentials({"kagi": {"api_key": "very-secret"}})

    assert web_search.load_credentials() == {"kagi": {"api_key": "very-secret"}}
    assert "very-secret" not in (tmp_path / "web_search_credentials.enc").read_text()
    assert web_search.credential_status() == {"kagi": True, "google": False}
    assert (tmp_path / ".web_search_credentials.key").stat().st_mode & 0o777 == 0o600


def test_domain_blocklist_overrides_allowlist() -> None:
    settings = web_search.SearchSettings(
        enabled=True,
        domain_allowlist=("example.com",),
        domain_blocklist=("blocked.example.com",),
    )

    assert web_search._host_allowed("https://www.example.com/path", settings)
    assert not web_search._host_allowed("https://blocked.example.com/path", settings)
    assert not web_search._host_allowed("https://other.example.net/path", settings)


def test_clean_domains_strips_scheme_port_and_path() -> None:
    # Users paste full URLs into the allow/block fields; the stored entry must
    # reduce to the bare hostname so it can match urlparse(url).hostname.
    assert web_search._clean_domains(
        ["http://Example.com", "https://foo.test/path", "bar.test:8443", " baz.test "]
    ) == ("example.com", "foo.test", "bar.test", "baz.test")


def test_host_allowed_matches_domain_entered_with_scheme() -> None:
    settings = web_search.SearchSettings(
        enabled=True,
        domain_allowlist=web_search._clean_domains(["http://example.com"]),
    )

    assert web_search._host_allowed("https://example.com/a", settings)
    assert web_search._host_allowed("https://www.example.com/a", settings)
    assert not web_search._host_allowed("https://other.test/a", settings)


@pytest.mark.asyncio
async def test_kagi_search_normalizes_and_filters_results(monkeypatch) -> None:
    from ha_mcp.config import get_global_settings

    get_global_settings().enable_web_search = True
    web_search.save_search_settings(
        web_search.SearchSettings(enabled=True, domain_allowlist=("allowed.test",))
    )
    web_search.save_credentials({"kagi": {"api_key": "secret"}})

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "title": "Allowed",
                            "url": "https://allowed.test/a",
                            "snippet": "yes",
                        },
                        {
                            "title": "Blocked",
                            "url": "https://other.test/a",
                            "snippet": "no",
                        },
                    ]
                },
            )

    monkeypatch.setattr(web_search.httpx, "AsyncClient", lambda **_kwargs: Client())
    result = await web_search.search_web("lights", "kagi", 8)

    assert result["provider"] == "kagi"
    assert result["results"] == [
        {
            "title": "Allowed",
            "url": "https://allowed.test/a",
            "display_url": "https://allowed.test/a",
            "snippet": "yes",
            "provider": "kagi",
        }
    ]


def test_normalize_google_result_tolerates_null_pagemap() -> None:
    # Google Custom Search can return an explicit ``"pagemap": null`` for items
    # lacking metadata; ``dict.get(key, {})`` returns None (not {}) in that case.
    item = {
        "title": "T",
        "link": "https://example.com/a",
        "snippet": "s",
        "pagemap": None,
    }

    result = web_search._normalize_result(item, "google")

    assert result["url"] == "https://example.com/a"
    assert "published_date" not in result


@pytest.mark.asyncio
async def test_provider_network_error_is_wrapped_as_provider_error(monkeypatch) -> None:
    from ha_mcp.config import get_global_settings

    get_global_settings().enable_web_search = True
    web_search.save_search_settings(web_search.SearchSettings(enabled=True))
    web_search.save_credentials({"kagi": {"api_key": "secret"}})

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            raise httpx.ConnectError("name resolution failed")

    monkeypatch.setattr(web_search.httpx, "AsyncClient", lambda **_kwargs: Client())

    # A raw httpx failure must surface as WebSearchProviderError (which the
    # tool maps to a structured ToolError), not an unhandled exception.
    with pytest.raises(web_search.WebSearchProviderError):
        await web_search.search_web("lights", "kagi", 5)


@pytest.mark.asyncio
async def test_provider_malformed_json_is_wrapped_as_provider_error(monkeypatch) -> None:
    from ha_mcp.config import get_global_settings

    get_global_settings().enable_web_search = True
    web_search.save_search_settings(web_search.SearchSettings(enabled=True))
    web_search.save_credentials({"kagi": {"api_key": "secret"}})

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            return httpx.Response(200, text="not json")

    monkeypatch.setattr(web_search.httpx, "AsyncClient", lambda **_kwargs: Client())

    with pytest.raises(web_search.WebSearchProviderError):
        await web_search.search_web("lights", "kagi", 5)


@pytest.mark.asyncio
async def test_search_rejects_disabled_without_calling_provider() -> None:
    web_search.save_search_settings(web_search.SearchSettings(enabled=False))

    # The registration feature flag is the authoritative enablement control.
    from ha_mcp.config import get_global_settings

    get_global_settings().enable_web_search = False

    with pytest.raises(web_search.WebSearchConfigurationError, match="disabled"):
        await web_search.search_web("lights", None, None)


def test_corrupt_encrypted_credentials_are_rejected(tmp_path) -> None:
    (tmp_path / "web_search_credentials.enc").write_text("not encrypted")
    (tmp_path / ".web_search_credentials.key").write_bytes(
        web_search.Fernet.generate_key()
    )

    with pytest.raises(web_search.WebSearchConfigurationError, match="decrypt"):
        web_search.load_credentials()


def _json_request(payload: dict[str, object]) -> Request:
    body = json.dumps(payload).encode()

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )


@pytest.mark.asyncio
async def test_settings_endpoint_masks_credentials_and_can_clear(monkeypatch) -> None:
    from ha_mcp.settings_ui import _handlers_web_search as handler_module

    monkeypatch.setattr(
        handler_module,
        "get_global_settings",
        lambda: type("Settings", (), {"enable_web_search": False})(),
    )
    web_search.save_credentials({"kagi": {"api_key": "secret"}})
    handlers = build_web_search_handlers()
    response = await handlers["get_web_search"](
        Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    )
    assert "secret" not in response.body.decode()
    assert json.loads(response.body)["credentials"]["kagi"] is True

    saved = await handlers["save_web_search"](
        _json_request(
            {
                "enabled": False,
                "default_provider": "kagi",
                "safe_search": "moderate",
                "max_results": 5,
                "domain_allowlist": [],
                "domain_blocklist": [],
                "credentials": {"kagi": {"clear": True}},
            }
        )
    )
    assert saved.status_code == 200
    assert web_search.credential_status()["kagi"] is False


@pytest.mark.asyncio
async def test_save_persists_submitted_enabled_flag(monkeypatch) -> None:
    from ha_mcp.settings_ui import _handlers_web_search as handler_module

    monkeypatch.setattr(
        handler_module,
        "get_global_settings",
        lambda: type("Settings", (), {"enable_web_search": False})(),
    )
    handlers = build_web_search_handlers()

    saved = await handlers["save_web_search"](
        _json_request(
            {
                "enabled": True,
                "default_provider": "kagi",
                "safe_search": "moderate",
                "max_results": 5,
                "domain_allowlist": [],
                "domain_blocklist": [],
            }
        )
    )

    assert saved.status_code == 200
    # The submitted flag must round-trip, not be silently forced to False.
    assert web_search.load_search_settings().enabled is True


def test_tool_registration_is_opt_in(monkeypatch) -> None:
    settings = type("Settings", (), {"enable_web_search": False})()
    monkeypatch.setattr("ha_mcp.config.get_global_settings", lambda: settings)
    mcp = MagicMock()

    tools_web_search.register_web_search_tools(mcp, MagicMock())
    mcp.add_tool.assert_not_called()

    settings.enable_web_search = True
    tools_web_search.register_web_search_tools(mcp, MagicMock())
    assert mcp.add_tool.call_args.args[0].__name__ == "ha_web_search"
