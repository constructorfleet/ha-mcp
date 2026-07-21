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


@pytest.fixture(autouse=True)
def _reset_settings_singleton():
    from ha_mcp.config import _reset_global_settings

    _reset_global_settings()
    yield
    _reset_global_settings()


def test_credentials_are_encrypted_and_never_returned(tmp_path) -> None:
    web_search.save_credentials({"kagi": {"api_key": "very-secret"}})

    assert web_search.load_credentials() == {"kagi": {"api_key": "very-secret"}}
    assert "very-secret" not in (tmp_path / "web_search_credentials.enc").read_text()
    assert web_search.credential_status() == {"kagi": True, "google": False}
    assert (tmp_path / ".web_search_credentials.key").stat().st_mode & 0o777 == 0o600


def test_key_creation_writes_locked_down_key(tmp_path) -> None:
    path = tmp_path / ".web_search_credentials.key"

    key = web_search._create_key(path)

    assert path.read_bytes() == key
    assert len(key) == 44
    assert path.stat().st_mode & 0o777 == 0o600


def test_key_creation_preserves_existing_key_under_race(tmp_path) -> None:
    path = tmp_path / ".web_search_credentials.key"
    existing = web_search.Fernet.generate_key()
    path.write_bytes(existing)

    # _create_key models the branch a racing writer takes once the key file
    # exists — it must return the existing key, never overwrite it (which would
    # orphan credentials already encrypted under the first key).
    assert web_search._create_key(path) == existing
    assert path.read_bytes() == existing


def test_load_migrates_legacy_safe_search_tier_to_on(tmp_path) -> None:
    # Kagi ignores safe-search and Google's API is on/off only, so the old
    # three-tier model collapsed to On/Off. A persisted legacy value must
    # migrate to "on" (it meant "safe search enabled"), not error out.
    (tmp_path / "web_search.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "default_provider": "kagi",
                "safe_search": "strict",
                "max_results": 5,
                "domain_allowlist": [],
                "domain_blocklist": [],
            }
        )
    )

    assert web_search.load_search_settings().safe_search == "on"


def test_load_rejects_unknown_safe_search_value(tmp_path) -> None:
    (tmp_path / "web_search.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "default_provider": "kagi",
                "safe_search": "banana",
                "max_results": 5,
                "domain_allowlist": [],
                "domain_blocklist": [],
            }
        )
    )

    with pytest.raises(web_search.WebSearchConfigurationError):
        web_search.load_search_settings()


@pytest.mark.asyncio
async def test_google_safe_search_maps_on_to_active_off_to_off(monkeypatch) -> None:
    from ha_mcp.config import get_global_settings

    get_global_settings().enable_web_search = True
    web_search.save_credentials({"google": {"api_key": "k", "engine_id": "e"}})
    captured: dict[str, str] = {}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, params=None, **_kwargs):
            captured["safe"] = params["safe"]
            return httpx.Response(200, json={"items": []})

    monkeypatch.setattr(web_search.httpx, "AsyncClient", lambda **_kwargs: Client())

    web_search.save_search_settings(
        web_search.SearchSettings(
            enabled=True, default_provider="google", safe_search="on"
        )
    )
    await web_search.search_web("q", "google", 5)
    assert captured["safe"] == "active"

    web_search.save_search_settings(
        web_search.SearchSettings(
            enabled=True, default_provider="google", safe_search="off"
        )
    )
    await web_search.search_web("q", "google", 5)
    assert captured["safe"] == "off"


@pytest.mark.asyncio
async def test_save_rejects_legacy_safe_search_value(monkeypatch) -> None:
    from ha_mcp.settings_ui import _handlers_web_search as handler_module

    monkeypatch.setattr(
        handler_module,
        "get_global_settings",
        type("Settings", (), {"enable_web_search": False}),
    )
    handlers = build_web_search_handlers()

    saved = await handlers["save_web_search"](
        _json_request(
            {
                "enabled": False,
                "default_provider": "kagi",
                "safe_search": "moderate",
                "max_results": 5,
                "domain_allowlist": [],
                "domain_blocklist": [],
            }
        )
    )

    # The UI now offers only On/Off; the save API rejects the retired tiers.
    assert saved.status_code == 400


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

        async def post(self, *_args, **_kwargs):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "search": [
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
                    }
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


@pytest.mark.asyncio
async def test_kagi_request_uses_documented_contract(monkeypatch) -> None:
    # Regression test for #web-search: the API is a POST with a JSON body
    # and Bearer auth (verified against Kagi's real endpoint), not the GET
    # + query-params + "Bot" auth this originally shipped with — which 404'd
    # unconditionally regardless of the API key's validity.
    from ha_mcp.config import get_global_settings

    get_global_settings().enable_web_search = True
    web_search.save_search_settings(
        web_search.SearchSettings(enabled=True, safe_search="on")
    )
    web_search.save_credentials({"kagi": {"api_key": "secret"}})
    captured = {}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return httpx.Response(200, json={"data": {"search": []}})

    monkeypatch.setattr(web_search.httpx, "AsyncClient", lambda **_kwargs: Client())
    await web_search.search_web("lights", "kagi", 3)

    assert captured["url"] == "https://kagi.com/api/v1/search"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"] == {
        "query": "lights",
        "workflow": "search",
        "limit": 3,
        "safe_search": True,
    }


@pytest.mark.asyncio
async def test_kagi_error_response_surfaces_provider_message(monkeypatch) -> None:
    from ha_mcp.config import get_global_settings

    get_global_settings().enable_web_search = True
    web_search.save_search_settings(web_search.SearchSettings(enabled=True))
    web_search.save_credentials({"kagi": {"api_key": "bad-token"}})

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return httpx.Response(
                400,
                json={
                    "meta": {},
                    "data": None,
                    "errors": [
                        {
                            "code": "general.invalid_token",
                            "message": "Token signature failed to verify.",
                        }
                    ],
                },
            )

    monkeypatch.setattr(web_search.httpx, "AsyncClient", lambda **_kwargs: Client())

    with pytest.raises(
        web_search.WebSearchProviderError, match="Token signature failed to verify"
    ):
        await web_search.search_web("lights", "kagi", 5)


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


def test_normalize_google_result_tolerates_dict_metatags() -> None:
    # The Google API has returned pagemap.metatags as a dict instead of a list
    # in some response schemas; indexing a dict with [0] raises KeyError.
    item = {
        "title": "T",
        "link": "https://example.com/a",
        "snippet": "s",
        "pagemap": {"metatags": {"article:published_time": "2024-01-01"}},
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

        async def post(self, *_args, **_kwargs):
            raise httpx.ConnectError("name resolution failed")

    monkeypatch.setattr(web_search.httpx, "AsyncClient", lambda **_kwargs: Client())

    # A raw httpx failure must surface as WebSearchProviderError (which the
    # tool maps to a structured ToolError), not an unhandled exception.
    with pytest.raises(web_search.WebSearchProviderError):
        await web_search.search_web("lights", "kagi", 5)


@pytest.mark.asyncio
async def test_provider_malformed_json_is_wrapped_as_provider_error(
    monkeypatch,
) -> None:
    from ha_mcp.config import get_global_settings

    get_global_settings().enable_web_search = True
    web_search.save_search_settings(web_search.SearchSettings(enabled=True))
    web_search.save_credentials({"kagi": {"api_key": "secret"}})

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
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


@pytest.mark.asyncio
async def test_disabled_flag_reported_before_reading_corrupt_settings(
    tmp_path,
) -> None:
    (tmp_path / "web_search.json").write_text("not json {{{")
    from ha_mcp.config import get_global_settings

    get_global_settings().enable_web_search = False

    # A corrupt settings file must not mask the actionable "disabled" error:
    # the flag is checked before any config is read from disk.
    with pytest.raises(web_search.WebSearchConfigurationError, match="disabled"):
        await web_search.search_web("lights", None, None)


@pytest.mark.asyncio
async def test_zero_limit_is_rejected_not_silently_replaced() -> None:
    # limit=0 must be treated as an invalid value (not silently replaced with
    # max_results by ``limit or max_results``, which treats 0 as falsy).
    from ha_mcp.config import get_global_settings

    get_global_settings().enable_web_search = True
    web_search.save_search_settings(web_search.SearchSettings(enabled=True))
    web_search.save_credentials({"kagi": {"api_key": "secret"}})

    with pytest.raises(web_search.WebSearchConfigurationError, match="at least 1"):
        await web_search.search_web("lights", "kagi", 0)


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
        type("Settings", (), {"enable_web_search": False}),
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
                "safe_search": "on",
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
async def test_blank_api_key_keeps_stored_google_credential(monkeypatch) -> None:
    from ha_mcp.settings_ui import _handlers_web_search as handler_module

    monkeypatch.setattr(
        handler_module,
        "get_global_settings",
        type("Settings", (), {"enable_web_search": False}),
    )
    web_search.save_credentials({"google": {"api_key": "stored", "engine_id": "old"}})
    handlers = build_web_search_handlers()

    # The UI leaves the key box blank once a key is configured, so editing only
    # the engine ID posts an empty api_key — that must keep the stored secret,
    # not reject the save.
    saved = await handlers["save_web_search"](
        _json_request(
            {
                "enabled": False,
                "default_provider": "google",
                "safe_search": "on",
                "max_results": 5,
                "domain_allowlist": [],
                "domain_blocklist": [],
                "credentials": {"google": {"api_key": "", "engine_id": "new"}},
            }
        )
    )

    assert saved.status_code == 200
    assert web_search.load_credentials()["google"] == {
        "api_key": "stored",
        "engine_id": "new",
    }


@pytest.mark.asyncio
async def test_blank_api_key_without_stored_credential_is_rejected(monkeypatch) -> None:
    from ha_mcp.settings_ui import _handlers_web_search as handler_module

    monkeypatch.setattr(
        handler_module,
        "get_global_settings",
        type("Settings", (), {"enable_web_search": False}),
    )
    handlers = build_web_search_handlers()

    saved = await handlers["save_web_search"](
        _json_request(
            {
                "enabled": False,
                "default_provider": "google",
                "safe_search": "on",
                "max_results": 5,
                "domain_allowlist": [],
                "domain_blocklist": [],
                "credentials": {"google": {"api_key": "", "engine_id": "new"}},
            }
        )
    )

    assert saved.status_code == 400
    assert web_search.credential_status()["google"] is False


def test_atomic_write_uses_a_temp_path_unique_to_each_write(tmp_path) -> None:
    target = tmp_path / "web_search.json"
    observed: list[str] = []
    real_replace = web_search.os.replace

    def record(src, dst):
        observed.append(str(src))
        real_replace(src, dst)

    web_search._atomic_write(target, b"first")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(web_search.os, "replace", record)
        web_search._atomic_write(target, b"second")
        web_search._atomic_write(target, b"third")

    # A shared "<path>.tmp" name lets concurrent writers truncate and unlink
    # each other's in-flight file; distinct temp paths per write cannot.
    assert len(set(observed)) == 2
    assert target.read_bytes() == b"third"
    assert target.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.asyncio
async def test_save_persists_submitted_enabled_flag(monkeypatch) -> None:
    from ha_mcp.settings_ui import _handlers_web_search as handler_module

    monkeypatch.setattr(
        handler_module,
        "get_global_settings",
        type("Settings", (), {"enable_web_search": False}),
    )
    handlers = build_web_search_handlers()

    saved = await handlers["save_web_search"](
        _json_request(
            {
                "enabled": True,
                "default_provider": "kagi",
                "safe_search": "on",
                "max_results": 5,
                "domain_allowlist": [],
                "domain_blocklist": [],
            }
        )
    )

    assert saved.status_code == 200
    # The submitted flag must round-trip, not be silently forced to False.
    assert web_search.load_search_settings().enabled is True


@pytest.mark.asyncio
async def test_save_returns_structured_error_when_credentials_corrupt(
    monkeypatch, tmp_path
) -> None:
    from ha_mcp.settings_ui import _handlers_web_search as handler_module

    monkeypatch.setattr(
        handler_module,
        "get_global_settings",
        type("Settings", (), {"enable_web_search": False}),
    )
    # A corrupt credentials store makes load_credentials() raise; the save
    # handler must surface a structured 409 like the GET handler does, not a 500.
    (tmp_path / "web_search_credentials.enc").write_text("not encrypted")
    (tmp_path / ".web_search_credentials.key").write_bytes(
        web_search.Fernet.generate_key()
    )
    handlers = build_web_search_handlers()

    saved = await handlers["save_web_search"](
        _json_request(
            {
                "enabled": False,
                "default_provider": "kagi",
                "safe_search": "on",
                "max_results": 5,
                "domain_allowlist": [],
                "domain_blocklist": [],
            }
        )
    )

    assert saved.status_code == 409


def test_tool_registration_is_opt_in(monkeypatch) -> None:
    settings = type("Settings", (), {"enable_web_search": False})()
    monkeypatch.setattr("ha_mcp.config.get_global_settings", lambda: settings)
    mcp = MagicMock()

    tools_web_search.register_web_search_tools(mcp, MagicMock())
    mcp.add_tool.assert_not_called()

    settings.enable_web_search = True
    tools_web_search.register_web_search_tools(mcp, MagicMock())
    assert mcp.add_tool.call_args.args[0].__name__ == "ha_web_search"


@pytest.mark.asyncio
async def test_provider_failure_maps_to_service_call_failed(monkeypatch) -> None:
    """A declined provider request must not be reported as CONNECTION_FAILED.

    WebSearchProviderError covers both a transport failure and a provider that
    answered with an error (bad key, exhausted quota). CONNECTION_FAILED means
    "cannot reach Home Assistant" everywhere else in this codebase — its very
    suggestions tell the user to check HOMEASSISTANT_URL — so an expired Kagi
    token pointed people at the wrong system entirely.
    """
    import json as _json

    from fastmcp.exceptions import ToolError

    async def _boom(*_args, **_kwargs):
        raise web_search.WebSearchProviderError("Kagi search failed (HTTP 401)")

    monkeypatch.setattr(tools_web_search, "search_web", _boom)

    with pytest.raises(ToolError) as excinfo:
        await tools_web_search.WebSearchTools().ha_web_search("lights")

    payload = _json.loads(str(excinfo.value))
    assert payload["error"]["code"] == "SERVICE_CALL_FAILED"


@pytest.mark.asyncio
async def test_configuration_failure_maps_to_config_validation_failed(
    monkeypatch,
) -> None:
    """The sibling path stays put — an unconfigured provider is a config
    problem, not a failed call."""
    import json as _json

    from fastmcp.exceptions import ToolError

    async def _boom(*_args, **_kwargs):
        raise web_search.WebSearchConfigurationError("Kagi API key is not configured.")

    monkeypatch.setattr(tools_web_search, "search_web", _boom)

    with pytest.raises(ToolError) as excinfo:
        await tools_web_search.WebSearchTools().ha_web_search("lights")

    payload = _json.loads(str(excinfo.value))
    assert payload["error"]["code"] == "CONFIG_VALIDATION_FAILED"
