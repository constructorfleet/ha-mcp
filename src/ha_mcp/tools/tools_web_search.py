"""Opt-in public web-search MCP tool."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastmcp.tools import tool
from pydantic import Field

from ..errors import ErrorCode, create_error_response
from ..web_search import WebSearchConfigurationError, WebSearchProviderError, search_web
from .helpers import log_tool_usage, raise_tool_error, register_tool_methods


class WebSearchTools:
    @tool(
        name="ha_web_search",
        tags={"Web Search"},
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    @log_tool_usage
    async def ha_web_search(
        self,
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=2048,
                description="Query sent to the selected public search provider.",
            ),
        ],
        provider: Annotated[
            Literal["kagi", "google"] | None,
            Field(
                description="Configured provider to use. Omit to use the dashboard default."
            ),
        ] = None,
        result_limit: Annotated[
            int | None,
            Field(
                ge=1,
                le=10,
                description="Maximum results; server privacy limit still applies.",
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Search the public web through a configured provider.

        Query text is sent to Kagi or Google. Returns provider snippets only; it does not fetch result pages.
        """
        try:
            return await search_web(query, provider, result_limit)
        except WebSearchConfigurationError as exc:
            raise_tool_error(
                create_error_response(
                    ErrorCode.CONFIG_VALIDATION_FAILED,
                    str(exc),
                    suggestions=[
                        "Configure Web Search in Settings and restart the server."
                    ],
                )
            )
        except WebSearchProviderError as exc:
            raise_tool_error(
                create_error_response(
                    ErrorCode.CONNECTION_FAILED,
                    str(exc),
                    suggestions=[
                        "Check the provider credential, quota, and network connection."
                    ],
                )
            )


def register_web_search_tools(mcp: Any, client: Any, **kwargs: Any) -> None:
    from ..config import get_global_settings

    if get_global_settings().enable_web_search:
        register_tool_methods(mcp, WebSearchTools())
