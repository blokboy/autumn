"""Tavily-backed web search tool for Groq/OpenAI-style function calling."""

import json
from typing import Any

import httpx

import credentials

TOOL_NAME = "search_web"
TAVILY_ENV_VAR = "TAVILY_API_KEY"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_MAX_RESULTS = 5

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Search the public web for current or outside-world information. "
            "Use this when the user explicitly asks to search the web, asks "
            "about recent facts, or needs sources outside Autumn's local docs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query to send to Tavily.",
                }
            },
            "required": ["query"],
        },
    },
}

class SearchWebError(RuntimeError):
    """Raised when Tavily search cannot be completed."""


def _coerce_query(arguments: dict[str, Any]) -> str:
    query = arguments.get("query", "")
    if not isinstance(query, str):
        query = json.dumps(query)
    return query


def _resolve_api_key() -> str:
    api_key = credentials.resolve_key("tavily", TAVILY_ENV_VAR)
    if not api_key:
        raise SearchWebError(
            "Tavily API key is not configured. Run `autumn keys add tavily <key>` "
            f"or set {TAVILY_ENV_VAR}."
        )
    return api_key


def _error_detail(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("error"), str):
            return detail["error"]
        if isinstance(detail, str):
            return detail
        if isinstance(payload.get("error"), str):
            return payload["error"]
    return fallback


def _request_search(query: str, api_key: str, *, timeout_seconds: float) -> dict[str, Any]:
    try:
        response = httpx.post(
            TAVILY_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": _DEFAULT_MAX_RESULTS,
                "topic": "general",
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False,
            },
            timeout=timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        raise SearchWebError(f"Tavily search timed out after {timeout_seconds:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise SearchWebError(f"Tavily search request failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise SearchWebError("Tavily search returned invalid JSON.") from exc

    if response.status_code >= 400:
        detail = _error_detail(payload, response.text)
        raise SearchWebError(f"Tavily search failed with HTTP {response.status_code}: {detail}")

    if not isinstance(payload, dict):
        raise SearchWebError("Tavily search returned an unexpected response shape.")
    return payload


def _search(query: str, *, timeout_seconds: float) -> dict[str, Any]:
    api_key = _resolve_api_key()
    return _request_search(query, api_key, timeout_seconds=timeout_seconds)


def _result_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, dict)]


def format_results(payload: dict[str, Any]) -> str:
    """Renders Tavily's response as tool-role content for the follow-up LLM call."""
    sections: list[str] = []
    answer = payload.get("answer")
    if isinstance(answer, str) and answer.strip():
        sections.append(f"Answer summary:\n{answer.strip()}")

    for index, result in enumerate(_result_items(payload), start=1):
        title = result.get("title") if isinstance(result.get("title"), str) else "Untitled result"
        url = result.get("url") if isinstance(result.get("url"), str) else ""
        content = result.get("content") if isinstance(result.get("content"), str) else ""
        body = content.strip() or "No snippet provided."
        if url:
            sections.append(f"{index}. {title}\nURL: {url}\nSnippet: {body}")
        else:
            sections.append(f"{index}. {title}\nSnippet: {body}")

    if not sections:
        return "No web results found."
    return "\n\n".join(sections)


def citation_urls(payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for result in _result_items(payload):
        url = result.get("url")
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)
    return urls


def run_tool(arguments: dict[str, Any], *, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> str:
    """Executes the `search_web` tool call and returns text for a tool message."""
    query = _coerce_query(arguments)
    payload = _search(query, timeout_seconds=timeout_seconds)
    return format_results(payload)


def citations_for_tool_call(
    arguments: dict[str, Any], *, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
) -> list[str]:
    """Citation URLs for a successful `search_web` call's arguments."""
    query = _coerce_query(arguments)
    payload = _search(query, timeout_seconds=timeout_seconds)
    return citation_urls(payload)
