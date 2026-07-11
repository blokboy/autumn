import httpx
import pytest

import search_web


class _FakeResponse:
    def __init__(self, status_code: int, payload, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _set_key(monkeypatch, value: str | None = "tvly-test-key"):
    monkeypatch.setattr(search_web.credentials, "resolve_key", lambda provider, env_var: value)


def test_run_tool_raises_clear_error_when_tavily_key_is_missing(monkeypatch):
    _set_key(monkeypatch, None)

    with pytest.raises(search_web.SearchWebError, match="Tavily API key is not configured"):
        search_web.run_tool({"query": "latest Python release"})


def test_run_tool_posts_to_tavily_and_formats_results(monkeypatch):
    _set_key(monkeypatch)
    calls: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _FakeResponse(
            200,
            {
                "answer": "Python 3.14 is the current feature release.",
                "results": [
                    {
                        "title": "Python Releases",
                        "url": "https://www.python.org/downloads/",
                        "content": "Download the latest version of Python.",
                    },
                    {
                        "title": "Python 3.14 docs",
                        "url": "https://docs.python.org/3.14/",
                        "content": "Library reference and language documentation.",
                    },
                ],
            },
        )

    monkeypatch.setattr(search_web.httpx, "post", fake_post)

    text = search_web.run_tool({"query": "latest Python release"}, timeout_seconds=3)

    assert calls == [
        {
            "url": search_web.TAVILY_SEARCH_URL,
            "headers": {
                "Authorization": "Bearer tvly-test-key",
                "Content-Type": "application/json",
            },
            "json": {
                "query": "latest Python release",
                "search_depth": "basic",
                "max_results": 5,
                "topic": "general",
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False,
            },
            "timeout": 3,
        }
    ]
    assert "Answer summary:" in text
    assert "Python 3.14 is the current feature release." in text
    assert "1. Python Releases" in text
    assert "URL: https://www.python.org/downloads/" in text
    assert "Snippet: Download the latest version of Python." in text
    assert "2. Python 3.14 docs" in text


def test_citations_for_tool_call_returns_unique_result_urls(monkeypatch):
    _set_key(monkeypatch)
    call_count = 0

    def fake_post(url, *, headers, json, timeout):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(
            200,
            {
                "results": [
                    {"title": "One", "url": "https://example.com/one", "content": "First"},
                    {"title": "Duplicate", "url": "https://example.com/one", "content": "Again"},
                    {"title": "Two", "url": "https://example.com/two", "content": "Second"},
                    {"title": "No URL", "content": "Ignored"},
                ]
            },
        )

    monkeypatch.setattr(search_web.httpx, "post", fake_post)

    assert "https://example.com/one" in search_web.run_tool({"query": "example"})
    assert search_web.citations_for_tool_call({"query": "example"}) == [
        "https://example.com/one",
        "https://example.com/two",
    ]
    assert call_count == 2


def test_run_tool_surfaces_tavily_api_error(monkeypatch):
    _set_key(monkeypatch)

    def fake_post(url, *, headers, json, timeout):
        return _FakeResponse(
            401,
            {"detail": {"error": "Unauthorized: missing or invalid API key."}},
            text="unauthorized",
        )

    monkeypatch.setattr(search_web.httpx, "post", fake_post)

    with pytest.raises(
        search_web.SearchWebError,
        match="Tavily search failed with HTTP 401: Unauthorized: missing or invalid API key.",
    ):
        search_web.run_tool({"query": "anything"})


def test_run_tool_surfaces_timeout(monkeypatch):
    _set_key(monkeypatch)

    def fake_post(url, *, headers, json, timeout):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(search_web.httpx, "post", fake_post)

    with pytest.raises(search_web.SearchWebError, match="Tavily search timed out after 2.5 seconds."):
        search_web.run_tool({"query": "slow query"}, timeout_seconds=2.5)
