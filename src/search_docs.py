"""Keyword/BM25-style retrieval over this project's own documentation --
`docs/**/*.md`, `README.md`, `AGENTS.md`. Not GEPA run history, not chat
history (different retrieval shape, deferred -- see
docs/prd/chat-search-tools.md).

No embeddings, no vector store: the corpus is small and mostly static, so a
hand-rolled BM25 ranker over markdown-heading-delimited chunks, rebuilt fresh
on every call, is simpler than standing up an embedding provider or an
on-disk index for the size of documentation this project has.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
_HEADING_RE = re.compile(r"^#{1,2}\s+(.*)")

# Standard Okapi BM25 constants -- term-frequency saturation and length
# normalization strength, respectively. Untuned defaults; this corpus is far
# too small for the choice to matter much.
_BM25_K1 = 1.5
_BM25_B = 0.75

_DEFAULT_TOP_K = 5
# Chunks fed back to the model are capped so one huge section doesn't blow
# out the second Groq call's context.
_MAX_CHUNK_CHARS = 1500

TOOL_NAME = "search_docs"

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Search this project's own documentation (docs/**/*.md, README.md, "
            "AGENTS.md) for text relevant to a query. Use this for questions "
            "about Autumn's CLI commands/flags, features, or how the project "
            "itself works -- not for anything about the outside world."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or a short question to search the docs for.",
                }
            },
            "required": ["query"],
        },
    },
}


class SearchDocsError(RuntimeError):
    """Raised when the docs corpus can't be read to build the search index."""


@dataclass(frozen=True)
class DocChunk:
    """One retrievable unit: a markdown file's content split on top-level
    (`#`/`##`) headings. `heading` is `None` for a file's pre-heading
    preamble, or for a file with no headings at all."""

    path: str
    heading: str | None
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: DocChunk
    score: float


def _repo_root() -> Path:
    """Assumes this module lives at `<repo_root>/src/search_docs.py` -- true
    both for a plain source checkout and for an editable install (`pip
    install -e .`), since editable installs still resolve `__file__` to the
    real checkout path. A non-editable wheel install wouldn't ship `docs/`
    alongside it at all (`pyproject.toml`'s hatch build only packages `src`),
    so that case isn't handled here -- out of scope for this ticket."""
    return Path(__file__).resolve().parent.parent


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _iter_doc_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        paths.extend(sorted(docs_dir.rglob("*.md")))
    for name in ("README.md", "AGENTS.md"):
        candidate = root / name
        if candidate.is_file():
            paths.append(candidate)
    return paths


def _split_into_chunks(relative_path: str, text: str) -> list[DocChunk]:
    """Splits `text` on top-level (`#`/`##`) markdown headings. Anything
    before the first heading (or the entire file, if it has none) becomes
    its own headingless chunk. Deeper headings (`###`+) stay folded into
    their enclosing `#`/`##` chunk rather than fragmenting into many tiny
    chunks."""
    chunks: list[DocChunk] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append(DocChunk(path=relative_path, heading=current_heading, text=body))

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_heading = match.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()

    if not chunks:
        stripped = text.strip()
        if stripped:
            chunks.append(DocChunk(path=relative_path, heading=None, text=stripped))
    return chunks


def _load_corpus(root: Path) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    for path in _iter_doc_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SearchDocsError(f"could not read {path}: {exc}") from exc
        chunks.extend(_split_into_chunks(str(path.relative_to(root)), text))
    return chunks


def _bm25_scores(query_terms: list[str], chunk_tokens: list[list[str]]) -> list[float]:
    """Okapi BM25 over an in-memory token matrix -- ~30 lines, no dependency.
    Given this corpus's size (a handful of files, a few dozen chunks at
    most), this is cheap enough to redo on every call."""
    doc_count = len(chunk_tokens)
    if doc_count == 0:
        return []

    lengths = [len(tokens) for tokens in chunk_tokens]
    avg_length = sum(lengths) / doc_count

    doc_freq: dict[str, int] = {}
    for tokens in chunk_tokens:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scores = [0.0] * doc_count
    for term in set(query_terms):
        freq = doc_freq.get(term)
        if not freq:
            continue
        idf = math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)
        for i, tokens in enumerate(chunk_tokens):
            term_freq = tokens.count(term)
            if term_freq == 0:
                continue
            denom = term_freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * lengths[i] / avg_length)
            scores[i] += idf * (term_freq * (_BM25_K1 + 1)) / denom
    return scores


def search_docs(query: str, *, root: Path | None = None, top_k: int = _DEFAULT_TOP_K) -> list[SearchResult]:
    """Ranks this project's doc chunks against `query` with BM25, returning
    up to `top_k` results with positive score (best first). Rebuilds the
    in-memory index fresh on every call -- see module docstring.

    Raises `SearchDocsError` if any file under the corpus can't be read.
    Returns an empty list (not an error) for a query with no matching terms,
    or a corpus with no files at all.
    """
    resolved_root = root if root is not None else _repo_root()
    chunks = _load_corpus(resolved_root)
    if not chunks:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return []

    chunk_tokens = [_tokenize(f"{chunk.heading or ''}\n{chunk.text}") for chunk in chunks]
    scores = _bm25_scores(query_terms, chunk_tokens)

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    results = [SearchResult(chunk=chunk, score=score) for chunk, score in ranked if score > 0]
    return results[:top_k]


def format_results(results: list[SearchResult]) -> str:
    """Renders `search_docs` results as the plain-text tool-result content
    handed back to Groq on the second call."""
    if not results:
        return "No matching documentation found."

    sections = []
    for result in results:
        label = result.chunk.path
        if result.chunk.heading:
            label = f"{label} — {result.chunk.heading}"
        body = result.chunk.text
        if len(body) > _MAX_CHUNK_CHARS:
            body = body[:_MAX_CHUNK_CHARS].rstrip() + "…"
        sections.append(f"### {label}\n{body}")
    return "\n\n".join(sections)


def _coerce_query(arguments: dict[str, Any]) -> str:
    query = arguments.get("query", "")
    if not isinstance(query, str):
        query = json.dumps(query)
    return query


def run_tool(arguments: dict[str, Any], *, root: Path | None = None) -> str:
    """Entry point `GroqRunner` calls to execute a `search_docs` tool call:
    parses the `query` argument and returns the formatted result text.
    Propagates `SearchDocsError` on an unreadable corpus."""
    results = search_docs(_coerce_query(arguments), root=root)
    return format_results(results)


def citation_labels(results: list[SearchResult]) -> list[str]:
    """Human-readable "<path> — <heading>" (or bare "<path>" for a
    headingless chunk) source labels for `results`, in the same rank order
    `search_docs` returned them -- the structured citation counterpart to
    `format_results`'s plain-text rendering fed to the model, kept in sync
    with it by sharing the same label-building logic rather than
    re-deriving it independently."""
    labels = []
    for result in results:
        label = result.chunk.path
        if result.chunk.heading:
            label = f"{label} — {result.chunk.heading}"
        labels.append(label)
    return labels


def citations_for_tool_call(arguments: dict[str, Any], *, root: Path | None = None) -> list[str]:
    """Citation-source labels for a `search_docs` tool call's arguments --
    used by `GroqRunner` to attach source attribution to the eventual
    assistant reply once the tool call succeeds (see
    docs/prd/chat-search-tools.md, "Tool-call round" and `ToolCitation` in
    models.py).

    Deliberately a separate call from `run_tool` (re-running the same cheap
    BM25 search -- see module docstring on why rebuilding the index per call
    is fine for this corpus) rather than changing `run_tool`'s return shape:
    `run_tool` stays a single plain-text-in, plain-text-out entry point so
    it keeps being easy to swap out as a single unit (e.g. in tests that
    monkeypatch it to simulate a tool failure) independent of citation
    plumbing.
    """
    results = search_docs(_coerce_query(arguments), root=root)
    return citation_labels(results)
