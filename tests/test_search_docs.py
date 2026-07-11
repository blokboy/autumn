"""Behavioral tests for `search_docs`'s BM25-style retrieval over
docs/**/*.md, README.md, AGENTS.md."""

import pytest

from search_docs import SearchDocsError, format_results, search_docs


def _write_corpus(root):
    (root / "docs").mkdir()
    (root / "docs" / "prd").mkdir()
    (root / "README.md").write_text(
        "# autumn\n"
        "\n"
        "A CLI for GEPA runs.\n"
        "\n"
        "## Usage\n"
        "\n"
        "```bash\n"
        "autumn run <script.py> --dry-run\n"
        "```\n"
        "Launches the live dashboard. Without `--dry-run`, the script is run for "
        "real. `--dry-run` instead replays a scripted, dependency-free sequence "
        "of optimization events so you can see the sidebar update live without a "
        "real GEPA run.\n"
        "\n"
        "## Install\n"
        "\n"
        "```bash\n"
        "pip install -e .\n"
        "```\n"
    )
    (root / "AGENTS.md").write_text(
        "## Agent skills\n"
        "\n"
        "### Issue tracker\n"
        "\n"
        "Issues are tracked in GitHub Issues via the `gh` CLI.\n"
    )
    (root / "docs" / "prd" / "example.md").write_text(
        "# PRD: Example feature\n"
        "\n"
        "## Problem\n"
        "\n"
        "This section is unrelated to CLI flags -- it's about a completely "
        "different problem involving Tavily search credentials.\n"
    )


def test_search_docs_finds_the_grounded_section_for_a_realistic_query(tmp_path):
    _write_corpus(tmp_path)

    results = search_docs("what does --dry-run do", root=tmp_path)

    assert results, "expected at least one match"
    top = results[0]
    assert top.chunk.path == "README.md"
    assert top.chunk.heading == "Usage"
    assert "--dry-run" in top.chunk.text


def test_search_docs_ranks_more_relevant_chunk_first(tmp_path):
    _write_corpus(tmp_path)

    results = search_docs("issue tracker github", root=tmp_path)

    assert results
    # "### Issue tracker" is a sub-heading nested under "## Agent skills" --
    # chunking only splits on top-level (#/##) headings, so it stays folded
    # into the enclosing chunk rather than becoming its own.
    assert results[0].chunk.path == "AGENTS.md"
    assert results[0].chunk.heading == "Agent skills"
    assert "Issue tracker" in results[0].chunk.text


def test_search_docs_returns_empty_list_for_unmatched_query(tmp_path):
    _write_corpus(tmp_path)

    results = search_docs("quantum entanglement submarine", root=tmp_path)

    assert results == []


def test_search_docs_returns_empty_list_for_empty_corpus(tmp_path):
    results = search_docs("anything", root=tmp_path)

    assert results == []


def test_search_docs_respects_top_k(tmp_path):
    _write_corpus(tmp_path)

    results = search_docs("the", root=tmp_path, top_k=1)

    assert len(results) <= 1


def test_search_docs_raises_on_unreadable_file(tmp_path):
    (tmp_path / "docs").mkdir()
    broken = tmp_path / "docs" / "broken.md"
    broken.symlink_to(tmp_path / "docs" / "does-not-exist.md")

    with pytest.raises(SearchDocsError):
        search_docs("anything", root=tmp_path)


def test_format_results_includes_path_and_heading_labels(tmp_path):
    _write_corpus(tmp_path)

    results = search_docs("dry-run", root=tmp_path)
    text = format_results(results)

    assert "README.md" in text
    assert "Usage" in text
    assert "--dry-run" in text


def test_format_results_reports_no_matches_without_raising(tmp_path):
    _write_corpus(tmp_path)

    results = search_docs("quantum entanglement submarine", root=tmp_path)

    assert format_results(results) == "No matching documentation found."
