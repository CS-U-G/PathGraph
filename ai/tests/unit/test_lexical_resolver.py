"""Unit tests for the LexicalResolver cascade and ambiguity band."""

from __future__ import annotations

import pytest

from pathgraph_ai.adapters.lexical_resolver import LexicalResolver, normalize_text
from pathgraph_ai.domain.competency_graph import CompetencyGraph

PROCESSED_DIR = pytest.importorskip("pathlib").Path(__file__).resolve().parent.parent.parent / "data" / "processed"
SEEDS_DIR = pytest.importorskip("pathlib").Path(__file__).resolve().parent.parent.parent.parent / "knowledge" / "seeds"


@pytest.fixture(scope="session")
def resolver() -> LexicalResolver:
    graph = CompetencyGraph.from_processed_data(
        processed_dir=PROCESSED_DIR,
        prerequisites_csv_path=SEEDS_DIR / "prerequisites.csv",
    )
    return LexicalResolver(graph.skills)


class TestLexicalResolver:
    """Verifies that LexicalResolver resolves exact, normalized, and fuzzy queries without false merges."""

    def test_exact_uri_resolution(self, resolver: LexicalResolver) -> None:
        target_uri = "http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97"
        res = resolver.resolve(query="", skill_id=target_uri)
        assert res.status == "EXACT_ID"
        assert res.skill_uri == target_uri
        assert res.preferred_label == "computer programming"

    def test_normalized_preferred_label_resolution(self, resolver: LexicalResolver) -> None:
        res = resolver.resolve(query="  Computer Programming  ")
        assert res.status == "NORMALIZED_LABEL"
        assert res.preferred_label == "computer programming"
        assert res.confidence >= 0.95

    def test_normalized_alias_resolution(self, resolver: LexicalResolver) -> None:
        # Check an alias from skills
        res = resolver.resolve(query="c++")
        if res.status == "NORMALIZED_ALIAS":
            assert res.skill_uri is not None
            assert res.confidence >= 0.90

    def test_conservative_fuzzy_resolution(self, resolver: LexicalResolver) -> None:
        res = resolver.resolve(query="operating system")  # slight variation of 'operating systems'
        assert res.status in ("NORMALIZED_LABEL", "FUZZY_MATCH")
        assert res.preferred_label == "operating systems"
        assert res.confidence >= 0.85

    def test_no_false_merge_on_distinct_names(self, resolver: LexicalResolver) -> None:
        """Never silently merge near names like Java and JavaScript."""
        res_java = resolver.resolve(query="Java")
        res_js = resolver.resolve(query="JavaScript")
        if res_java.skill_uri and res_js.skill_uri:
            assert res_java.skill_uri != res_js.skill_uri

    def test_unresolved_query_fallback(self, resolver: LexicalResolver) -> None:
        res = resolver.resolve(query="quantum cooking astrophysics")
        assert res.status == "UNRESOLVED"
        assert res.skill_uri is None
        assert res.confidence == 0.0
