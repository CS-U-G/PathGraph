"""Tests for Competency Graph validation and integrity checks (V1-V8)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx
import pytest

from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.domain.prerequisites import (
    PrerequisiteValidationError,
    load_prerequisites_csv,
)
from pathgraph_ai.domain.validation import GraphValidator
from pathgraph_ai.settings import settings

AI_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = AI_DIR.parent
PROCESSED_DIR = AI_DIR / "data" / "processed"
SEEDS_DIR = REPO_ROOT / "knowledge" / "seeds"


@pytest.fixture(scope="session")
def active_graph() -> CompetencyGraph:
    """Loads the active 4-layer competency graph."""
    prereq_csv = SEEDS_DIR / "prerequisites.csv"
    assert prereq_csv.exists(), f"Prerequisite seed file missing at {prereq_csv}"
    return CompetencyGraph.from_processed_data(
        processed_dir=PROCESSED_DIR,
        prerequisites_csv_path=prereq_csv,
    )


class TestGraphValidationSuite:
    """Verifies that the canonical graph passes V1-V8 and detects synthetic violations."""

    def test_canonical_graph_passes_all_checks(self, active_graph: CompetencyGraph) -> None:
        validator = GraphValidator(active_graph)
        report = validator.run_all()
        assert report.overall_pass is True, f"Validation failed: {report.summary}"
        assert report.passed_checks == 8
        assert report.failed_checks == 0

    def test_v1_dangling_edges_detection(self, active_graph: CompetencyGraph) -> None:
        # Create an isolated graph and inject a dangling edge
        graph_copy = CompetencyGraph.from_processed_data(
            processed_dir=PROCESSED_DIR,
            prerequisites_csv_path=SEEDS_DIR / "prerequisites.csv",
        )
        fake_skill_uri = "http://data.europa.eu/esco/skill/00000000-0000-0000-0000-000000000000"
        real_occ_uri = next(iter(graph_copy.occupations.keys()))
        graph_copy.requirement_graph.add_edge(real_occ_uri, fake_skill_uri)

        validator = GraphValidator(graph_copy)
        res = validator.validate_v1_no_dangling_edges()
        assert res.passed is False
        assert res.violations_count > 0
        assert any(d["target"] == fake_skill_uri for d in res.details["dangling_requirement_edges"])

    def test_v2_cycle_rejection_and_scc_reporting(self, active_graph: CompetencyGraph) -> None:
        # Clone active graph and inject a cycle in prerequisite DAG
        graph_copy = CompetencyGraph.from_processed_data(
            processed_dir=PROCESSED_DIR,
            prerequisites_csv_path=SEEDS_DIR / "prerequisites.csv",
        )
        # Pick two active skills
        edges = list(graph_copy.prerequisite_graph.edges())
        assert len(edges) >= 2
        u, v = edges[0]
        # Adding edge from descendant back to ancestor creates a cycle
        graph_copy.prerequisite_graph.add_edge(v, u, relation_type="strict", provenance="inferred", confidence=0.8)

        is_dag, sccs = graph_copy.check_prerequisite_acyclicity()
        assert is_dag is False
        assert len(sccs) >= 1
        # Check SCC contains u and v
        assert {u, v}.issubset(set(sccs[0]))

        validator = GraphValidator(graph_copy)
        res = validator.validate_v2_prerequisite_dag()
        assert res.passed is False
        assert res.violations_count >= 1
        assert len(res.details["strongly_connected_components"]) >= 1

    def test_v3_minimum_essential_skills_audit(self, active_graph: CompetencyGraph) -> None:
        validator = GraphValidator(active_graph)
        res = validator.validate_v3_min_essential_skills()
        assert res.passed is True
        assert res.details["total_occupations_audited"] == 96
        assert len(res.details["failing_occupations"]) == 0

    def test_v6_reason_codes_exact_parity(self, active_graph: CompetencyGraph) -> None:
        validator = GraphValidator(active_graph)
        res = validator.validate_v6_reason_codes_parity()
        assert res.passed is True
        assert res.violations_count == 0

    def test_v7_missing_provenance_rejection(self, active_graph: CompetencyGraph) -> None:
        graph_copy = CompetencyGraph.from_processed_data(
            processed_dir=PROCESSED_DIR,
            prerequisites_csv_path=SEEDS_DIR / "prerequisites.csv",
        )
        # Inject an edge with missing provenance
        edges = list(graph_copy.prerequisite_graph.edges())
        u, v = edges[0]
        graph_copy.prerequisite_graph[u][v]["provenance"] = ""

        validator = GraphValidator(graph_copy)
        res = validator.validate_v7_provenance_and_confidence()
        assert res.passed is False
        assert res.violations_count == 1

    def test_v8_contradictory_edges_detection(self, active_graph: CompetencyGraph) -> None:
        graph_copy = CompetencyGraph.from_processed_data(
            processed_dir=PROCESSED_DIR,
            prerequisites_csv_path=SEEDS_DIR / "prerequisites.csv",
        )
        edges = list(graph_copy.prerequisite_graph.edges())
        u, v = edges[0]
        # Inject contradictory reverse edge
        graph_copy.prerequisite_graph.add_edge(v, u, relation_type="strict", provenance="inferred", confidence=0.8)

        validator = GraphValidator(graph_copy)
        res = validator.validate_v8_duplicates_and_contradictions()
        assert res.passed is False
        assert res.violations_count >= 1


class TestPrerequisiteLoaderValidation:
    """Verifies that load_prerequisites_csv enforces domain constraints."""

    def test_reject_unknown_uris(self, tmp_path: Path, active_graph: CompetencyGraph) -> None:
        fake_csv = tmp_path / "bad_uri.csv"
        valid_uris = set(active_graph.skills.keys())
        real_uri = next(iter(valid_uris))
        fake_uri = "http://data.europa.eu/esco/skill/unknown-uuid"

        with open(fake_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prerequisite_skill_uri", "target_skill_uri", "relation_type", "provenance", "confidence", "evidence_source"])
            writer.writerow([fake_uri, real_uri, "strict", "inferred", "0.85", "CS_CURRICULUM_PEDAGOGY"])

        with pytest.raises(PrerequisiteValidationError, match="not in skill vocabulary"):
            load_prerequisites_csv(fake_csv, valid_uris)

    def test_reject_missing_provenance(self, tmp_path: Path, active_graph: CompetencyGraph) -> None:
        fake_csv = tmp_path / "no_prov.csv"
        valid_uris = set(active_graph.skills.keys())
        u1, u2 = list(valid_uris)[:2]

        with open(fake_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prerequisite_skill_uri", "target_skill_uri", "relation_type", "provenance", "confidence", "evidence_source"])
            writer.writerow([u1, u2, "strict", "", "0.85", "CS_CURRICULUM_PEDAGOGY"])

        with pytest.raises(PrerequisiteValidationError, match="Missing provenance"):
            load_prerequisites_csv(fake_csv, valid_uris)

    def test_reject_contradictory_mutual_edges(self, tmp_path: Path, active_graph: CompetencyGraph) -> None:
        fake_csv = tmp_path / "contradictory.csv"
        valid_uris = set(active_graph.skills.keys())
        u1, u2 = list(valid_uris)[:2]

        with open(fake_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prerequisite_skill_uri", "target_skill_uri", "relation_type", "provenance", "confidence", "evidence_source"])
            writer.writerow([u1, u2, "strict", "inferred", "0.85", "CS_CURRICULUM_PEDAGOGY"])
            writer.writerow([u2, u1, "strict", "inferred", "0.85", "CS_CURRICULUM_PEDAGOGY"])

        with pytest.raises(PrerequisiteValidationError, match="Contradictory mutual prerequisite"):
            load_prerequisites_csv(fake_csv, valid_uris)
