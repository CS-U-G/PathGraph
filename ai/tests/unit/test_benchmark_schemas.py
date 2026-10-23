"""Unit tests for benchmark persona and report schemas."""

import pytest
from pydantic import ValidationError

from pathgraph_ai.benchmarks.personas import ALL_BENCHMARK_PERSONAS
from pathgraph_ai.benchmarks.schemas import (
    AggregateBenchmarkReport,
    BenchmarkPersona,
    BenchmarkSkill,
    PersonaMetricResult,
)


def test_all_benchmark_personas_valid():
    """Confirms all 5 pre-configured computing personas pass Pydantic schema validation."""
    assert len(ALL_BENCHMARK_PERSONAS) == 5
    for p in ALL_BENCHMARK_PERSONAS:
        assert isinstance(p, BenchmarkPersona)
        assert len(p.current_skills) > 0
        assert len(p.expected_missing_essential) > 0
        assert len(p.known_irrelevant_skills) > 0
        assert len(p.provenance_and_limitations) > 20


def test_benchmark_skill_validation():
    """Confirms validation of skill proficiency bounds [0.0, 1.0]."""
    skill = BenchmarkSkill(skill_uri="uri:test", label="Test", proficiency=0.75)
    assert skill.proficiency == 0.75

    with pytest.raises(ValidationError):
        BenchmarkSkill(skill_uri="uri:test", label="Test", proficiency=1.5)

    with pytest.raises(ValidationError):
        BenchmarkSkill(skill_uri="uri:test", label="Test", proficiency=-0.1)


def test_aggregate_report_serialization():
    """Confirms AggregateBenchmarkReport serialization round-trip."""
    dummy_res = PersonaMetricResult(
        persona_id="test_persona",
        target_occupation_label="tester",
        precision_at_5=0.8,
        precision_at_10=0.7,
        recall_at_5=0.5,
        recall_at_10=0.8,
        ndcg_at_5=0.85,
        ndcg_at_10=0.78,
        map_score=0.75,
        essential_curriculum_coverage=1.0,
        prerequisite_violation_rate=0.0,
        prerequisite_violations_count=0,
        prerequisites_checked_count=2,
        curriculum_bloat_ratio=1.5,
        bloat_numerator=6,
        bloat_denominator=4,
        leakage_rate=0.0,
        leaked_skills=[],
        generated_path_length=6,
        verified_missing_essential_count=4,
    )

    report = AggregateBenchmarkReport(
        execution_timestamp="2026-10-22T00:00:00Z",
        environment_info={"os": "test"},
        configuration={"k": 5},
        personas_evaluated=[dummy_res],
        mean_precision_at_5=0.8,
        mean_precision_at_10=0.7,
        mean_recall_at_5=0.5,
        mean_recall_at_10=0.8,
        mean_ndcg_at_5=0.85,
        mean_ndcg_at_10=0.78,
        mean_map=0.75,
        mean_essential_coverage=1.0,
        mean_prerequisite_violation_rate=0.0,
        mean_curriculum_bloat_ratio=1.5,
        mean_leakage_rate=0.0,
        observed_weaknesses=["Minor bloat."],
        recommendations_for_improvement=["Constrain path length."],
    )

    json_str = report.model_dump_json()
    loaded = AggregateBenchmarkReport.model_validate_json(json_str)
    assert loaded.mean_precision_at_5 == 0.8
    assert len(loaded.personas_evaluated) == 1
