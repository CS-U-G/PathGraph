"""Pydantic schemas for recommendation quality benchmarks."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class BenchmarkSkill(BaseModel):
    """Specification of a learner's existing skill and proficiency."""

    skill_uri: str = Field(..., description="Canonical ESCO skill URI.")
    label: str = Field(..., description="Human-readable skill label.")
    proficiency: float = Field(default=1.0, ge=0.0, le=1.0, description="Proficiency level in [0.0, 1.0].")


class BenchmarkPersona(BaseModel):
    """A realistic computing career-transition persona with expert ground-truth labels."""

    persona_id: str = Field(..., description="Unique alphanumeric identifier for the persona.")
    title: str = Field(..., description="Descriptive transition title.")
    current_role: str = Field(..., description="Learner's current occupational role.")
    target_role: str = Field(..., description="Target occupational role.")
    target_occupation_uri: str = Field(..., description="Target ESCO occupation concept URI.")
    target_occupation_label: str = Field(..., description="Preferred label of the target occupation.")
    current_skills: list[BenchmarkSkill] = Field(..., description="Skills already acquired by learner.")
    expected_missing_essential: list[str] = Field(
        ...,
        description="URIs of essential skills that the learner is verified to lack and must acquire.",
    )
    expected_optional_skills: list[str] = Field(
        default_factory=list,
        description="URIs of optional/elective skills relevant to the target role.",
    )
    expert_relevant_target_skills: list[str] = Field(
        ...,
        description="Union of all domain-relevant skills (essential + optional) expected for target competence.",
    )
    known_irrelevant_skills: list[str] = Field(
        default_factory=list,
        description="Cross-domain skills injected to test recommendation leakage (negative controls).",
    )
    explicit_prerequisites: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Curated prerequisite dependencies: (prerequisite_uri, dependent_uri).",
    )
    provenance_and_limitations: str = Field(
        ...,
        description="Documentation of label sources, curricular rationale, and known limitations.",
    )


class PersonaMetricResult(BaseModel):
    """Evaluation metrics computed for a single benchmark persona."""

    persona_id: str
    target_occupation_label: str
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    map_score: float
    essential_curriculum_coverage: float
    prerequisite_violation_rate: float
    prerequisite_violations_count: int
    prerequisites_checked_count: int
    curriculum_bloat_ratio: float
    bloat_numerator: int
    bloat_denominator: int
    leakage_rate: float
    leaked_skills: list[str]
    generated_path_length: int
    verified_missing_essential_count: int


class AggregateBenchmarkReport(BaseModel):
    """Comprehensive benchmark execution report across all personas."""

    benchmark_version: str = "1.0.0"
    execution_timestamp: str
    environment_info: dict[str, Any]
    configuration: dict[str, Any]
    personas_evaluated: list[PersonaMetricResult]
    mean_precision_at_5: float
    mean_precision_at_10: float
    mean_recall_at_5: float
    mean_recall_at_10: float
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    mean_map: float
    mean_essential_coverage: float
    mean_prerequisite_violation_rate: float
    mean_curriculum_bloat_ratio: float
    mean_leakage_rate: float
    observed_weaknesses: list[str]
    recommendations_for_improvement: list[str]
