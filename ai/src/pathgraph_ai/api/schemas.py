"""Pydantic models and schemas representing the PathGraph Recommendation API Contract.

These models directly implement the OpenAPI 3.1 contract defined in `contracts/ai.openapi.yaml`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReasonCode(str, Enum):
    """Enumerated machine-readable explanation tags describing why a skill is recommended."""

    REQUIRED_BY_TARGET = "REQUIRED_BY_TARGET"
    MASTERY_GAP = "MASTERY_GAP"
    PREREQ_SATISFIED = "PREREQ_SATISFIED"
    HIGH_RELEVANCE = "HIGH_RELEVANCE"
    HIGH_DEMAND = "HIGH_DEMAND"
    TRANSVERSAL_FOUNDATION = "TRANSVERSAL_FOUNDATION"
    STEPPING_STONE = "STEPPING_STONE"


class TargetType(str, Enum):
    """Enumeration of recommendation target goal types."""

    OCCUPATION = "occupation"
    CUSTOM_GOAL = "custom_goal"


class LearnerSkill(BaseModel):
    """A skill currently possessed or claimed by the learner, either pre-resolved or raw text."""

    model_config = ConfigDict(extra="forbid")

    skill_id: Optional[str] = Field(
        default=None,
        description=(
            "Canonical ESCO skill URI or identifier if resolved; "
            "null if submitting raw label for AI semantic resolution."
        ),
        json_schema_extra={
            "example": "http://data.europa.eu/esco/skill/0007833a-6ff5-4089-a292-6284693b4dd6"
        },
    )
    label: str = Field(
        ...,
        min_length=1,
        description=(
            "Human-readable skill label as entered by learner or resolved by client; "
            "used as fallback for semantic mapping when skill_id is null."
        ),
        json_schema_extra={"example": "Python (programming language)"},
    )
    proficiency: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Learner proficiency from 0.0 to 1.0. Optional; if omitted or null, "
            "defaults to 0.5 (representing baseline working competence)."
        ),
        json_schema_extra={"example": 0.7},
    )

    @field_validator("proficiency", mode="before")
    @classmethod
    def default_proficiency_if_none(cls, v: Any) -> float:
        """Explicit decision: omitted or null proficiency defaults to 0.5 baseline."""
        if v is None:
            return 0.5
        return float(v)


class RecommendationTarget(BaseModel):
    """The target occupational goal for which the recommendation is generated."""

    model_config = ConfigDict(extra="forbid")

    occupation_id: str = Field(
        ...,
        min_length=1,
        description="Canonical ESCO occupation URI or identifier targeted by the learner.",
        json_schema_extra={"example": "http://data.europa.eu/esco/occupation/2512.1"},
    )
    target_type: TargetType = Field(
        default=TargetType.OCCUPATION,
        description="Type of target goal. Defaults to 'occupation'. Reserved for future custom goal extensions.",
        json_schema_extra={"example": "occupation"},
    )
    custom_goal: Optional[str] = Field(
        default=None,
        description=(
            "Optional free-text description of custom learning goal or specialization; "
            "reserved for future semantic matching extensions."
        ),
        json_schema_extra={"example": None},
    )


class LearnerProfile(BaseModel):
    """Learner profile containing current skills and target career goal."""

    model_config = ConfigDict(extra="forbid")

    skills: list[LearnerSkill] = Field(
        default_factory=list,
        description="List of learner claimed skills (pre-resolved ESCO skills or raw user-entered labels).",
    )
    target: RecommendationTarget = Field(
        ...,
        description="The target occupational goal for which the recommendation is generated.",
    )


class RecommendationRequest(BaseModel):
    """Payload submitted to request a personalized learning path recommendation."""

    model_config = ConfigDict(extra="forbid")

    learner: LearnerProfile = Field(
        ...,
        description="Learner profile containing current skills and target career goal.",
    )


class PathSkillRecommendation(BaseModel):
    """An individual recommended skill milestone within the ordered learning path."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(
        ...,
        description="Canonical ESCO skill URI or persistent identifier.",
        json_schema_extra={
            "example": "http://data.europa.eu/esco/skill/0007833a-6ff5-4089-a292-6284693b4dd6"
        },
    )
    label: str = Field(
        ...,
        description="Canonical human-readable label of the skill.",
        json_schema_extra={"example": "Python (programming language)"},
    )
    priority: int = Field(
        ...,
        ge=1,
        description="1-indexed ordering position in the recommended path sequence.",
        json_schema_extra={"example": 1},
    )
    gap: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Competency gap between target requirement and current learner proficiency "
            "(required level minus learner mastery, bounded in [0.0, 1.0])."
        ),
        json_schema_extra={"example": 0.3},
    )
    prerequisite_satisfied: bool = Field(
        ...,
        description=(
            "True if all prerequisite skills in the curriculum graph are already "
            "possessed by learner or scheduled earlier in this path."
        ),
        json_schema_extra={"example": True},
    )
    relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized semantic similarity and contextual importance score "
            "relative to target occupation and learner background."
        ),
        json_schema_extra={"example": 0.95},
    )
    demand: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized labor market demand index and cross-occupational transferability score."
        ),
        json_schema_extra={"example": 0.88},
    )
    estimated_cost: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional estimated learning effort/cost in normalized hours or complexity units.",
        json_schema_extra={"example": 30.0},
    )
    reason: str = Field(
        ...,
        description="Human-readable pedagogical explanation of why this skill was selected and sequenced at this step.",
        json_schema_extra={
            "example": (
                "Essential core skill for target occupation with an identified mastery "
                "gap of 0.30; all prerequisites are satisfied."
            )
        },
    )
    reason_codes: list[ReasonCode] = Field(
        ...,
        min_length=1,
        description="Array of enumerated machine-readable tags for frontend badge/icon rendering.",
    )


class RecommendationMetadata(BaseModel):
    """Contextual metadata about the generated recommendation run."""

    model_config = ConfigDict(extra="forbid")

    target_occupation_id: str = Field(
        ...,
        description="Canonical target occupation identifier that grounded this recommendation calculation.",
        json_schema_extra={"example": "http://data.europa.eu/esco/occupation/2512.1"},
    )
    unresolved_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Input skill labels that the semantic resolver could not map to canonical ESCO skills, "
            "enabling client UI to inform the user."
        ),
        json_schema_extra={"example": ["quantum cooking"]},
    )
    generated_at: datetime = Field(
        ...,
        description="ISO-8601 UTC timestamp when this recommendation plan was generated.",
        json_schema_extra={"example": "2026-09-27T08:00:00Z"},
    )


class RecommendationResponse(BaseModel):
    """Complete response containing ordered learning path recommendations and execution metadata."""

    model_config = ConfigDict(extra="forbid")

    learning_path: list[PathSkillRecommendation] = Field(
        ...,
        description="Ordered list of skill learning recommendations ranked by prerequisite topological order and multi-criteria scoring.",
    )
    meta: RecommendationMetadata = Field(
        ...,
        description="Contextual metadata about the generated recommendation run.",
    )


class ErrorDetail(BaseModel):
    """Specific field-level detail for request validation or semantic resolution errors."""

    model_config = ConfigDict(extra="forbid")

    field: Optional[str] = Field(
        default=None,
        description="Name of the invalid field or parameter, if applicable.",
        json_schema_extra={"example": "learner.skills[0].proficiency"},
    )
    issue: str = Field(
        ...,
        description="Specific description of the validation failure or semantic error.",
        json_schema_extra={"example": "Value must be between 0.0 and 1.0."},
    )


class ErrorResponse(BaseModel):
    """Standardized error response shape for all 4xx/5xx API responses."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description="Machine-readable uppercase snake_case error code.",
        json_schema_extra={"example": "INVALID_TARGET_OCCUPATION"},
    )
    message: str = Field(
        ...,
        description="Human-readable error message explaining the failure.",
        json_schema_extra={"example": "Target occupation does not exist in ESCO taxonomy."},
    )
    details: Optional[list[ErrorDetail]] = Field(
        default=None,
        description="Optional list of granular field-level error details.",
    )
    timestamp: datetime = Field(
        ...,
        description="ISO-8601 UTC timestamp when the error occurred.",
        json_schema_extra={"example": "2026-09-27T08:00:00Z"},
    )


# --- Extended AI Service Schemas (Graph and Recommendation V1 API) ---


class HealthResponse(BaseModel):
    """Service health and diagnostic status."""

    status: str = Field(..., json_schema_extra={"example": "healthy"})
    app_name: str = Field(..., json_schema_extra={"example": "PathGraph AI Service"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    esco_data_available: bool = Field(default=False)


class SkillItem(BaseModel):
    """Skill representation for skill extraction or catalog listings."""

    uri: Optional[str] = Field(default=None, description="ESCO concept URI")
    label: str = Field(..., description="Skill preferred label or title")
    skill_type: Optional[str] = Field(default=None, description="knowledge or skill/competence")
    relation_type: Optional[str] = Field(default=None, description="essential or optional")


class SkillExtractionRequest(BaseModel):
    """Request payload for extracting skills from free-form text."""

    text: str = Field(..., min_length=1, description="Raw CV or job description text")


class SkillExtractionResponse(BaseModel):
    """Response payload containing extracted skills."""

    extracted_skills: list[SkillItem] = Field(default_factory=list)
    total_extracted: int = 0


class SkillGapRequest(BaseModel):
    """Request payload for direct skill gap analysis."""

    target_occupation_uri: str = Field(..., description="ESCO URI of target occupation")
    user_skill_uris: list[str] = Field(default_factory=list, description="List of ESCO skill URIs user possesses")


class SkillMatchDetail(BaseModel):
    """Detailed record of a matched or missing skill."""

    skill_uri: str
    skill_label: str
    relation_type: str
    similarity_score: float = 1.0


class SkillGapResponse(BaseModel):
    """Response payload detailing exact and semantic skill gap breakdown."""

    target_occupation_uri: str
    target_occupation_label: str
    essential_match_percentage: float
    overall_match_percentage: float
    matched_essential_skills: list[SkillMatchDetail] = Field(default_factory=list)
    missing_essential_skills: list[SkillMatchDetail] = Field(default_factory=list)
    matched_optional_skills: list[SkillMatchDetail] = Field(default_factory=list)
    missing_optional_skills: list[SkillMatchDetail] = Field(default_factory=list)


class LearningPathRequest(BaseModel):
    """Request payload for sequencing missing skills into a learning path."""

    target_occupation_uri: str = Field(..., description="ESCO URI of target occupation")
    missing_skill_uris: list[str] = Field(..., min_length=1, description="Skill URIs requiring sequencing")


class LearningPathStepResponse(BaseModel):
    """An individual sequenced milestone step in a learning path."""

    step_number: int
    skill_uri: str
    skill_label: str
    relation_type: str
    prerequisites: list[str] = Field(default_factory=list)


class LearningPathResponse(BaseModel):
    """Response payload containing sequenced learning path."""

    target_occupation_uri: str
    target_occupation_label: str
    total_steps: int
    steps: list[LearningPathStepResponse] = Field(default_factory=list)
    has_prerequisite_cycles: bool = False

