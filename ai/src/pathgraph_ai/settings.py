"""Typed configuration settings for PathGraph AI Recommendation Service."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directories
PACKAGE_DIR = Path(__file__).resolve().parent
AI_DIR = PACKAGE_DIR.parent.parent
REPO_ROOT = AI_DIR.parent


class Settings(BaseSettings):
    """Application and pipeline settings loaded from environment or defaults."""

    model_config = SettingsConfigDict(
        env_prefix="PATHGRAPH_AI_",
        env_file=".env",
        extra="ignore",
    )

    # Environment & Service
    environment: Literal["development", "staging", "production", "test"] = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    request_timeout_seconds: float = 5.0
    max_skills_per_request: int = 50

    # Data & Artifact Paths
    raw_data_dir: Path = AI_DIR / "data" / "raw" / "esco_v1.2.1_en_csv"
    interim_data_dir: Path = AI_DIR / "data" / "interim"
    processed_data_dir: Path = AI_DIR / "data" / "processed"
    artifacts_dir: Path = AI_DIR / "artifacts"
    seeds_dir: Path = REPO_ROOT / "knowledge" / "seeds"

    # Taxonomy & Rule Versions
    esco_version: str = "v1.2.1"
    filter_rule_version: str = "1.0.0"
    graph_version: str = "1.0.0"

    # Gap Analyser & Competency Target Thresholds (Configurable Project Conventions)
    # Target proficiencies on the contract's [0.0, 1.0] scale:
    essential_target_proficiency: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Target proficiency requirement for essential skills (0.75 corresponds to level 3 of 4).",
    )
    optional_target_proficiency: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Target proficiency requirement for optional skills (0.50 corresponds to level 2 of 4).",
    )

    # Requirement Weights (r)
    essential_requirement_weight: float = Field(
        default=1.0,
        description="Requirement weight multiplier r for essential skills.",
    )
    optional_requirement_weight: float = Field(
        default=0.4,
        description="Requirement weight multiplier r for optional skills.",
    )

    # Evidence Confidence (q) by source
    assumed_evidence_confidence: float = Field(
        default=0.3,
        description="Evidence confidence q for unassessed/assumed default proficiencies (e.g. validator 0.5 default).",
    )
    self_reported_evidence_confidence: float = Field(
        default=0.5,
        description="Evidence confidence q for self-reported proficiencies without external verification.",
    )
    project_evidence_confidence: float = Field(
        default=0.8,
        description="Evidence confidence q for project-backed competencies.",
    )
    assessed_evidence_confidence: float = Field(
        default=1.0,
        description="Evidence confidence q for verified assessment test scores.",
    )

    # Embedding & Resolver Settings
    use_embeddings: bool = Field(
        default=False,
        description="Enable neural embedding retrieval in the resolver cascade. If False, falls back to lexical.",
    )
    embeddings_lazy: bool = Field(
        default=True,
        description="Lazily load embedding model on first retrieval request rather than at boot.",
    )
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_revision: str = "e4ce9877abf3ed59ced85d2278e763d47d74e44a"
    embedding_dim: int = 384
    max_seq_length: int = 64
    inference_threads: int = 2
    embedding_accept_threshold: float = 0.82
    embedding_ambiguity_threshold: float = 0.65


settings = Settings()
