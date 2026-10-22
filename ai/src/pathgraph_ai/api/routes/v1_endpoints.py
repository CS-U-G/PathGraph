"""Modular API v1 router exposing skill gap, learning path, and health endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from pathgraph_ai.api.schemas import (
    HealthResponse,
    LearningPathRequest,
    LearningPathResponse,
    LearningPathStepResponse,
    SkillExtractionRequest,
    SkillExtractionResponse,
    SkillGapRequest,
    SkillGapResponse,
    SkillMatchDetail,
)
from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.recommenders.path_generator import LearningPathGenerator, PrerequisiteGraph
from pathgraph_ai.recommenders.skill_gap import SkillGapAnalyzer
from pathgraph_ai.settings import settings

router = APIRouter(prefix="/api/v1", tags=["API v1"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API v1 Health Check",
)
async def v1_health_check(request: Request) -> HealthResponse:
    """Returns the operational health status and verifies ESCO dataset availability."""
    graph: CompetencyGraph | None = getattr(request.app.state, "graph", None)
    esco_available = graph is not None and len(graph.occupations) > 0 and len(graph.skills) > 0
    return HealthResponse(
        status="healthy",
        app_name="PathGraph AI Service",
        version="1.0.0",
        esco_data_available=esco_available,
    )


@router.post(
    "/skills/extract",
    response_model=SkillExtractionResponse,
    summary="Extract skills from text (In Development)",
    tags=["Skills"],
)
async def extract_skills(payload: SkillExtractionRequest) -> SkillExtractionResponse:
    """Endpoint stub for extracting skills from CVs or job descriptions."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Skill extraction engine is currently in development.",
    )


@router.post(
    "/skills/gap-analysis",
    response_model=SkillGapResponse,
    summary="Analyze skill gap for target occupation",
    tags=["Skills"],
)
async def analyze_skill_gap(
    payload: SkillGapRequest,
    request: Request,
) -> SkillGapResponse:
    """Analyzes exact and semantic gap between user skills and target occupation requirements."""
    graph: CompetencyGraph = request.app.state.graph
    occ_uri = payload.target_occupation_uri.strip()

    if occ_uri not in graph.occupations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target occupation '{occ_uri}' not found in ESCO knowledge graph.",
        )

    occ_info = graph.get_occupation(occ_uri) or {}
    occ_label = occ_info.get("preferredLabel", occ_uri)

    reqs = graph.get_occupation_requirements(occ_uri)
    required_skills = [
        {
            "skillUri": r.skill_uri,
            "skillLabel": (graph.get_skill(r.skill_uri) or {}).get("preferredLabel", r.skill_uri),
            "relationType": r.relation_type,
            "skillType": (graph.get_skill(r.skill_uri) or {}).get("skillType"),
        }
        for r in reqs
    ]

    # Initialize analyzer
    analyzer = SkillGapAnalyzer(
        artifact_path=settings.skills_embeddings_path if settings.skills_embeddings_path.exists() else None
    )
    user_skill_set = set(payload.user_skill_uris)

    report = analyzer.analyze_gap_exact(
        occupation_uri=occ_uri,
        occupation_label=occ_label,
        required_skills=required_skills,
        user_skill_uris=user_skill_set,
    )

    return SkillGapResponse(
        target_occupation_uri=report.occupation_uri,
        target_occupation_label=report.occupation_label,
        essential_match_percentage=round(report.essential_match_percentage, 2),
        overall_match_percentage=round(report.overall_match_percentage, 2),
        matched_essential_skills=[
            SkillMatchDetail(
                skill_uri=m.skill_uri,
                skill_label=m.skill_label,
                relation_type=m.relation_type,
                similarity_score=m.similarity_score,
            )
            for m in report.matched_essential
        ],
        missing_essential_skills=[
            SkillMatchDetail(
                skill_uri=m.skill_uri,
                skill_label=m.skill_label,
                relation_type=m.relation_type,
                similarity_score=0.0,
            )
            for m in report.missing_essential
        ],
        matched_optional_skills=[
            SkillMatchDetail(
                skill_uri=m.skill_uri,
                skill_label=m.skill_label,
                relation_type=m.relation_type,
                similarity_score=m.similarity_score,
            )
            for m in report.matched_optional
        ],
        missing_optional_skills=[
            SkillMatchDetail(
                skill_uri=m.skill_uri,
                skill_label=m.skill_label,
                relation_type=m.relation_type,
                similarity_score=0.0,
            )
            for m in report.missing_optional
        ],
    )


@router.post(
    "/learning-path/generate",
    response_model=LearningPathResponse,
    summary="Generate personalized learning path",
    tags=["Learning Path"],
)
async def generate_learning_path(
    payload: LearningPathRequest,
    request: Request,
) -> LearningPathResponse:
    """Sequences missing skills into a pedagogical learning path using Kahn's topological sort."""
    graph: CompetencyGraph = request.app.state.graph
    occ_uri = payload.target_occupation_uri.strip()

    if occ_uri not in graph.occupations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target occupation '{occ_uri}' not found in ESCO knowledge graph.",
        )

    occ_info = graph.get_occupation(occ_uri) or {}
    occ_label = occ_info.get("preferredLabel", occ_uri)

    # Build missing skills list with relation types
    reqs_by_skill = {r.skill_uri: r.relation_type for r in graph.get_occupation_requirements(occ_uri)}
    missing_skills = [
        {
            "skillUri": s_uri,
            "skillLabel": (graph.get_skill(s_uri) or {}).get("preferredLabel", s_uri),
            "relationType": reqs_by_skill.get(s_uri, "optional"),
        }
        for s_uri in payload.missing_skill_uris
    ]

    # Build DAG from competency graph's prerequisite edges
    prereq_graph = PrerequisiteGraph()
    for s_uri in payload.missing_skill_uris:
        direct_prereqs = graph.get_prerequisites(s_uri, transitive=False)
        for p in direct_prereqs:
            prereq_graph.add_prerequisite(p, s_uri)

    generator = LearningPathGenerator(prerequisite_graph=prereq_graph)
    path = generator.generate_path(
        target_occupation_uri=occ_uri,
        target_occupation_label=occ_label,
        missing_skills=missing_skills,
    )

    steps_response = [
        LearningPathStepResponse(
            step_number=step.step_number,
            skill_uri=step.skill_uri,
            skill_label=step.skill_label,
            relation_type=step.relation_type,
            prerequisites=step.prerequisites,
        )
        for step in path.steps
    ]

    return LearningPathResponse(
        target_occupation_uri=path.target_occupation_uri,
        target_occupation_label=path.target_occupation_label,
        total_steps=path.total_steps,
        steps=steps_response,
        has_prerequisite_cycles=path.has_prerequisite_cycles,
    )
