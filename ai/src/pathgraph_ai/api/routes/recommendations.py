"""Learning path recommendation route implementing POST /v1/recommendations."""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from pathgraph_ai.adapters.lexical_resolver import LexicalResolver
from pathgraph_ai.api.schemas import (
    LearnerProfile,
    LearnerSkill,
    RecommendationRequest,
    RecommendationResponse,
)
from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.recommenders.path_engine import PathEngine

router = APIRouter(tags=["Recommendations"])


@router.post(
    "/v1/recommendations",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate personalized learning path recommendations",
    description=(
        "Analyzes a learner profile against a target canonical ESCO occupation, "
        "resolves free-text skill labels, evaluates prerequisite graph constraints, "
        "computes skill gaps, and returns a prioritized, multi-criteria-ranked learning path."
    ),
)
async def generate_recommendations(
    payload: RecommendationRequest,
    request: Request,
) -> RecommendationResponse:
    """Processes learning path recommendation request against canonical ESCO graph."""
    graph: CompetencyGraph = request.app.state.graph
    resolver: LexicalResolver = request.app.state.resolver
    path_engine: PathEngine = request.app.state.path_engine

    unresolved_skills: list[str] = []
    resolved_skills: list[LearnerSkill] = []

    for s in payload.learner.skills:
        skill_uri = s.skill_id.strip() if s.skill_id else None
        clean_label = s.label.strip() if s.label else ""

        # Case 1: skill_id is already a verified canonical ESCO URI
        if skill_uri and skill_uri in graph.skills:
            resolved_skills.append(
                LearnerSkill(
                    skill_id=skill_uri,
                    label=s.label,
                    proficiency=s.proficiency,
                )
            )
            continue

        # Case 2: Resolve via LexicalResolver cascade
        res = resolver.resolve(query=clean_label, skill_id=skill_uri)
        if res.skill_uri and res.skill_uri in graph.skills:
            resolved_skills.append(
                LearnerSkill(
                    skill_id=res.skill_uri,
                    label=res.preferred_label or s.label,
                    proficiency=s.proficiency,
                )
            )
        else:
            # Unresolved skill: record in meta.unresolved_skills
            if clean_label and clean_label not in unresolved_skills:
                unresolved_skills.append(clean_label)

    # Form resolved recommendation request for the path engine
    resolved_request = RecommendationRequest(
        learner=LearnerProfile(
            skills=resolved_skills,
            target=payload.learner.target,
        )
    )

    return path_engine.recommend(
        request=resolved_request,
        unresolved_skills=unresolved_skills,
    )
