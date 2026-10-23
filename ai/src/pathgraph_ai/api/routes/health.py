"""Health check and service diagnostic endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status

from pathgraph_ai.settings import settings

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns service availability, version, and loaded ESCO taxonomy metrics.",
)
async def health_check(request: Request) -> dict[str, Any]:
    """Inspects service status and returns runtime graph metadata."""
    graph = getattr(request.app.state, "graph", None)
    occupations_count = len(graph.occupations) if graph is not None else 0
    skills_count = len(graph.skills) if graph is not None else 0

    return {
        "status": "healthy",
        "service": "PathGraph AI Recommendation Service",
        "version": "0.1.0",
        "esco_version": settings.esco_version,
        "graph_version": settings.graph_version,
        "occupations_count": occupations_count,
        "skills_count": skills_count,
    }
