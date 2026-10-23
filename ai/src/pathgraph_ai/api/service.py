"""FastAPI application factory, lifespan management, and global error handling."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pathgraph_ai.adapters.lexical_resolver import LexicalResolver
from pathgraph_ai.api.routes.health import router as health_router
from pathgraph_ai.api.routes.recommendations import router as recommendations_router
from pathgraph_ai.api.routes.v1_endpoints import router as v1_router
from pathgraph_ai.api.schemas import ErrorDetail, ErrorResponse
from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.recommenders.path_engine import (
    InvalidTargetOccupationError,
    PathEngine,
    PrerequisiteCycleError,
    UnsupportedTargetTypeError,
)
from pathgraph_ai.settings import settings

logger = logging.getLogger("pathgraph_ai")


def _utc_now() -> datetime:
    """Returns current UTC timestamp for contract error responses."""
    return datetime.now(timezone.utc)


def _format_error_loc(loc: tuple[Any, ...]) -> str:
    """Formats Pydantic error location tuple into standard dot/index notation."""
    parts = [str(x) for x in loc if x != "body"]
    res: list[str] = []
    for p in parts:
        if p.isdigit():
            if res:
                res[-1] = f"{res[-1]}[{p}]"
            else:
                res.append(f"[{p}]")
        else:
            res.append(p)
    return ".".join(res)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manages service lifespan: loads knowledge graph and caches into memory at boot."""
    logger.info("Initializing PathGraph Competency Graph from %s", settings.processed_data_dir)
    prereqs_path = settings.seeds_dir / "prerequisites.csv"

    # Load 4-layer competency graph
    graph = CompetencyGraph.from_processed_data(
        processed_dir=settings.processed_data_dir,
        prerequisites_csv_path=prereqs_path,
    )

    # Initialize lexical resolver and path recommendation engine
    resolver = LexicalResolver(skills=graph.skills)
    path_engine = PathEngine(graph=graph)

    # Attach to application state for zero-cost per-request access
    app.state.graph = graph
    app.state.resolver = resolver
    app.state.path_engine = path_engine

    logger.info(
        "PathGraph AI Service successfully booted (Occupations: %d, Skills: %d)",
        len(graph.occupations),
        len(graph.skills),
    )
    yield
    logger.info("Shutting down PathGraph AI Service")


def create_app() -> FastAPI:
    """Builds and configures the FastAPI application instance conforming to OpenAPI contract."""
    app = FastAPI(
        title="PathGraph AI Recommendation Service",
        version="1.0.0",
        description=(
            "Foundational contract for the PathGraph AI Recommendation Service. "
            "Computes personalized, skill-gap-driven, and prerequisite-aware learning paths "
            "for learners targeting specific canonical ESCO occupations."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handler: Request Schema Validation Errors (HTTP 422)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            ErrorDetail(
                field=_format_error_loc(err.get("loc", ())),
                issue=err.get("msg", "Validation error"),
            )
            for err in exc.errors()
        ]
        err_payload = ErrorResponse(
            code="VALIDATION_ERROR",
            message="Validation failed for one or more request fields.",
            details=details,
            timestamp=_utc_now(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=err_payload.model_dump(mode="json"),
        )

    # Exception Handler: Invalid Target Occupation (HTTP 400)
    @app.exception_handler(InvalidTargetOccupationError)
    async def invalid_target_handler(
        request: Request, exc: InvalidTargetOccupationError
    ) -> JSONResponse:
        err_payload = ErrorResponse(
            code="INVALID_TARGET_OCCUPATION",
            message=str(exc),
            details=None,
            timestamp=_utc_now(),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=err_payload.model_dump(mode="json"),
        )

    # Exception Handler: Unsupported Target Type (HTTP 400)
    @app.exception_handler(UnsupportedTargetTypeError)
    async def unsupported_target_type_handler(
        request: Request, exc: UnsupportedTargetTypeError
    ) -> JSONResponse:
        err_payload = ErrorResponse(
            code="UNSUPPORTED_TARGET_TYPE",
            message=str(exc),
            details=None,
            timestamp=_utc_now(),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=err_payload.model_dump(mode="json"),
        )

    # Exception Handler: Prerequisite Cycle Failure (HTTP 500)
    @app.exception_handler(PrerequisiteCycleError)
    async def prerequisite_cycle_handler(
        request: Request, exc: PrerequisiteCycleError
    ) -> JSONResponse:
        err_payload = ErrorResponse(
            code="PREREQUISITE_CYCLE_DETECTED",
            message=str(exc),
            details=None,
            timestamp=_utc_now(),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=err_payload.model_dump(mode="json"),
        )

    # Exception Handler: Generic Unexpected Internal Server Error (HTTP 500)
    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unexpected error in PathGraph AI service: %s", exc)
        err_payload = ErrorResponse(
            code="INTERNAL_SERVER_ERROR",
            message=f"An unexpected internal error occurred: {exc}",
            details=None,
            timestamp=_utc_now(),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=err_payload.model_dump(mode="json"),
        )

    # Mount API routers
    app.include_router(health_router)
    app.include_router(recommendations_router)
    app.include_router(v1_router)

    return app


app = create_app()
