"""API route modules for PathGraph AI."""

from pathgraph_ai.api.routes.health import router as health_router
from pathgraph_ai.api.routes.recommendations import router as recommendations_router

__all__ = ["health_router", "recommendations_router"]
