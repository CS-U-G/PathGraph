"""Shared test fixtures for PathGraph AI test suite."""
from pathlib import Path
import pytest

from pathgraph_ai.domain.competency_graph import CompetencyGraph

AI_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = AI_DIR / "data" / "processed"
SEEDS_DIR = AI_DIR.parent / "knowledge" / "seeds"


@pytest.fixture(scope="session")
def graph() -> CompetencyGraph:
    """Loads and caches the full CompetencyGraph for the entire test session."""
    return CompetencyGraph.from_processed_data(
        processed_dir=PROCESSED_DIR,
        prerequisites_csv_path=SEEDS_DIR / "prerequisites.csv",
    )
