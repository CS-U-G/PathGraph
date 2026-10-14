"""Generates and validates contract-compliant fixtures for the 5 seed occupations."""

from __future__ import annotations

import json
from pathlib import Path

from pathgraph_ai.api.schemas import (
    LearnerProfile,
    LearnerSkill,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationTarget,
)
from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.recommenders.path_engine import PathEngine
from pathgraph_ai.settings import settings

AI_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = AI_DIR / "data" / "processed"
SEEDS_DIR = AI_DIR.parent / "knowledge" / "seeds"
FIXTURES_DIR = AI_DIR / "tests" / "fixtures" / "seed_occupations"

SEED_OCCUPATIONS = {
    "software_developer": "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1",
    "web_developer": "http://data.europa.eu/esco/occupation/c40a2919-48a9-40ea-b506-1f34f693496d",
    "data_engineer": "http://data.europa.eu/esco/occupation/2079755f-d809-49e6-8037-4de6180e54c0",
    "ict_system_administrator": "http://data.europa.eu/esco/occupation/9e2e6e1e-363b-4e1b-a673-7bc0f7343300",
    "ethical_hacker": "http://data.europa.eu/esco/occupation/76ef0a87-6afe-4560-b5d0-9a086abe45c5",
}


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    graph = CompetencyGraph.from_processed_data(
        processed_dir=PROCESSED_DIR,
        prerequisites_csv_path=SEEDS_DIR / "prerequisites.csv",
    )
    engine = PathEngine(graph)

    print("Generating fixtures for 5 seed occupations...")
    for slug, uri in SEED_OCCUPATIONS.items():
        req = RecommendationRequest(
            learner=LearnerProfile(
                skills=[
                    LearnerSkill(
                        skill_id="http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97",
                        label="computer programming",
                        proficiency=0.6,
                    )
                ],
                target=RecommendationTarget(occupation_id=uri),
            )
        )
        resp = engine.recommend(req)

        # Validate with Pydantic
        validated_resp = RecommendationResponse.model_validate(resp.model_dump())

        fixture_payload = {
            "occupation_slug": slug,
            "occupation_uri": uri,
            "request": req.model_dump(mode="json"),
            "response": validated_resp.model_dump(mode="json"),
        }

        out_path = FIXTURES_DIR / f"{slug}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fixture_payload, f, indent=2)
        print(f"  -> Wrote {out_path} ({len(resp.learning_path)} items)")


if __name__ == "__main__":
    main()
