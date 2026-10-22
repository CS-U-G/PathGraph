"""Integration tests for PathGraph AI Recommendation Service HTTP endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from pathgraph_ai.api.schemas import ErrorResponse, RecommendationResponse
from pathgraph_ai.api.service import create_app

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "contracts" / "fixtures"


def load_fixture(filename: str) -> dict[str, Any]:
    with open(FIXTURES_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Creates a shared TestClient with full knowledge graph loaded in lifespan."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


class TestServiceEndpoints:
    """Verifies HTTP endpoints against OpenAPI 3.1 contract specifications."""

    def test_health_endpoint(self, client: TestClient) -> None:
        """GET /health must return 200 with service health and graph metadata."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "PathGraph AI Recommendation Service"
        assert data["occupations_count"] == 96
        assert data["skills_count"] == 13939

    def test_recommendations_valid_full_fixture(self, client: TestClient) -> None:
        """POST /v1/recommendations with valid full fixture and real ESCO target must return 200."""
        payload = load_fixture("valid_request_full.json")
        # Ground to canonical Software Developer occupation
        payload["learner"]["target"]["occupation_id"] = "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1"
        resp = client.post("/v1/recommendations", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        res_model = RecommendationResponse.model_validate(data)
        assert len(res_model.learning_path) > 0
        assert res_model.meta.target_occupation_id == payload["learner"]["target"]["occupation_id"]

        # Check priorities are 1-indexed and contiguous
        for idx, item in enumerate(res_model.learning_path, start=1):
            assert item.priority == idx
            assert 0.0 <= item.gap <= 1.0
            assert 0.0 <= item.relevance <= 1.0
            assert len(item.reason_codes) >= 1
            assert len(item.reason) > 0

    def test_recommendations_valid_minimal_fixture(self, client: TestClient) -> None:
        """POST /v1/recommendations with minimal payload and real ESCO target must return 200."""
        payload = load_fixture("valid_request_minimal.json")
        # Ground to canonical Web Developer occupation
        payload["learner"]["target"]["occupation_id"] = "http://data.europa.eu/esco/occupation/c40a2919-48a9-40ea-b506-1f34f693496d"
        resp = client.post("/v1/recommendations", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        res_model = RecommendationResponse.model_validate(data)
        assert len(res_model.learning_path) > 0

    def test_recommendations_raw_fixture_dummy_uri_returns_400(self, client: TestClient) -> None:
        """POST /v1/recommendations with raw valid_request_full.json (which contains placeholder URI 2512.1) returns 400."""
        payload = load_fixture("valid_request_full.json")
        resp = client.post("/v1/recommendations", json=payload)
        assert resp.status_code == 400
        err = ErrorResponse.model_validate(resp.json())
        assert err.code == "INVALID_TARGET_OCCUPATION"

    def test_recommendations_invalid_target_occupation_returns_400(self, client: TestClient) -> None:
        """POST /v1/recommendations with unknown occupation must return HTTP 400 with INVALID_TARGET_OCCUPATION."""
        payload = {
            "learner": {
                "skills": [],
                "target": {
                    "occupation_id": "http://data.europa.eu/esco/occupation/invalid_target_99999",
                },
            }
        }
        resp = client.post("/v1/recommendations", json=payload)
        assert resp.status_code == 400

        err = ErrorResponse.model_validate(resp.json())
        assert err.code == "INVALID_TARGET_OCCUPATION"
        assert "invalid_target_99999" in err.message

    def test_recommendations_unsupported_target_type_returns_400(self, client: TestClient) -> None:
        """POST /v1/recommendations with custom_goal target_type must return HTTP 400."""
        payload = {
            "learner": {
                "skills": [],
                "target": {
                    "occupation_id": "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1",
                    "target_type": "custom_goal",
                    "custom_goal": "Senior AI Architect",
                },
            }
        }
        resp = client.post("/v1/recommendations", json=payload)
        assert resp.status_code == 400

        err = ErrorResponse.model_validate(resp.json())
        assert err.code == "UNSUPPORTED_TARGET_TYPE"

    def test_recommendations_bad_proficiency_returns_422(self, client: TestClient) -> None:
        """POST /v1/recommendations with proficiency > 1.0 must return HTTP 422 with VALIDATION_ERROR."""
        payload = load_fixture("invalid_request_bad_proficiency.json")
        resp = client.post("/v1/recommendations", json=payload)
        assert resp.status_code == 422

        err = ErrorResponse.model_validate(resp.json())
        assert err.code == "VALIDATION_ERROR"
        assert err.details is not None
        assert len(err.details) >= 1
        assert "proficiency" in err.details[0].field

    def test_recommendations_missing_target_returns_422(self, client: TestClient) -> None:
        """POST /v1/recommendations with missing target must return HTTP 422 with VALIDATION_ERROR."""
        payload = load_fixture("invalid_request_missing_target.json")
        resp = client.post("/v1/recommendations", json=payload)
        assert resp.status_code == 422

        err = ErrorResponse.model_validate(resp.json())
        assert err.code == "VALIDATION_ERROR"
        assert err.details is not None
        assert len(err.details) >= 1
        assert "target" in err.details[0].field

    def test_v1_health_endpoint(self, client: TestClient) -> None:
        """GET /api/v1/health returns status, version, and component status."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["app_name"] == "PathGraph AI Service"
        assert data["esco_data_available"] is True

    def test_v1_skills_extract_endpoint(self, client: TestClient) -> None:
        """POST /api/v1/skills/extract returns 501 Not Implemented in accordance with ESCO roadmap."""
        payload = {
            "text": "Expert in Python and SQL database queries.",
        }
        resp = client.post("/api/v1/skills/extract", json=payload)
        assert resp.status_code == 501

    def test_v1_skill_gap_analysis_endpoint(self, client: TestClient) -> None:
        """POST /api/v1/skills/gap-analysis performs gap analysis for an occupation."""
        payload = {
            "target_occupation_uri": "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1",
            "user_skill_uris": [
                "http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97",
            ],
        }
        resp = client.post("/api/v1/skills/gap-analysis", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["target_occupation_uri"] == payload["target_occupation_uri"]
        assert len(data["matched_essential_skills"]) == 1
        assert len(data["missing_essential_skills"]) == 23

    def test_v1_learning_path_generate_endpoint(self, client: TestClient) -> None:
        """POST /api/v1/learning-path/generate orders missing skills topologically."""
        payload = {
            "target_occupation_uri": "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1",
            "missing_skill_uris": [
                "http://data.europa.eu/esco/skill/925463a7-d51f-4d5b-9f79-4d28cf30acde",
                "http://data.europa.eu/esco/skill/19a8293b-8e95-4de3-983f-77484079c389",
            ],
        }
        resp = client.post("/api/v1/learning-path/generate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_steps"] == 2
        assert len(data["steps"]) == 2

