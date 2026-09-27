"""Contract validation tests for PathGraph Recommendation API.

Ensures that the OpenAPI specification, Pydantic models, and JSON payloads
remain in strict parity without silent drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from referencing import Registry, Resource

from pathgraph_ai.api.schemas import (
    ErrorResponse,
    LearnerSkill,
    ReasonCode,
    RecommendationRequest,
    RecommendationResponse,
)

# Resolve repository paths
AI_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = AI_DIR.parent
CONTRACTS_DIR = REPO_ROOT / "contracts"
OPENAPI_SPEC_PATH = CONTRACTS_DIR / "ai.openapi.yaml"
FIXTURES_DIR = CONTRACTS_DIR / "fixtures"


@pytest.fixture(scope="session")
def openapi_spec() -> dict[str, Any]:
    """Loads the parsed OpenAPI specification."""
    assert OPENAPI_SPEC_PATH.exists(), f"OpenAPI contract not found at {OPENAPI_SPEC_PATH}"
    with open(OPENAPI_SPEC_PATH, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    return spec


@pytest.fixture(scope="session")
def schema_registry(openapi_spec: dict[str, Any]) -> Registry:
    """Builds a referencing registry containing all OpenAPI component schemas."""
    schemas = openapi_spec.get("components", {}).get("schemas", {})
    resource = Resource.from_contents(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "components": {"schemas": schemas},
        }
    )
    return Registry().with_resource("urn:pathgraph-api", resource)


def validate_payload(
    schema_name: str,
    payload: dict[str, Any],
    openapi_spec: dict[str, Any],
    registry: Registry,
) -> None:
    """Helper to validate payload against an OpenAPI component schema."""
    schemas = openapi_spec["components"]["schemas"]
    assert schema_name in schemas, f"Schema {schema_name} not found in OpenAPI spec"

    schema = schemas[schema_name]
    # Anchor references inside the whole document
    full_doc = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "components": {"schemas": schemas},
        **schema,
    }
    validator = Draft202012Validator(full_doc, registry=registry)
    validator.validate(payload)


def load_fixture(name: str) -> dict[str, Any]:
    """Helper to load a JSON fixture file."""
    path = FIXTURES_DIR / name
    assert path.exists(), f"Fixture {name} not found at {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestOpenApiContractParity:
    """Validates OpenAPI spec structure and parity with Python models."""

    def test_openapi_spec_structure(self, openapi_spec: dict[str, Any]) -> None:
        """Verifies fundamental metadata and endpoint existence."""
        assert openapi_spec.get("openapi") == "3.1.0"
        paths = openapi_spec.get("paths", {})
        assert "/v1/recommendations" in paths
        rec_post = paths["/v1/recommendations"].get("post")
        assert rec_post is not None
        assert rec_post.get("operationId") == "generateRecommendations"

        responses = rec_post.get("responses", {})
        assert "200" in responses
        assert "400" in responses
        assert "422" in responses
        assert "500" in responses

    def test_reason_codes_exact_sync(self, openapi_spec: dict[str, Any]) -> None:
        """Ensures Pydantic ReasonCode enum exactly mirrors OpenAPI ReasonCode enum."""
        spec_enums = openapi_spec["components"]["schemas"]["ReasonCode"]["enum"]
        pydantic_enums = [code.value for code in ReasonCode]

        assert set(spec_enums) == set(pydantic_enums), (
            f"Drift detected between OpenAPI ReasonCode {spec_enums} "
            f"and Pydantic ReasonCode {pydantic_enums}"
        )
        assert len(spec_enums) == len(pydantic_enums)

    def test_required_fields_sync(self, openapi_spec: dict[str, Any]) -> None:
        """Ensures required fields in OpenAPI match Pydantic model required fields."""
        schemas = openapi_spec["components"]["schemas"]

        # RecommendationRequest
        assert schemas["RecommendationRequest"]["required"] == ["learner"]

        # RecommendationTarget
        assert schemas["RecommendationTarget"]["required"] == ["occupation_id"]

        # LearnerSkill
        assert schemas["LearnerSkill"]["required"] == ["label"]

        # PathSkillRecommendation
        expected_path_required = {
            "skill_id",
            "label",
            "priority",
            "gap",
            "prerequisite_satisfied",
            "relevance",
            "demand",
            "reason",
            "reason_codes",
        }
        assert set(schemas["PathSkillRecommendation"]["required"]) == expected_path_required

        # RecommendationMetadata
        expected_meta_required = {"target_occupation_id", "unresolved_skills", "generated_at"}
        assert set(schemas["RecommendationMetadata"]["required"]) == expected_meta_required


class TestJsonSchemaFixtures:
    """Validates minimal and full fixtures against OpenAPI JSON Schema definitions."""

    def test_valid_request_full_fixture(
        self, openapi_spec: dict[str, Any], schema_registry: Registry
    ) -> None:
        payload = load_fixture("valid_request_full.json")
        validate_payload("RecommendationRequest", payload, openapi_spec, schema_registry)

    def test_valid_request_minimal_fixture(
        self, openapi_spec: dict[str, Any], schema_registry: Registry
    ) -> None:
        payload = load_fixture("valid_request_minimal.json")
        validate_payload("RecommendationRequest", payload, openapi_spec, schema_registry)

    def test_valid_response_fixture(
        self, openapi_spec: dict[str, Any], schema_registry: Registry
    ) -> None:
        payload = load_fixture("valid_response.json")
        validate_payload("RecommendationResponse", payload, openapi_spec, schema_registry)

    def test_error_response_400_fixture(
        self, openapi_spec: dict[str, Any], schema_registry: Registry
    ) -> None:
        payload = load_fixture("error_response_400.json")
        validate_payload("ErrorResponse", payload, openapi_spec, schema_registry)

    def test_error_response_422_fixture(
        self, openapi_spec: dict[str, Any], schema_registry: Registry
    ) -> None:
        payload = load_fixture("error_response_422.json")
        validate_payload("ErrorResponse", payload, openapi_spec, schema_registry)

    def test_invalid_request_bad_proficiency_rejected_by_json_schema(
        self, openapi_spec: dict[str, Any], schema_registry: Registry
    ) -> None:
        payload = load_fixture("invalid_request_bad_proficiency.json")
        with pytest.raises(Exception):
            validate_payload("RecommendationRequest", payload, openapi_spec, schema_registry)

    def test_invalid_request_missing_target_rejected_by_json_schema(
        self, openapi_spec: dict[str, Any], schema_registry: Registry
    ) -> None:
        payload = load_fixture("invalid_request_missing_target.json")
        with pytest.raises(Exception):
            validate_payload("RecommendationRequest", payload, openapi_spec, schema_registry)


class TestPydanticModelBehavior:
    """Validates runtime serialization, defaults, and boundary conditions in Pydantic."""

    def test_pydantic_valid_request_full(self) -> None:
        payload = load_fixture("valid_request_full.json")
        req = RecommendationRequest.model_validate(payload)
        assert len(req.learner.skills) == 3
        assert req.learner.skills[0].proficiency == 0.8
        assert req.learner.skills[1].skill_id is None
        assert req.learner.skills[1].proficiency == 0.4
        # Third skill has omitted proficiency, must default to 0.5
        assert req.learner.skills[2].proficiency == 0.5
        assert req.learner.target.occupation_id == "http://data.europa.eu/esco/occupation/2512.1"

    def test_pydantic_valid_request_minimal(self) -> None:
        payload = load_fixture("valid_request_minimal.json")
        req = RecommendationRequest.model_validate(payload)
        assert len(req.learner.skills) == 1
        assert req.learner.skills[0].skill_id is None
        assert req.learner.skills[0].label == "Python programming"
        assert req.learner.skills[0].proficiency == 0.5

    def test_pydantic_proficiency_default_and_none_handling(self) -> None:
        # Case 1: omitted
        skill1 = LearnerSkill(label="SQL")
        assert skill1.proficiency == 0.5

        # Case 2: explicitly None
        skill2 = LearnerSkill(label="SQL", proficiency=None)
        assert skill2.proficiency == 0.5

        # Case 3: explicitly 0.0 (boundary)
        skill3 = LearnerSkill(label="SQL", proficiency=0.0)
        assert skill3.proficiency == 0.0

        # Case 4: explicitly 1.0 (boundary)
        skill4 = LearnerSkill(label="SQL", proficiency=1.0)
        assert skill4.proficiency == 1.0

    def test_pydantic_proficiency_out_of_bounds(self) -> None:
        with pytest.raises(ValidationError):
            LearnerSkill(label="SQL", proficiency=1.01)

        with pytest.raises(ValidationError):
            LearnerSkill(label="SQL", proficiency=-0.01)

    def test_pydantic_empty_labels_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LearnerSkill(label="")

    def test_pydantic_valid_response(self) -> None:
        payload = load_fixture("valid_response.json")
        res = RecommendationResponse.model_validate(payload)
        assert len(res.learning_path) == 2
        first = res.learning_path[0]
        assert first.priority == 1
        assert first.gap == 0.2
        assert first.prerequisite_satisfied is True
        assert ReasonCode.REQUIRED_BY_TARGET in first.reason_codes
        assert res.meta.unresolved_skills == ["quantum cooking"]

    def test_pydantic_error_responses(self) -> None:
        err400 = ErrorResponse.model_validate(load_fixture("error_response_400.json"))
        assert err400.code == "INVALID_TARGET_OCCUPATION"
        assert err400.details is None

        err422 = ErrorResponse.model_validate(load_fixture("error_response_422.json"))
        assert err422.code == "VALIDATION_ERROR"
        assert err422.details is not None
        assert len(err422.details) == 1
        assert err422.details[0].field == "learner.skills[0].proficiency"
