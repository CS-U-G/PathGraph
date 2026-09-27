# PathGraph Recommendation API Contract

## 1. Boundary & Architectural Role

`contracts/ai.openapi.yaml` defines the foundational contract for PathGraph's AI recommendation engine. 

```
┌───────────────────────────────┐
│     Client (Web / Mobile)     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  ASP.NET Core Backend Gateway │
└───────────────┬───────────────┘
                │
                ▼ POST /v1/recommendations
┌───────────────────────────────┐
│      FastAPI AI Service       │
│  (ESCO Graph & Recommender)   │
└───────────────────────────────┘
```

- **Single Source of Truth**: The ASP.NET Core backend gateway, the Python FastAPI AI service, and all clients (Angular web, mobile) consume this contract as-is.
- **No Independent Redefinitions**: Neither the backend, the AI service, nor client applications are permitted to invent bespoke variations or let schemas drift.
- **Generated Types & Models**:
  - **TypeScript**: Auto-generated into `contracts/types/ai.contract.ts` using `openapi-typescript`.
  - **Python**: Mirrored in `ai/src/pathgraph_ai/api/schemas.py` and validated against the schema in CI tests.

---

## 2. API Contract Specification

- **Endpoint**: `POST /v1/recommendations`
- **Specification File**: [`contracts/ai.openapi.yaml`](./ai.openapi.yaml) (OpenAPI 3.1.0)

### 2.1 Request Structure (`RecommendationRequest`)

```json
{
  "learner": {
    "skills": [
      {
        "skill_id": "http://data.europa.eu/esco/skill/0007833a-6ff5-4089-a292-6284693b4dd6",
        "label": "Python (programming language)",
        "proficiency": 0.8
      },
      {
        "skill_id": null,
        "label": "fastapi web development"
      }
    ],
    "target": {
      "occupation_id": "http://data.europa.eu/esco/occupation/2512.1",
      "target_type": "occupation",
      "custom_goal": null
    }
  }
}
```

### 2.2 Response Structure (`RecommendationResponse`)

```json
{
  "learning_path": [
    {
      "skill_id": "http://data.europa.eu/esco/skill/0007833a-6ff5-4089-a292-6284693b4dd6",
      "label": "Python (programming language)",
      "priority": 1,
      "gap": 0.2,
      "prerequisite_satisfied": true,
      "relevance": 0.96,
      "demand": 0.91,
      "estimated_cost": 25.0,
      "reason": "Foundational requirement with 0.20 mastery gap; all prerequisite competencies met.",
      "reason_codes": [
        "REQUIRED_BY_TARGET",
        "MASTERY_GAP",
        "PREREQ_SATISFIED",
        "HIGH_RELEVANCE"
      ]
    }
  ],
  "meta": {
    "target_occupation_id": "http://data.europa.eu/esco/occupation/2512.1",
    "unresolved_skills": [
      "quantum cooking"
    ],
    "generated_at": "2026-09-27T08:30:00Z"
  }
}
```

### 2.3 Reason Codes Enum (`ReasonCode`)

Machine-readable badge/icon codes for frontend rendering:

| Reason Code | Definition |
|-------------|------------|
| `REQUIRED_BY_TARGET` | Skill is an essential or optional requirement defined for the target occupation in ESCO taxonomy. |
| `MASTERY_GAP` | Learner's current proficiency is below the required threshold for the target role. |
| `PREREQ_SATISFIED` | All foundational prerequisites in the skill dependency DAG are met. |
| `HIGH_RELEVANCE` | High semantic and graph proximity to the learner profile and target role. |
| `HIGH_DEMAND` | Skill possesses strong labor market demand and cross-domain utility. |
| `TRANSVERSAL_FOUNDATION` | Core transversal skill underpinning multiple competencies. |
| `STEPPING_STONE` | Serves as an essential prerequisite for higher-priority target skills downstream in the learning path. |

### 2.4 Error Responses

All 4xx and 5xx responses conform to a unified `ErrorResponse` model:

| Status Code | Code Identifier | Description |
|-------------|-----------------|-------------|
| `400 Bad Request` | `INVALID_TARGET_OCCUPATION` | Target occupation does not exist or cannot be resolved in ESCO. |
| `422 Unprocessable Entity` | `VALIDATION_ERROR` | Schema validation error (e.g. proficiency > 1.0, missing required fields). |
| `500 Internal Server Error` | `INTERNAL_SERVER_ERROR` | Unexpected inference, graph algorithm, or service failure. |

---

## 3. Explicit Design Decisions

### 3.1 Proficiency Semantics (Optional Default = `0.5`)

- **Decision**: `proficiency` is an optional float (`0.0 <= proficiency <= 1.0`). If omitted or provided as `null`, it strictly defaults to `0.5` ("baseline working competence").
- **Rationale**:
  1. **Semantic Intent**: When a learner lists a skill without qualifying strength, they assert working familiarity with it (intermediate-beginner / level 2-3 on a 5-point scale).
  2. **Avoiding Over-Estimation (Binary 1.0)**: Defaulting to 1.0 would falsely assume mastery, removing even advanced target skills from the learning path.
  3. **Avoiding Under-Estimation (0.0)**: Defaulting to 0.0 would render listing the skill pointless (treating it identically to having no knowledge).
  4. **Mathematical Alignment**: In gap computation `gap = max(0.0, required_proficiency - learner_proficiency)`:
     - If the target requires advanced proficiency `0.8`, assuming `0.5` yields a reasonable `0.3` learning gap.
     - If the target requires introductory familiarity `0.4`, assuming `0.5` satisfies the requirement (`gap = 0.0`).
- **Enforcement**: Documented in `ai.openapi.yaml` (`default: 0.5`), enforced in Pydantic schema validator (`@field_validator`), and verified via contract tests.

### 3.2 Unresolved Skill Handling (Fail-Soft)

- **Decision**: `skill_id` is nullable in request inputs.
  - If `skill_id` is supplied: The AI service verifies it directly against the canonical ESCO graph.
  - If `skill_id` is null or absent: The AI semantic layer maps the raw `label` to canonical ESCO skills using vector embedding similarity.
  - If a skill cannot be mapped with confidence: It is placed into `meta.unresolved_skills` (`array[string]`).
  - The recommendation pipeline **proceeds with all resolved skills** (fail-soft degradation) rather than aborting.
  - Frontend clients use `meta.unresolved_skills` to render warning chips/alerts ("We couldn't match 'X'").
- **Contrast with Target Occupation**: Unlike individual skills, `learner.target.occupation_id` is non-negotiable. If the target occupation is invalid, the engine cannot guess a learning destination and aborts with HTTP `400 Bad Request` (`INVALID_TARGET_OCCUPATION`).

### 3.3 Target Goal Extensibility

- `learner.target` requires `occupation_id: string`, but includes forward-compatible optional fields:
  - `target_type`: `"occupation" | "custom_goal"` (default: `"occupation"`).
  - `custom_goal`: `string | null` (default: `null`).
- This allows extending the recommender in future phases to support bespoke career aspirations without breaking existing client implementations.

---

## 4. Versioning Policy

1. **Path-Based Major Versions**:
   - The contract uses URI path versioning: `/v1/recommendations`.
   - Any breaking change (removing fields, altering semantics, changing types) requires incrementing the URL path to `/v2/recommendations`.
2. **Non-Breaking Minor Evolutions**:
   - Purely additive changes (adding optional request fields, adding new properties to response metadata, adding new enum values to `reason_codes`) remain within `/v1/`.
   - The OpenAPI document metadata `info.version` increments semantically (e.g., `1.0.0` -> `1.1.0`).
3. **Deprecation Window**:
   - When `/v2/` is released, `/v1/` will be marked `deprecated: true` in OpenAPI and supported for a sunset migration window.

---

## 5. Drift Prevention & Verification

- **Code Generation (TypeScript)**:
  Run from repository root:
  ```bash
  npx openapi-typescript contracts/ai.openapi.yaml -o contracts/types/ai.contract.ts
  ```
- **Automated Parity Tests (Python)**:
  Run in the `ai` module:
  ```bash
  pytest tests/unit/test_contract.py -v
  ```
  Validates that:
  - OpenAPI specification complies with OpenAPI 3.1.0 and JSON Schema Draft 2020-12.
  - Fixtures (`contracts/fixtures/*.json`) validate against both OpenAPI JSON Schema and Pydantic models.
  - `ReasonCode` enum values in Python match OpenAPI enum values exactly.
  - Required fields in Pydantic match OpenAPI required fields.
  - Proficiency defaults and out-of-bound errors are verified.
