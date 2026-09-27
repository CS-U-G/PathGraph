import json
import shutil
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
contracts_dir = repo_root / "contracts"
types_dir = contracts_dir / "types"
frontend_contracts_dir = repo_root / "frontend" / "src" / "app" / "core" / "contracts"

types_dir.mkdir(parents=True, exist_ok=True)
frontend_contracts_dir.mkdir(parents=True, exist_ok=True)

# 1. Write contracts/types/index.ts
index_ts_content = """/**
 * PathGraph AI Contract TypeScript Definitions
 * Convenience re-exports and aliases from the generated OpenAPI specification.
 *
 * Generated from contracts/ai.openapi.yaml via openapi-typescript.
 */

export * from "./ai.contract";
import type { components, operations } from "./ai.contract";

export type Schema<T extends keyof components["schemas"]> = components["schemas"][T];

export type ReasonCode = components["schemas"]["ReasonCode"];
export type LearnerSkill = components["schemas"]["LearnerSkill"];
export type RecommendationTarget = components["schemas"]["RecommendationTarget"];
export type LearnerProfile = components["schemas"]["LearnerProfile"];
export type RecommendationRequest = components["schemas"]["RecommendationRequest"];
export type PathSkillRecommendation = components["schemas"]["PathSkillRecommendation"];
export type RecommendationMetadata = components["schemas"]["RecommendationMetadata"];
export type RecommendationResponse = components["schemas"]["RecommendationResponse"];
export type ErrorDetail = components["schemas"]["ErrorDetail"];
export type ErrorResponse = components["schemas"]["ErrorResponse"];

export type GenerateRecommendationsOperation = operations["generateRecommendations"];
"""

(types_dir / "index.ts").write_text(index_ts_content, encoding="utf-8")
print(f"Wrote {types_dir / 'index.ts'}")

# 2. Write contracts/package.json
package_json = {
    "name": "@pathgraph/contracts",
    "version": "1.0.0",
    "description": "PathGraph cross-cutting API and AI contracts",
    "private": True,
    "main": "types/index.ts",
    "scripts": {
        "codegen": "openapi-typescript ai.openapi.yaml -o types/ai.contract.ts",
        "codegen:frontend": "openapi-typescript ai.openapi.yaml -o ../frontend/src/app/core/contracts/ai.contract.ts"
    },
    "devDependencies": {
        "openapi-typescript": "^7.13.0"
    }
}
(contracts_dir / "package.json").write_text(json.dumps(package_json, indent=2), encoding="utf-8")
print(f"Wrote {contracts_dir / 'package.json'}")

# 3. Copy generated files to frontend core contracts
shutil.copy2(types_dir / "ai.contract.ts", frontend_contracts_dir / "ai.contract.ts")
(frontend_contracts_dir / "index.ts").write_text(index_ts_content, encoding="utf-8")
print(f"Copied contract types to {frontend_contracts_dir}")
