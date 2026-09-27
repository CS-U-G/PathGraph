/**
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
