"""Tests for SkillGapAnalyzer and Embedding-Aware Skill Gap Analysis.

Verifies:
1. Exact matching backward compatibility and report metrics.
2. Embedding-aware semantic candidate discovery without compromising exact gap metrics.
3. Strict separation of exact match certification from semantic suggestions.
4. Error handling: missing artifacts, invalid dimensions, non-finite vectors, unknown URIs.
5. Deterministic tie-breaking and duplicate prevention.
6. Absence of SentenceTransformer / PyTorch loading during precomputed queries.
7. End-to-end integration with precomputed skills artifact and knowledge graph.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pathgraph_ai.embeddings.artifact import EmbeddingArtifact
from pathgraph_ai.recommenders.skill_gap import (
    MatchedSkill,
    MissingSkill,
    SemanticCandidate,
    SkillGapAnalyzer,
    SkillGapReport,
)
from pathgraph_ai.settings import settings


# -----------------------------------------------------------------------------
# Fixtures and Helpers
# -----------------------------------------------------------------------------


def create_synthetic_artifact(
    uris: Optional[List[str]] = None,
    embeddings: Optional[np.ndarray] = None,
    dim: int = 4,
) -> EmbeddingArtifact:
    """Creates a deterministic synthetic EmbeddingArtifact with unit-normalized vectors."""
    if uris is None:
        uris = [
            "http://data.europa.eu/esco/skill/s1",
            "http://data.europa.eu/esco/skill/s2",
            "http://data.europa.eu/esco/skill/s3",
            "http://data.europa.eu/esco/skill/s4",
        ]

    if embeddings is None:
        # Predefined normalized vectors in 4D:
        # s1: [1.0, 0.0, 0.0, 0.0]
        # s2: [0.8, 0.6, 0.0, 0.0]  (cosine similarity with s1 is 0.8)
        # s3: [0.8, 0.6, 0.0, 0.0]  (identical vector to s2, testing ties)
        # s4: [0.0, 0.0, 1.0, 0.0]  (orthogonal to s1, cosine similarity 0.0)
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.6, 0.0, 0.0],
                [0.8, 0.6, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

    metadata = {
        "model_name": "synthetic-test",
        "entity_type": "skill",
        "entity_count": len(uris),
        "embedding_dimension": dim,
    }
    return EmbeddingArtifact(embeddings=embeddings, uris=uris, metadata=metadata)


# -----------------------------------------------------------------------------
# 1. Exact-Match Behavior and Backward Compatibility
# -----------------------------------------------------------------------------


class TestExactMatchCompatibility:
    """Verifies that existing exact-match functionality is 100% preserved."""

    def test_analyze_gap_exact_essential_and_optional(self):
        analyzer = SkillGapAnalyzer()
        required = [
            {"skillUri": "uri:python", "skillLabel": "Python", "relationType": "essential"},
            {"skillUri": "uri:sql", "skillLabel": "SQL", "relationType": "essential"},
            {"skillUri": "uri:git", "skillLabel": "Git", "relationType": "optional"},
            {"skillUri": "uri:docker", "skillLabel": "Docker", "relationType": "optional"},
        ]
        user_skills = {"uri:python", "uri:git"}

        report = analyzer.analyze_gap_exact(
            occupation_uri="occ:dev",
            occupation_label="Developer",
            required_skills=required,
            user_skill_uris=user_skills,
        )

        assert report.occupation_uri == "occ:dev"
        assert report.occupation_label == "Developer"
        assert report.total_required_skills == 4

        # Matched essential
        assert len(report.matched_essential) == 1
        assert report.matched_essential[0].skill_uri == "uri:python"
        assert report.matched_essential[0].match_source == "exact_uri"
        assert report.matched_essential[0].similarity_score == 1.0

        # Missing essential
        assert len(report.missing_essential) == 1
        assert report.missing_essential[0].skill_uri == "uri:sql"

        # Matched optional
        assert len(report.matched_optional) == 1
        assert report.matched_optional[0].skill_uri == "uri:git"

        # Missing optional
        assert len(report.missing_optional) == 1
        assert report.missing_optional[0].skill_uri == "uri:docker"

        # Percentages
        assert report.essential_match_percentage == pytest.approx(50.0)
        assert report.overall_match_percentage == pytest.approx(50.0)
        assert report.semantic_candidates == []

    def test_analyze_gap_exact_100_percent_match(self):
        analyzer = SkillGapAnalyzer()
        required = [
            {"skillUri": "uri:s1", "skillLabel": "Skill 1", "relationType": "essential"},
            {"skillUri": "uri:s2", "skillLabel": "Skill 2", "relationType": "optional"},
        ]
        user_skills = {"uri:s1", "uri:s2"}

        report = analyzer.analyze_gap_exact("occ:1", "Occ 1", required, user_skills)
        assert report.essential_match_percentage == pytest.approx(100.0)
        assert report.overall_match_percentage == pytest.approx(100.0)
        assert len(report.missing_essential) == 0
        assert len(report.missing_optional) == 0

    def test_analyze_gap_exact_zero_percent_match(self):
        analyzer = SkillGapAnalyzer()
        required = [
            {"skillUri": "uri:s1", "skillLabel": "Skill 1", "relationType": "essential"},
        ]
        user_skills = {"uri:unrelated"}

        report = analyzer.analyze_gap_exact("occ:1", "Occ 1", required, user_skills)
        assert report.essential_match_percentage == pytest.approx(0.0)
        assert report.overall_match_percentage == pytest.approx(0.0)
        assert len(report.matched_essential) == 0
        assert len(report.missing_essential) == 1


# -----------------------------------------------------------------------------
# 2. Exact Matches vs. Semantic Candidate Suggestions
# -----------------------------------------------------------------------------


class TestSemanticMatchingSemantics:
    """Verifies that semantic similarity is surfaced strictly as review candidates."""

    def test_semantic_match_does_not_certify_proficiency(self):
        """High cosine similarity must NOT remove skills from missing_essential or inflate match percentage."""
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        # Target requires s1 (essential). User has s2 (which has 0.8 cosine similarity with s1).
        required = [
            {"skillUri": "http://data.europa.eu/esco/skill/s1", "skillLabel": "Skill One", "relationType": "essential"}
        ]
        user_skills = {"http://data.europa.eu/esco/skill/s2"}

        report = analyzer.analyze_gap_semantic(
            occupation_uri="occ:target",
            occupation_label="Target Occupation",
            required_skills=required,
            user_skill_uris=user_skills,
            top_k_candidates=1,
            similarity_threshold=0.5,
        )

        # 1. Exact match metrics MUST remain strictly 0.0%
        assert report.essential_match_percentage == 0.0
        assert report.overall_match_percentage == 0.0
        assert len(report.matched_essential) == 0

        # 2. Skill One MUST remain in missing_essential
        assert len(report.missing_essential) == 1
        assert report.missing_essential[0].skill_uri == "http://data.europa.eu/esco/skill/s1"

        # 3. Candidate suggestion must be present in semantic_candidates
        assert len(report.semantic_candidates) == 1
        cand = report.semantic_candidates[0]
        assert cand.required_skill_uri == "http://data.europa.eu/esco/skill/s1"
        assert cand.candidate_skill_uri == "http://data.europa.eu/esco/skill/s2"
        assert cand.similarity_score == pytest.approx(0.8, abs=1e-5)
        assert cand.matching_method == "precomputed_cosine"

    def test_exact_match_separated_from_semantic_suggestions(self):
        """User has an exact match for s1 and a related skill s3 for missing s2."""
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        required = [
            {"skillUri": "http://data.europa.eu/esco/skill/s1", "skillLabel": "Skill One", "relationType": "essential"},
            {"skillUri": "http://data.europa.eu/esco/skill/s2", "skillLabel": "Skill Two", "relationType": "essential"},
        ]
        user_skills = {
            "http://data.europa.eu/esco/skill/s1",
            "http://data.europa.eu/esco/skill/s3",
        }

        report = analyzer.analyze_gap_semantic(
            occupation_uri="occ:test",
            required_skills=required,
            user_skill_uris=user_skills,
            top_k_candidates=1,
        )

        assert len(report.matched_essential) == 1
        assert report.matched_essential[0].skill_uri == "http://data.europa.eu/esco/skill/s1"
        assert report.matched_essential[0].match_source == "exact_uri"

        assert len(report.missing_essential) == 1
        assert report.missing_essential[0].skill_uri == "http://data.europa.eu/esco/skill/s2"

        assert len(report.semantic_candidates) == 1
        cand = report.semantic_candidates[0]
        assert cand.required_skill_uri == "http://data.europa.eu/esco/skill/s2"
        assert cand.candidate_skill_uri == "http://data.europa.eu/esco/skill/s3"
        assert cand.similarity_score == pytest.approx(1.0, abs=1e-5)

    def test_candidate_grouping_helper(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        required = [
            {"skillUri": "http://data.europa.eu/esco/skill/s1", "skillLabel": "Skill One", "relationType": "essential"}
        ]
        user_skills = {
            "http://data.europa.eu/esco/skill/s2",
            "http://data.europa.eu/esco/skill/s3",
        }

        report = analyzer.analyze_gap_semantic(
            occupation_uri="occ:test",
            required_skills=required,
            user_skill_uris=user_skills,
            top_k_candidates=2,
        )

        by_req = report.candidates_by_required_skill
        assert "http://data.europa.eu/esco/skill/s1" in by_req
        assert len(by_req["http://data.europa.eu/esco/skill/s1"]) == 2


# -----------------------------------------------------------------------------
# 3. Missing Artifacts and Unknown URIs
# -----------------------------------------------------------------------------


class TestArtifactAndURIErrors:
    """Verifies explicit handling of missing files, unknown URIs, and unmapped inputs."""

    def test_missing_artifact_raises_file_not_found(self):
        non_existent = Path("non_existent_test_directory/does_not_exist.npz")
        analyzer = SkillGapAnalyzer(artifact_path=non_existent)

        with pytest.raises(FileNotFoundError, match="not found"):
            _ = analyzer.skills_artifact

    def test_find_similar_skills_unknown_uri_raises_key_error(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        with pytest.raises(KeyError, match="not found in precomputed embedding artifact"):
            analyzer.find_similar_skills("http://data.europa.eu/esco/skill/unknown_uri")

    def test_unknown_uri_in_user_skills_does_not_crash_gap_analysis(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        required = [
            {"skillUri": "http://data.europa.eu/esco/skill/s1", "skillLabel": "Skill One", "relationType": "essential"}
        ]
        user_skills = {"http://data.europa.eu/esco/skill/non_existent_user_skill"}

        report = analyzer.analyze_gap_semantic(
            occupation_uri="occ:test",
            required_skills=required,
            user_skill_uris=user_skills,
        )

        assert len(report.missing_essential) == 1
        assert len(report.semantic_candidates) == 0

    def test_unknown_uri_in_required_skills_does_not_crash_gap_analysis(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        required = [
            {"skillUri": "http://data.europa.eu/esco/skill/unknown_required", "skillLabel": "Unknown", "relationType": "essential"}
        ]
        user_skills = {"http://data.europa.eu/esco/skill/s1"}

        report = analyzer.analyze_gap_semantic(
            occupation_uri="occ:test",
            required_skills=required,
            user_skill_uris=user_skills,
        )

        assert len(report.missing_essential) == 1
        assert len(report.semantic_candidates) == 0


# -----------------------------------------------------------------------------
# 4. Empty Inputs and Edge Cases
# -----------------------------------------------------------------------------


class TestEmptyInputsAndEdgeCases:
    """Verifies behavior on boundary conditions and empty containers."""

    def test_empty_user_skills(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)
        required = [
            {"skillUri": "http://data.europa.eu/esco/skill/s1", "skillLabel": "S1", "relationType": "essential"}
        ]

        report = analyzer.analyze_gap_semantic(
            occupation_uri="occ:test",
            required_skills=required,
            user_skill_uris=set(),
        )

        assert report.total_required_skills == 1
        assert report.essential_match_percentage == 0.0
        assert len(report.missing_essential) == 1
        assert len(report.semantic_candidates) == 0

    def test_empty_required_skills(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        report = analyzer.analyze_gap_semantic(
            occupation_uri="occ:test",
            required_skills=[],
            user_skill_uris={"http://data.europa.eu/esco/skill/s1"},
        )

        assert report.total_required_skills == 0
        assert report.essential_match_percentage == 0.0
        assert report.overall_match_percentage == 0.0
        assert len(report.semantic_candidates) == 0

    def test_both_inputs_empty(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        report = analyzer.analyze_gap_semantic(
            occupation_uri="occ:test",
            required_skills=[],
            user_skill_uris=set(),
        )

        assert report.total_required_skills == 0
        assert report.essential_match_percentage == 0.0
        assert report.overall_match_percentage == 0.0


# -----------------------------------------------------------------------------
# 5. Invalid Dimensions, Non-Finite Vectors, and Invalid Artifacts
# -----------------------------------------------------------------------------


class TestValidationAndErrorGuards:
    """Verifies that malformed vectors and invalid artifacts raise descriptive errors."""

    def test_invalid_artifact_dimensions(self):
        embeddings_3d = np.zeros((4, 4, 2), dtype=np.float32)
        artifact = EmbeddingArtifact(
            embeddings=embeddings_3d,
            uris=["u1", "u2", "u3", "u4"],
            metadata={},
        )
        with pytest.raises(ValueError, match="must be a 2D matrix"):
            SkillGapAnalyzer(skills_artifact=artifact)

    def test_invalid_artifact_non_finite_values(self):
        embeddings_nan = np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32)
        artifact = EmbeddingArtifact(
            embeddings=embeddings_nan,
            uris=["u1", "u2"],
            metadata={},
        )
        with pytest.raises(ValueError, match="non-finite values"):
            SkillGapAnalyzer(skills_artifact=artifact)

    def test_invalid_artifact_row_count_mismatch(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        artifact = EmbeddingArtifact(
            embeddings=embeddings,
            uris=["u1"],
            metadata={},
        )
        with pytest.raises(ValueError, match="row count mismatch"):
            SkillGapAnalyzer(skills_artifact=artifact)

    def test_query_vector_dimension_mismatch(self):
        artifact = create_synthetic_artifact(dim=4)
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        wrong_dim_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            analyzer.find_similar_by_vector(wrong_dim_vec)

    def test_query_vector_non_finite(self):
        artifact = create_synthetic_artifact(dim=4)
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        nan_vec = np.array([1.0, np.nan, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite values"):
            analyzer.find_similar_by_vector(nan_vec)

    def test_query_vector_zero_norm(self):
        artifact = create_synthetic_artifact(dim=4)
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        zero_vec = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="zero or invalid norm"):
            analyzer.find_similar_by_vector(zero_vec)

    def test_invalid_top_k(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            analyzer.find_similar_skills("http://data.europa.eu/esco/skill/s1", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            analyzer.find_similar_skills("http://data.europa.eu/esco/skill/s1", top_k=-2)

    def test_invalid_candidate_pool(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        with pytest.raises(ValueError, match="Invalid candidate_pool"):
            analyzer.analyze_gap_semantic(
                occupation_uri="occ:1",
                required_skills=[{"skillUri": "u1", "skillLabel": "l1", "relationType": "essential"}],
                user_skill_uris=set(),
                candidate_pool="invalid_pool",
            )


# -----------------------------------------------------------------------------
# 6. Duplicate Prevention, Self-Exclusion, and Deterministic Top-K
# -----------------------------------------------------------------------------


class TestDeterministicSearchAndExclusions:
    """Verifies that similarity search excludes self, avoids duplicates, and breaks ties deterministically."""

    def test_self_exclusion_in_find_similar_skills(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        query_uri = "http://data.europa.eu/esco/skill/s1"
        results = analyzer.find_similar_skills(query_uri, top_k=5, exclude_self=True)

        result_uris = [uri for uri, _ in results]
        assert query_uri not in result_uris

    def test_deterministic_ordering_on_score_ties(self):
        """When two skills have identical similarity, they must be ordered alphabetically by URI."""
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        results = analyzer.find_similar_skills(
            "http://data.europa.eu/esco/skill/s1",
            top_k=5,
            exclude_self=True,
        )

        assert len(results) >= 2
        assert results[0][0] == "http://data.europa.eu/esco/skill/s2"
        assert results[1][0] == "http://data.europa.eu/esco/skill/s3"
        assert results[0][1] == pytest.approx(results[1][1], abs=1e-5)

    def test_candidate_uris_deduplication(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        duplicate_candidates = [
            "http://data.europa.eu/esco/skill/s2",
            "http://data.europa.eu/esco/skill/s2",
            "http://data.europa.eu/esco/skill/s3",
        ]

        results = analyzer.find_similar_skills(
            "http://data.europa.eu/esco/skill/s1",
            top_k=5,
            candidate_uris=duplicate_candidates,
            exclude_self=True,
        )

        result_uris = [u for u, _ in results]
        assert len(result_uris) == len(set(result_uris))
        assert len(result_uris) == 2


# -----------------------------------------------------------------------------
# 7. Proof that Precomputed Queries Do Not Invoke SentenceTransformer
# -----------------------------------------------------------------------------


class TestZeroModelReloading:
    """Verifies that precomputed gap analysis and similarity search never load SentenceTransformer."""

    def test_precomputed_queries_do_not_invoke_sentence_transformer(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        required = [
            {"skillUri": "http://data.europa.eu/esco/skill/s1", "skillLabel": "Skill One", "relationType": "essential"}
        ]
        user_skills = {"http://data.europa.eu/esco/skill/s2"}

        with patch("pathgraph_ai.embeddings.model.SentenceTransformer", side_effect=RuntimeError("SentenceTransformer was loaded!")):
            rep_exact = analyzer.analyze_gap_exact("occ:1", "Occ", required, user_skills)
            assert rep_exact.essential_match_percentage == 0.0

            rep_semantic = analyzer.analyze_gap_semantic(
                "occ:1",
                user_skills,
                required_skills=required,
                top_k_candidates=1,
            )
            assert len(rep_semantic.semantic_candidates) == 1

            sim = analyzer.find_similar_skills("http://data.europa.eu/esco/skill/s1", top_k=2)
            assert len(sim) == 2


# -----------------------------------------------------------------------------
# 8. Knowledge Graph Integration
# -----------------------------------------------------------------------------


class TestKnowledgeGraphIntegration:
    """Verifies integration with ESCOKnowledgeGraph when required_skills is omitted."""

    def test_analyze_gap_semantic_queries_knowledge_graph(self):
        artifact = create_synthetic_artifact()

        mock_kg = MagicMock()
        mock_relation = MagicMock()
        mock_relation.skill_uri = "http://data.europa.eu/esco/skill/s1"
        mock_relation.skill_label = "Skill One"
        mock_relation.relation_type = "essential"
        mock_relation.skill_type = "skill/competence"
        mock_kg.get_occupation_skills_detailed.return_value = [mock_relation]

        mock_occ_node = MagicMock()
        mock_occ_node.label = "Software Engineer"
        mock_kg.get_node.return_value = mock_occ_node

        analyzer = SkillGapAnalyzer(knowledge_graph=mock_kg, skills_artifact=artifact)

        report = analyzer.analyze_gap_semantic(
            occupation_uri="occ:swe",
            user_skill_uris={"http://data.europa.eu/esco/skill/s2"},
            top_k_candidates=1,
        )

        mock_kg.get_occupation_skills_detailed.assert_called_once_with("occ:swe")
        assert report.occupation_label == "Software Engineer"
        assert report.total_required_skills == 1
        assert len(report.missing_essential) == 1
        assert len(report.semantic_candidates) == 1

    def test_missing_required_skills_and_missing_kg_raises_error(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(knowledge_graph=None, skills_artifact=artifact)

        with pytest.raises(ValueError, match="Either required_skills or a knowledge_graph"):
            analyzer.analyze_gap_semantic(
                occupation_uri="occ:1",
                user_skill_uris=set(),
            )


# -----------------------------------------------------------------------------
# 9. Real Artifact End-to-End Test (if artifact exists on disk)
# -----------------------------------------------------------------------------


class TestRealArtifactEndToEnd:
    """Validates real skills_embeddings.npz on disk if present in settings.skills_embeddings_path."""

    @pytest.fixture
    def real_artifact_path(self):
        path = settings.skills_embeddings_path
        if not path.exists():
            pytest.skip("skills_embeddings.npz not present on disk.")
        return path

    def test_real_artifact_similarity_search(self, real_artifact_path):
        analyzer = SkillGapAnalyzer(artifact_path=real_artifact_path)
        first_uri = analyzer.skills_artifact.uris[0]

        similar = analyzer.find_similar_skills(first_uri, top_k=5, exclude_self=True)
        assert len(similar) == 5
        assert first_uri not in [u for u, _ in similar]
        scores = [s for _, s in similar]
        assert scores == sorted(scores, reverse=True)
        assert all(-1.0 <= s <= 1.0 for s in scores)


# -----------------------------------------------------------------------------
# 10. URI-to-Vector Mapping and Unified API Routing
# -----------------------------------------------------------------------------


class TestUriMappingAndUnifiedRouting:
    """Verifies that vector lookups resolve strictly by ESCO URI and unified analyze_gap routes correctly."""

    def test_correct_uri_to_vector_mapping_independent_of_array_position(self):
        uris = ["uri:third", "uri:first", "uri:second"]
        embeddings = np.array(
            [
                [0.0, 0.0, 1.0],  # third
                [1.0, 0.0, 0.0],  # first
                [0.0, 1.0, 0.0],  # second
            ],
            dtype=np.float32,
        )
        artifact = EmbeddingArtifact(
            embeddings=embeddings,
            uris=uris,
            metadata={"dimension": 3},
        )
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        results = analyzer.find_similar_skills("uri:first", top_k=2, exclude_self=True)
        assert len(results) == 2
        assert results[0][0] == "uri:second"
        assert results[1][0] == "uri:third"
        assert results[0][1] == pytest.approx(0.0)
        assert results[1][1] == pytest.approx(0.0)

    def test_analyze_gap_routing_exact_vs_semantic(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        required = [
            {"skillUri": "http://data.europa.eu/esco/skill/s1", "skillLabel": "S1", "relationType": "essential"}
        ]
        user_skills = {"http://data.europa.eu/esco/skill/s2"}

        rep_exact = analyzer.analyze_gap(
            occupation_uri="occ:1",
            user_skill_uris=user_skills,
            required_skills=required,
            enable_semantic_matching=False,
        )
        assert len(rep_exact.semantic_candidates) == 0
        assert rep_exact.essential_match_percentage == 0.0

        rep_semantic = analyzer.analyze_gap(
            occupation_uri="occ:1",
            user_skill_uris=user_skills,
            required_skills=required,
            enable_semantic_matching=True,
            top_k_candidates=1,
        )
        assert len(rep_semantic.semantic_candidates) == 1
        assert rep_semantic.semantic_candidates[0].candidate_skill_uri == "http://data.europa.eu/esco/skill/s2"

    def test_analyze_gap_semantic_catalog_candidate_pool(self):
        artifact = create_synthetic_artifact()
        analyzer = SkillGapAnalyzer(skills_artifact=artifact)

        required = [
            {"skillUri": "http://data.europa.eu/esco/skill/s1", "skillLabel": "S1", "relationType": "essential"}
        ]
        report = analyzer.analyze_gap_semantic(
            occupation_uri="occ:1",
            user_skill_uris=set(),
            required_skills=required,
            candidate_pool="catalog",
            top_k_candidates=2,
        )

        assert report.essential_match_percentage == 0.0
        assert len(report.missing_essential) == 1
        assert len(report.semantic_candidates) == 2
        cand_uris = [c.candidate_skill_uri for c in report.semantic_candidates]
        assert "http://data.europa.eu/esco/skill/s2" in cand_uris
        assert "http://data.europa.eu/esco/skill/s3" in cand_uris
