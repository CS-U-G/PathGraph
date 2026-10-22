"""End-to-End Integration Verification Test Suite for PathGraph Phase 3.

Verifies the integration of:
1. ESCODataLoader and ESCOKnowledgeGraph on real ESCO v1.2 dataset.
2. Precomputed embedding artifacts (data/processed/skills_embeddings.npz).
3. SkillGapAnalyzer with exact URI matching and embedding-aware candidate discovery.
4. LearningPathGenerator with Kahn's topological sort and PrerequisiteGraph.
5. Evaluation metrics (precision@k, recall@k, MRR, prerequisite violations, path completeness).

CRITICAL DESIGN ASSUMPTIONS TESTED:
- Exact matches remain distinct from semantic suggestions.
- Semantic suggestions do NOT certify proficiency or alter match percentages.
- Required skills with semantic candidates remain in the missing-skill lists.
- Unknown URIs and empty inputs produce consistent, graceful behavior.
- Prerequisite cycles are explicitly flagged; no false topological ordering is claimed.
- Embeddings are queried from precomputed unit-normalized artifacts with zero model reloading.
"""

from pathlib import Path
from typing import List, Set

import numpy as np
import pytest

from pathgraph_ai.domain.knowledge_graph import ESCOKnowledgeGraph
from pathgraph_ai.embeddings.artifact import load_embeddings_artifact
from pathgraph_ai.evaluation.metrics import (
    evaluate_path_completeness,
    evaluate_prerequisite_violations,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from pathgraph_ai.pipelines.data_loader import ESCODataLoader
from pathgraph_ai.recommenders.path_generator import (
    LearningPathGenerator,
    PrerequisiteGraph,
)
from pathgraph_ai.recommenders.skill_gap import SkillGapAnalyzer
from pathgraph_ai.settings import settings


# =============================================================================
# Documented Deterministic Synthetic Learner Profile & Real ESCO Occupation
# =============================================================================

REAL_OCCUPATION_URI = "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1"
REAL_OCCUPATION_LABEL = "software developer"

URI_COMPUTER_PROG = "http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97"
URI_IDE = "http://data.europa.eu/esco/skill/925463a7-d51f-4d5b-9f79-4d28cf30acde"
URI_JAVA = "http://data.europa.eu/esco/skill/19a8293b-8e95-4de3-983f-77484079c389"
URI_EXEC_TESTS = "http://data.europa.eu/esco/skill/913e7e83-b8f8-4574-b1ca-1b38f3fd974a"
URI_AGRI_MACHINERY = "http://data.europa.eu/esco/skill/00dd3271-077f-4a12-a958-e297fdd724ce"
URI_UNKNOWN = "http://data.europa.eu/esco/skill/UNKNOWN-SYNTHETIC-9999"

SYNTHETIC_LEARNER_PROFILE: Set[str] = {
    URI_COMPUTER_PROG,
    URI_IDE,
    URI_JAVA,
    URI_EXEC_TESTS,
    URI_AGRI_MACHINERY,
    URI_UNKNOWN,
}

URI_ANALYSE_SPECS = "http://data.europa.eu/esco/skill/f28617ad-afdd-4041-814c-216153a38998"
URI_DEFINE_REQS = "http://data.europa.eu/esco/skill/d9e5349e-8791-49c2-8ba4-839fdd1606c2"
URI_DESIGN_PATTERNS = "http://data.europa.eu/esco/skill/2b7a79e5-84d8-4880-be66-3d9bb05bea17"
URI_PROTOTYPE = "http://data.europa.eu/esco/skill/797a4416-0b8d-44f3-8fcd-eb10bd0be6b4"
URI_DEBUG_SOFTWARE = "http://data.europa.eu/esco/skill/2522a6ce-3202-4ac8-9f5b-b9cb5a3a83a1"


# =============================================================================
# Test Class: End-to-End Integration
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests using real ESCO data and artifacts."""

    @pytest.fixture(scope="class")
    def esco_kg(self):
        """Constructs the ESCO Knowledge Graph from the actual raw dataset."""
        loader = ESCODataLoader()
        kg = ESCOKnowledgeGraph.from_loader(loader=loader)
        return kg

    @pytest.fixture(scope="class")
    def analyzer(self, esco_kg):
        """Initializes SkillGapAnalyzer with real KG and precomputed embeddings."""
        return SkillGapAnalyzer(knowledge_graph=esco_kg)

    def test_real_occupation_skill_profile(self, esco_kg):
        """Verifies the real ESCO occupation node and required skill relations."""
        node = esco_kg.get_node(REAL_OCCUPATION_URI)
        assert node is not None
        assert node.label.lower() == REAL_OCCUPATION_LABEL

        detailed = esco_kg.get_occupation_skills_detailed(REAL_OCCUPATION_URI)
        assert len(detailed) == 108

        essential = [s for s in detailed if s.relation_type == "essential"]
        optional = [s for s in detailed if s.relation_type == "optional"]
        assert len(essential) == 24
        assert len(optional) == 84

    def test_skill_gap_analysis_exact_and_semantic_separation(self, analyzer):
        """Verifies skill gap analysis on the deterministic learner profile."""
        report = analyzer.analyze_gap(
            occupation_uri=REAL_OCCUPATION_URI,
            user_skill_uris=SYNTHETIC_LEARNER_PROFILE,
            top_k_candidates=2,
            candidate_pool="user_skills",
        )

        # 1. Total requirements
        assert report.total_required_skills == 108
        assert len(report.matched_essential) == 2
        assert len(report.matched_optional) == 1
        assert len(report.missing_essential) == 22
        assert len(report.missing_optional) == 83

        # 2. Exact match certification
        matched_essential_uris = {m.skill_uri for m in report.matched_essential}
        assert matched_essential_uris == {URI_COMPUTER_PROG, URI_IDE}
        for m in report.matched_essential:
            assert m.match_source == "exact_uri"
            assert m.similarity_score == 1.0

        matched_optional_uris = {m.skill_uri for m in report.matched_optional}
        assert matched_optional_uris == {URI_JAVA}
        for m in report.matched_optional:
            assert m.match_source == "exact_uri"
            assert m.similarity_score == 1.0

        # 3. Match percentages reflect strictly verified exact matches
        assert report.essential_match_percentage == pytest.approx(8.333333, abs=1e-4)
        assert report.overall_match_percentage == pytest.approx(2.777777, abs=1e-4)

        # 4. Semantic suggestions surfaced without altering proficiency
        assert len(report.semantic_candidates) > 0

        candidates_by_req = report.candidates_by_required_skill
        assert URI_DEBUG_SOFTWARE in candidates_by_req

        debug_cands = candidates_by_req[URI_DEBUG_SOFTWARE]
        assert len(debug_cands) >= 1
        top_cand = debug_cands[0]
        assert top_cand.candidate_skill_uri == URI_EXEC_TESTS
        assert top_cand.similarity_score > 0.65

        # 5. Required skills with semantic candidates MUST remain in missing list
        missing_essential_uris = {m.skill_uri for m in report.missing_essential}
        assert URI_DEBUG_SOFTWARE in missing_essential_uris

        # 6. Unknown URI was gracefully ignored and not matched
        all_matched = matched_essential_uris | matched_optional_uris
        assert URI_UNKNOWN not in all_matched

    def test_edge_cases_unknown_uri_and_empty_profile(self, analyzer):
        """Verifies robust, consistent behavior for empty profiles and unknown URIs."""
        rep_unknown = analyzer.analyze_gap(
            occupation_uri=REAL_OCCUPATION_URI,
            user_skill_uris={URI_UNKNOWN, "http://invalid/random"},
            candidate_pool="user_skills",
        )
        assert rep_unknown.essential_match_percentage == 0.0
        assert rep_unknown.overall_match_percentage == 0.0
        assert len(rep_unknown.matched_essential) == 0
        assert len(rep_unknown.matched_optional) == 0
        assert len(rep_unknown.missing_essential) == 24
        assert len(rep_unknown.missing_optional) == 84
        assert len(rep_unknown.semantic_candidates) == 0

        rep_empty = analyzer.analyze_gap(
            occupation_uri=REAL_OCCUPATION_URI,
            user_skill_uris=set(),
            candidate_pool="user_skills",
        )
        assert rep_empty.essential_match_percentage == 0.0
        assert rep_empty.overall_match_percentage == 0.0
        assert len(rep_empty.matched_essential) == 0
        assert len(rep_empty.missing_essential) == 24
        assert len(rep_empty.semantic_candidates) == 0

    def test_learning_path_generation_independent_skills(self, analyzer):
        """Verifies path generation on missing skills when skills are independent."""
        report = analyzer.analyze_gap(
            occupation_uri=REAL_OCCUPATION_URI,
            user_skill_uris=SYNTHETIC_LEARNER_PROFILE,
        )

        missing_skills = [
            {"skillUri": m.skill_uri, "skillLabel": m.skill_label, "relationType": m.relation_type}
            for m in report.missing_essential + report.missing_optional
        ]
        assert len(missing_skills) == 105

        generator = LearningPathGenerator()
        path = generator.generate_path(
            target_occupation_uri=REAL_OCCUPATION_URI,
            target_occupation_label=report.occupation_label,
            missing_skills=missing_skills,
        )

        assert path.total_steps == 105
        assert not path.has_prerequisite_cycles

        essential_steps = [s for s in path.steps[:22]]
        optional_steps = [s for s in path.steps[22:]]
        assert all(s.relation_type == "essential" for s in essential_steps)
        assert all(s.relation_type == "optional" for s in optional_steps)

        missing_uris = {m["skillUri"] for m in missing_skills}
        path_uris = [s.skill_uri for s in path.steps]
        comp_report = evaluate_path_completeness(missing_uris, path_uris)
        assert comp_report["is_complete"]
        assert comp_report["omitted_count"] == 0
        assert comp_report["duplicate_count"] == 0

    def test_learning_path_generation_with_declared_prerequisites(self, analyzer):
        """Verifies topological sequencing when a verified prerequisite DAG is declared."""
        report = analyzer.analyze_gap(
            occupation_uri=REAL_OCCUPATION_URI,
            user_skill_uris=SYNTHETIC_LEARNER_PROFILE,
        )

        missing_skills = [
            {"skillUri": m.skill_uri, "skillLabel": m.skill_label, "relationType": m.relation_type}
            for m in report.missing_essential + report.missing_optional
        ]

        dag = PrerequisiteGraph()
        dag.add_prerequisite(URI_ANALYSE_SPECS, URI_DEFINE_REQS)
        dag.add_prerequisite(URI_DEFINE_REQS, URI_DESIGN_PATTERNS)
        dag.add_prerequisite(URI_DESIGN_PATTERNS, URI_PROTOTYPE)
        dag.add_prerequisite(URI_PROTOTYPE, URI_DEBUG_SOFTWARE)

        generator = LearningPathGenerator(prerequisite_graph=dag)
        path = generator.generate_path(
            target_occupation_uri=REAL_OCCUPATION_URI,
            target_occupation_label=report.occupation_label,
            missing_skills=missing_skills,
        )

        assert path.total_steps == 105
        assert not path.has_prerequisite_cycles

        step_uris = [s.skill_uri for s in path.steps]
        idx_analyse = step_uris.index(URI_ANALYSE_SPECS)
        idx_define = step_uris.index(URI_DEFINE_REQS)
        idx_patterns = step_uris.index(URI_DESIGN_PATTERNS)
        idx_proto = step_uris.index(URI_PROTOTYPE)
        idx_debug = step_uris.index(URI_DEBUG_SOFTWARE)

        assert idx_analyse < idx_define < idx_patterns < idx_proto < idx_debug

        declared_pairs = [
            (URI_ANALYSE_SPECS, URI_DEFINE_REQS),
            (URI_DEFINE_REQS, URI_DESIGN_PATTERNS),
            (URI_DESIGN_PATTERNS, URI_PROTOTYPE),
            (URI_PROTOTYPE, URI_DEBUG_SOFTWARE),
        ]
        violation_report = evaluate_prerequisite_violations(step_uris, declared_pairs)
        assert violation_report["is_valid_dag"]
        assert violation_report["violations_count"] == 0.0
        assert violation_report["violation_rate"] == 0.0
        assert violation_report["total_prerequisites_checked"] == 4.0

    def test_learning_path_generation_with_prerequisite_cycles(self, analyzer):
        """Verifies that prerequisite cycles are detected, flagged, and evaluated as invalid DAGs."""
        report = analyzer.analyze_gap(
            occupation_uri=REAL_OCCUPATION_URI,
            user_skill_uris=SYNTHETIC_LEARNER_PROFILE,
        )

        missing_skills = [
            {"skillUri": m.skill_uri, "skillLabel": m.skill_label, "relationType": m.relation_type}
            for m in report.missing_essential + report.missing_optional
        ]

        dag_cycle = PrerequisiteGraph()
        dag_cycle.add_prerequisite(URI_ANALYSE_SPECS, URI_DEFINE_REQS)
        dag_cycle.add_prerequisite(URI_DEFINE_REQS, URI_DESIGN_PATTERNS)
        dag_cycle.add_prerequisite(URI_DESIGN_PATTERNS, URI_PROTOTYPE)
        dag_cycle.add_prerequisite(URI_PROTOTYPE, URI_DEBUG_SOFTWARE)
        dag_cycle.add_prerequisite(URI_DEBUG_SOFTWARE, URI_ANALYSE_SPECS)

        generator = LearningPathGenerator(prerequisite_graph=dag_cycle)
        path = generator.generate_path(
            target_occupation_uri=REAL_OCCUPATION_URI,
            target_occupation_label=report.occupation_label,
            missing_skills=missing_skills,
        )

        assert path.has_prerequisite_cycles is True
        assert path.total_steps == 105

        step_uris = [s.skill_uri for s in path.steps]
        cycle_pairs = [
            (URI_ANALYSE_SPECS, URI_DEFINE_REQS),
            (URI_DEFINE_REQS, URI_DESIGN_PATTERNS),
            (URI_DESIGN_PATTERNS, URI_PROTOTYPE),
            (URI_PROTOTYPE, URI_DEBUG_SOFTWARE),
            (URI_DEBUG_SOFTWARE, URI_ANALYSE_SPECS),
        ]
        violation_report = evaluate_prerequisite_violations(step_uris, cycle_pairs)
        assert violation_report["is_valid_dag"] is False
        assert violation_report["violations_count"] > 0
        assert violation_report["violation_rate"] > 0.0

    def test_ranking_evaluation_metrics(self):
        """Verifies calculation of precision@k, recall@k, and MRR."""
        recommended = ["s1", "s2", "s3", "s4", "s5"]
        ground_truth = {"s2", "s4"}

        assert precision_at_k(recommended, ground_truth, 1) == 0.0
        assert precision_at_k(recommended, ground_truth, 2) == 0.5
        assert precision_at_k(recommended, ground_truth, 4) == 0.5

        assert recall_at_k(recommended, ground_truth, 1) == 0.0
        assert recall_at_k(recommended, ground_truth, 2) == 0.5
        assert recall_at_k(recommended, ground_truth, 4) == 1.0

        mrr = mean_reciprocal_rank([recommended], [ground_truth])
        assert mrr == 0.5

    def test_actual_full_embedding_artifact_integrity(self):
        """Verifies the actual precomputed .npz files on disk without loading models."""
        skills_path = settings.skills_embeddings_path
        assert skills_path.exists()

        artifact = load_embeddings_artifact(skills_path)
        assert artifact.shape == (13939, 384)
        assert artifact.count == 13939
        assert artifact.dimension == 384

        assert np.all(np.isfinite(artifact.embeddings))

        norms = np.linalg.norm(artifact.embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

        occ_path = settings.occupations_embeddings_path
        assert occ_path.exists()

        occ_art = load_embeddings_artifact(occ_path)
        assert occ_art.shape == (3039, 384)
        assert occ_art.count == 3039
        assert occ_art.dimension == 384
        assert np.all(np.isfinite(occ_art.embeddings))
        occ_norms = np.linalg.norm(occ_art.embeddings, axis=1)
        assert np.allclose(occ_norms, 1.0, atol=1e-5)
