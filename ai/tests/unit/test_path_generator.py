"""Tests for LearningPathGenerator and PrerequisiteGraph.

Verifies:
1. Kahn's topological sort over verified Prerequisite DAGs.
2. Essential vs. optional requirement prioritization during tie-breaking.
3. Deterministic alphabetical label sorting when in-degree and essentiality tie.
4. Cycle detection, cycle flagging, and preservation of cycle nodes.
5. Handling of external/already-possessed prerequisites.
6. Edge cases: empty missing skills, single skill, independent skills.
7. Integration contract with SkillGapReport and evaluation metrics.
"""

import pytest

from pathgraph_ai.evaluation.metrics import (
    evaluate_path_completeness,
    evaluate_prerequisite_violations,
)
from pathgraph_ai.recommenders.path_generator import (
    LearningPath,
    LearningPathGenerator,
    LearningStep,
    PrerequisiteGraph,
)
from pathgraph_ai.recommenders.skill_gap import SkillGapAnalyzer


# -----------------------------------------------------------------------------
# 1. Prerequisite Graph Construction
# -----------------------------------------------------------------------------


class TestPrerequisiteGraph:
    """Verifies low-level operations of the explicit PrerequisiteGraph."""

    def test_add_and_get_prerequisites(self):
        dag = PrerequisiteGraph()
        dag.add_prerequisite("uri:algebra", "uri:calculus")
        dag.add_prerequisite("uri:calculus", "uri:diff_eq")

        assert dag.get_prerequisites("uri:calculus") == {"uri:algebra"}
        assert dag.get_prerequisites("uri:diff_eq") == {"uri:calculus"}
        assert dag.get_prerequisites("uri:algebra") == set()

    def test_unknown_skill_returns_empty_set(self):
        dag = PrerequisiteGraph()
        assert dag.get_prerequisites("uri:non_existent") == set()


# -----------------------------------------------------------------------------
# 2. Topological Sorting and Priority Ordering
# -----------------------------------------------------------------------------


class TestTopologicalSequencing:
    """Verifies that LearningPathGenerator correctly sequences curricula using Kahn's algorithm."""

    def test_linear_prerequisite_chain(self):
        dag = PrerequisiteGraph()
        dag.add_prerequisite("uri:basics", "uri:intermediate")
        dag.add_prerequisite("uri:intermediate", "uri:advanced")

        generator = LearningPathGenerator(prerequisite_graph=dag)
        missing = [
            {"skillUri": "uri:advanced", "skillLabel": "Advanced Python", "relationType": "essential"},
            {"skillUri": "uri:basics", "skillLabel": "Basic Python", "relationType": "essential"},
            {"skillUri": "uri:intermediate", "skillLabel": "Intermediate Python", "relationType": "essential"},
        ]

        path = generator.generate_path("occ:dev", "Developer", missing)

        assert path.total_steps == 3
        assert not path.has_prerequisite_cycles
        uris = [s.skill_uri for s in path.steps]
        assert uris == ["uri:basics", "uri:intermediate", "uri:advanced"]

    def test_diamond_dag_sequencing(self):
        dag = PrerequisiteGraph()
        # A -> B, A -> C, B -> D, C -> D
        dag.add_prerequisite("uri:A", "uri:B")
        dag.add_prerequisite("uri:A", "uri:C")
        dag.add_prerequisite("uri:B", "uri:D")
        dag.add_prerequisite("uri:C", "uri:D")

        generator = LearningPathGenerator(prerequisite_graph=dag)
        missing = [
            {"skillUri": "uri:D", "skillLabel": "Skill D", "relationType": "essential"},
            {"skillUri": "uri:B", "skillLabel": "Skill B", "relationType": "essential"},
            {"skillUri": "uri:C", "skillLabel": "Skill C", "relationType": "essential"},
            {"skillUri": "uri:A", "skillLabel": "Skill A", "relationType": "essential"},
        ]

        path = generator.generate_path("occ:test", "Target", missing)

        assert path.total_steps == 4
        assert not path.has_prerequisite_cycles
        uris = [s.skill_uri for s in path.steps]
        # A must be first, D must be last
        assert uris[0] == "uri:A"
        assert uris[-1] == "uri:D"
        assert set(uris[1:3]) == {"uri:B", "uri:C"}

    def test_essential_prioritized_over_optional_when_independent(self):
        """Independent skills must schedule essential requirements before optional ones."""
        generator = LearningPathGenerator()
        missing = [
            {"skillUri": "uri:opt_a", "skillLabel": "A Optional", "relationType": "optional"},
            {"skillUri": "uri:ess_b", "skillLabel": "B Essential", "relationType": "essential"},
            {"skillUri": "uri:ess_a", "skillLabel": "A Essential", "relationType": "essential"},
            {"skillUri": "uri:opt_b", "skillLabel": "B Optional", "relationType": "optional"},
        ]

        path = generator.generate_path("occ:test", "Target", missing)

        uris = [s.skill_uri for s in path.steps]
        assert uris == ["uri:ess_a", "uri:ess_b", "uri:opt_a", "uri:opt_b"]

    def test_external_prerequisite_does_not_block_progression(self):
        """A prerequisite already acquired by the user (not in missing_skills) should not block the dependent."""
        dag = PrerequisiteGraph()
        dag.add_prerequisite("uri:acquired_math", "uri:machine_learning")

        generator = LearningPathGenerator(prerequisite_graph=dag)
        missing = [
            {"skillUri": "uri:machine_learning", "skillLabel": "Machine Learning", "relationType": "essential"},
        ]

        path = generator.generate_path("occ:test", "Target", missing)
        assert path.total_steps == 1
        assert path.steps[0].skill_uri == "uri:machine_learning"
        assert path.steps[0].prerequisites == []


# -----------------------------------------------------------------------------
# 3. Cycle Detection and Fault Tolerance
# -----------------------------------------------------------------------------


class TestCycleHandling:
    """Verifies that cyclic prerequisite graphs are flagged and do not drop skills."""

    def test_cyclic_prerequisites_flagged_and_preserved(self):
        dag = PrerequisiteGraph()
        dag.add_prerequisite("uri:cycle_a", "uri:cycle_b")
        dag.add_prerequisite("uri:cycle_b", "uri:cycle_a")

        generator = LearningPathGenerator(prerequisite_graph=dag)
        missing = [
            {"skillUri": "uri:cycle_a", "skillLabel": "Cycle Skill A", "relationType": "essential"},
            {"skillUri": "uri:cycle_b", "skillLabel": "Cycle Skill B", "relationType": "essential"},
            {"skillUri": "uri:independent", "skillLabel": "Independent Skill", "relationType": "essential"},
        ]

        path = generator.generate_path("occ:test", "Target", missing)

        assert path.has_prerequisite_cycles is True
        assert path.total_steps == 3
        uris = [s.skill_uri for s in path.steps]
        assert "uri:independent" in uris
        assert "uri:cycle_a" in uris
        assert "uri:cycle_b" in uris
        assert uris[0] == "uri:independent"


# -----------------------------------------------------------------------------
# 4. Edge Cases and Integration Contracts
# -----------------------------------------------------------------------------


class TestEdgeCasesAndEvaluationIntegration:
    """Verifies boundary conditions and integration with evaluation metrics."""

    def test_empty_missing_skills(self):
        generator = LearningPathGenerator()
        path = generator.generate_path("occ:test", "Target", [])

        assert path.total_steps == 0
        assert len(path.steps) == 0
        assert not path.has_prerequisite_cycles

    def test_single_missing_skill(self):
        generator = LearningPathGenerator()
        missing = [{"skillUri": "uri:only_one", "skillLabel": "Solo", "relationType": "essential"}]
        path = generator.generate_path("occ:test", "Target", missing)

        assert path.total_steps == 1
        assert path.steps[0].step_number == 1
        assert path.steps[0].skill_uri == "uri:only_one"

    def test_end_to_end_gap_report_to_path_generation(self):
        """Validates contract between SkillGapAnalyzer output and LearningPathGenerator."""
        analyzer = SkillGapAnalyzer()
        required = [
            {"skillUri": "uri:python", "skillLabel": "Python", "relationType": "essential"},
            {"skillUri": "uri:django", "skillLabel": "Django", "relationType": "essential"},
            {"skillUri": "uri:k8s", "skillLabel": "Kubernetes", "relationType": "optional"},
        ]
        user_skills = {"uri:python"}

        report = analyzer.analyze_gap_exact("occ:swe", "SWE", required, user_skills)

        missing_payload = [
            {"skillUri": s.skill_uri, "skillLabel": s.skill_label, "relationType": s.relation_type}
            for s in report.missing_essential + report.missing_optional
        ]

        dag = PrerequisiteGraph()
        dag.add_prerequisite("uri:python", "uri:django")

        generator = LearningPathGenerator(prerequisite_graph=dag)
        path = generator.generate_path(report.occupation_uri, report.occupation_label, missing_payload)

        assert path.total_steps == 2
        missing_uris = {s["skillUri"] for s in missing_payload}
        path_uris = [s.skill_uri for s in path.steps]
        comp = evaluate_path_completeness(missing_uris, path_uris)
        assert comp["is_complete"] is True
        assert comp["omitted_count"] == 0

        violations = evaluate_prerequisite_violations(path_uris, [("uri:python", "uri:django")])
        assert violations["violations_count"] == 0
        assert violations["is_valid_dag"] is True
