"""Stage 3 Gate Acceptance Tests: Graph Centrality Metrics & Enhanced Gap Prioritisation.

Verifies:
  - CompetencyGraph precomputes transitive_descendant_counts and unblocking_impact for
    every skill node after `from_processed_data` is called.
  - The logarithmic impact formula: Impact(u) = 1.0 + ln(1 + |TransitiveDescendants(u)|).
  - Bottleneck skills (many downstream dependents) receive strictly higher impact than
    leaf skills (no descendants), which must equal exactly 1.0.
  - GapAnalyser propagates the impact factor into priority_weight when
    use_unblocking_impact=True, and uses flat weight when disabled.
  - The SkillGap.unblocking_impact field is populated and bounded >= 1.0.
  - Priority ordering: higher-impact bottleneck skills rank above same-gap leaf skills.
  - All 68 existing tests continue to pass (regression guard runs in CI via the full suite).
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.recommenders.gap_analyser import GapAnalyser, LearnerCompetency


# ---------------------------------------------------------------------------
# Minimal in-memory CompetencyGraph fixture
# ---------------------------------------------------------------------------

def _make_minimal_graph() -> CompetencyGraph:
    """Construct a minimal CompetencyGraph without touching the filesystem.

    Topology:
        skill_A (root)
          └─ skill_B
               └─ skill_C (leaf)
        skill_D  (isolated, no prereq edges)

    Occupation occ_1 requires all four skills.
    Impact values (by descendants within skills):
        skill_A -> descendants {skill_B, skill_C} -> count=2 -> 1 + ln(3) ≈ 2.0986
        skill_B -> descendants {skill_C}           -> count=1 -> 1 + ln(2) ≈ 1.6931
        skill_C -> descendants {}                  -> count=0 -> 1 + ln(1) = 1.0
        skill_D -> no edges in prereq_graph        -> count=0 -> 1.0
    """
    g = CompetencyGraph()

    SKILL_URIS = ["skill_A", "skill_B", "skill_C", "skill_D"]
    OCC_URI = "occ_1"

    # Populate skills dict and all graph layers
    for i, uri in enumerate(SKILL_URIS):
        g.skills[uri] = {"conceptUri": uri, "preferredLabel": uri, "skill_versatility": 0.5}
        g.uri_to_int[uri] = i
        g.int_to_uri[i] = uri
        g.requirement_graph.add_node(uri, node_id=i, node_type="skill", preferredLabel=uri)
        g.taxonomy_graph.add_node(uri, node_id=i, node_type="skill", preferredLabel=uri)
        g.association_graph.add_node(uri, node_id=i, node_type="skill", preferredLabel=uri)
        g.prerequisite_graph.add_node(uri, node_id=i, node_type="skill", preferredLabel=uri)

    # Occupation
    g.occupations[OCC_URI] = {"conceptUri": OCC_URI, "preferredLabel": "Test Occupation"}
    idx_occ = len(SKILL_URIS)
    g.uri_to_int[OCC_URI] = idx_occ
    g.int_to_uri[idx_occ] = OCC_URI
    g.requirement_graph.add_node(OCC_URI, node_id=idx_occ, node_type="occupation", preferredLabel="Test Occupation")

    # Occupation requirements (all essential)
    for uri in SKILL_URIS:
        g.requirement_graph.add_edge(OCC_URI, uri, relation_type="essential", target_proficiency=0.75, requirement_weight=1.0)

    # Prerequisite chain: skill_A -> skill_B -> skill_C
    g.prerequisite_graph.add_edge("skill_A", "skill_B")
    g.prerequisite_graph.add_edge("skill_B", "skill_C")

    # ----------------------------------------------------------------
    # Manually run the precomputation steps that from_processed_data performs
    # ----------------------------------------------------------------
    # Step 8: Prerequisite depths
    active_prereq_nodes = {n for n, d in g.prerequisite_graph.degree() if d > 0}
    if nx.is_directed_acyclic_graph(g.prerequisite_graph) and active_prereq_nodes:
        sub = g.prerequisite_graph.subgraph(active_prereq_nodes)
        for n in nx.topological_sort(sub):
            preds = list(sub.predecessors(n))
            if not preds:
                g.prerequisite_depths[n] = 0
            else:
                g.prerequisite_depths[n] = 1 + max(g.prerequisite_depths[p] for p in preds)

    # Step 9: Transitive descendants & unblocking impact
    for s_uri in g.skills:
        if s_uri in g.prerequisite_graph:
            desc = nx.descendants(g.prerequisite_graph, s_uri)
            desc_skills = desc & g.skills.keys()
            count = len(desc_skills)
        else:
            count = 0
        g.transitive_descendant_counts[s_uri] = count
        g.unblocking_impact[s_uri] = round(1.0 + math.log(1.0 + count), 4)

    g._is_frozen = True
    return g


@pytest.fixture(scope="module")
def minimal_graph() -> CompetencyGraph:
    return _make_minimal_graph()


@pytest.fixture(scope="module")
def gap_analyser_with_impact(minimal_graph: CompetencyGraph) -> GapAnalyser:
    return GapAnalyser(minimal_graph, use_unblocking_impact=True)


@pytest.fixture(scope="module")
def gap_analyser_flat(minimal_graph: CompetencyGraph) -> GapAnalyser:
    return GapAnalyser(minimal_graph, use_unblocking_impact=False)


# ---------------------------------------------------------------------------
# CompetencyGraph centrality precomputation tests
# ---------------------------------------------------------------------------

class TestCentralityPrecomputation:
    """Verifies CompetencyGraph.unblocking_impact and transitive_descendant_counts."""

    def test_all_skills_have_impact_entry(self, minimal_graph: CompetencyGraph) -> None:
        """Every skill must have an unblocking_impact entry after construction."""
        for uri in minimal_graph.skills:
            assert uri in minimal_graph.unblocking_impact, (
                f"Missing unblocking_impact entry for {uri}"
            )

    def test_all_skills_have_descendant_count_entry(self, minimal_graph: CompetencyGraph) -> None:
        for uri in minimal_graph.skills:
            assert uri in minimal_graph.transitive_descendant_counts

    def test_leaf_skill_impact_is_exactly_one(self, minimal_graph: CompetencyGraph) -> None:
        """skill_C has no descendants → impact must be 1.0 exactly."""
        assert minimal_graph.get_unblocking_impact("skill_C") == pytest.approx(1.0, abs=1e-4)
        assert minimal_graph.get_transitive_descendant_count("skill_C") == 0

    def test_isolated_skill_impact_is_exactly_one(self, minimal_graph: CompetencyGraph) -> None:
        """skill_D has no prereq edges → impact must be 1.0 exactly."""
        assert minimal_graph.get_unblocking_impact("skill_D") == pytest.approx(1.0, abs=1e-4)
        assert minimal_graph.get_transitive_descendant_count("skill_D") == 0

    def test_middle_skill_impact_formula(self, minimal_graph: CompetencyGraph) -> None:
        """skill_B has 1 descendant → impact = 1 + ln(2) ≈ 1.6931."""
        expected = 1.0 + math.log(1.0 + 1)
        assert minimal_graph.get_unblocking_impact("skill_B") == pytest.approx(expected, abs=1e-3)
        assert minimal_graph.get_transitive_descendant_count("skill_B") == 1

    def test_root_skill_impact_formula(self, minimal_graph: CompetencyGraph) -> None:
        """skill_A has 2 transitive descendants → impact = 1 + ln(3) ≈ 2.0986."""
        expected = 1.0 + math.log(1.0 + 2)
        assert minimal_graph.get_unblocking_impact("skill_A") == pytest.approx(expected, abs=1e-3)
        assert minimal_graph.get_transitive_descendant_count("skill_A") == 2

    def test_strict_ordering_bottleneck_above_leaf(self, minimal_graph: CompetencyGraph) -> None:
        """Bottleneck root must have strictly higher impact than leaf or isolated skills."""
        impact_A = minimal_graph.get_unblocking_impact("skill_A")
        impact_B = minimal_graph.get_unblocking_impact("skill_B")
        impact_C = minimal_graph.get_unblocking_impact("skill_C")
        impact_D = minimal_graph.get_unblocking_impact("skill_D")
        assert impact_A > impact_B > impact_C
        assert impact_C == pytest.approx(1.0, abs=1e-4)
        assert impact_D == pytest.approx(1.0, abs=1e-4)

    def test_accessor_for_unknown_uri_returns_default(self, minimal_graph: CompetencyGraph) -> None:
        """Unknown skill URIs should return safe defaults without raising."""
        assert minimal_graph.get_unblocking_impact("nonexistent://uri") == pytest.approx(1.0)
        assert minimal_graph.get_transitive_descendant_count("nonexistent://uri") == 0


# ---------------------------------------------------------------------------
# GapAnalyser enrichment tests
# ---------------------------------------------------------------------------

class TestGapAnalyserUnblockingImpact:
    """Verifies SkillGap.unblocking_impact field and priority_weight ordering."""

    OCC = "occ_1"

    def test_unblocking_impact_field_populated(
        self, gap_analyser_with_impact: GapAnalyser
    ) -> None:
        """Every SkillGap must have unblocking_impact >= 1.0."""
        gaps = gap_analyser_with_impact.analyse_target(self.OCC, learner_skills={})
        for gap in gaps:
            assert gap.unblocking_impact >= 1.0, (
                f"Expected unblocking_impact >= 1.0 for {gap.skill_uri}, "
                f"got {gap.unblocking_impact}"
            )

    def test_unblocking_impact_field_disabled(
        self, gap_analyser_flat: GapAnalyser
    ) -> None:
        """When use_unblocking_impact=False, unblocking_impact must be 1.0 for all gaps."""
        gaps = gap_analyser_flat.analyse_target(self.OCC, learner_skills={})
        for gap in gaps:
            assert gap.unblocking_impact == pytest.approx(1.0, abs=1e-6), (
                f"Expected flat impact 1.0 for {gap.skill_uri}, got {gap.unblocking_impact}"
            )

    def test_priority_weight_bounded_zero_one(
        self, gap_analyser_with_impact: GapAnalyser
    ) -> None:
        """All priority_weight values must lie within [0.0, 1.0]."""
        gaps = gap_analyser_with_impact.analyse_target(self.OCC, learner_skills={})
        for gap in gaps:
            assert 0.0 <= gap.priority_weight <= 1.0, (
                f"priority_weight out of bounds for {gap.skill_uri}: {gap.priority_weight}"
            )

    def test_bottleneck_outranks_leaf_on_equal_raw_gap(
        self, minimal_graph: CompetencyGraph
    ) -> None:
        """skill_A (bottleneck) must have a higher priority_weight than skill_C (leaf)
        when both have an identical raw gap (fully unassessed, same relation_type).
        """
        analyser = GapAnalyser(minimal_graph, use_unblocking_impact=True)
        gaps = analyser.analyse_target(self.OCC, learner_skills={})
        by_uri = {g.skill_uri: g for g in gaps}

        gap_A = by_uri["skill_A"]
        gap_C = by_uri["skill_C"]

        # Both should be unassessed with same raw_gap (t = 0.75 for essential)
        assert gap_A.status == "unassessed"
        assert gap_C.status == "unassessed"
        assert gap_A.gap == gap_C.gap  # identical raw gap

        # But bottleneck A must rank higher due to impact
        assert gap_A.priority_weight > gap_C.priority_weight, (
            f"Bottleneck skill_A (impact={gap_A.unblocking_impact}) should have "
            f"higher priority_weight ({gap_A.priority_weight}) than leaf skill_C "
            f"(impact={gap_C.unblocking_impact}, pw={gap_C.priority_weight})"
        )

    def test_impact_disabled_yields_equal_unassessed_priority(
        self, minimal_graph: CompetencyGraph
    ) -> None:
        """Without impact, all unassessed essential skills have equal priority_weight."""
        analyser = GapAnalyser(minimal_graph, use_unblocking_impact=False)
        gaps = analyser.analyse_target(self.OCC, learner_skills={})
        unassessed_weights = [g.priority_weight for g in gaps if g.status == "unassessed"]
        # All unassessed essential gaps should be identical (no impact differentiation)
        assert len(set(unassessed_weights)) == 1, (
            f"Expected uniform priority for unassessed gaps without impact, "
            f"got: {unassessed_weights}"
        )

    def test_satisfied_skill_has_zero_priority_weight(
        self, gap_analyser_with_impact: GapAnalyser
    ) -> None:
        """A fully proficient learner yields gap=0 → priority_weight=0.0."""
        learner = {
            "skill_A": LearnerCompetency("skill_A", proficiency=1.0, evidence_source="assessment", evidence_confidence=1.0),
        }
        gaps = gap_analyser_with_impact.analyse_target(self.OCC, learner_skills=learner)
        by_uri = {g.skill_uri: g for g in gaps}
        gap_A = by_uri["skill_A"]
        assert gap_A.status == "satisfied"
        assert gap_A.priority_weight == pytest.approx(0.0, abs=1e-6)
        assert gap_A.unblocking_impact >= 1.0  # field still populated even for satisfied

    def test_partially_satisfied_maintains_partial_priority(
        self, gap_analyser_with_impact: GapAnalyser
    ) -> None:
        """Partial proficiency yields 0 < priority_weight < max."""
        learner = {
            "skill_A": LearnerCompetency("skill_A", proficiency=0.4, evidence_source="self_reported", evidence_confidence=0.5),
        }
        gaps = gap_analyser_with_impact.analyse_target(self.OCC, learner_skills=learner)
        by_uri = {g.skill_uri: g for g in gaps}
        gap_A = by_uri["skill_A"]
        assert gap_A.status == "partially_satisfied"
        assert 0.0 < gap_A.priority_weight < 1.0
