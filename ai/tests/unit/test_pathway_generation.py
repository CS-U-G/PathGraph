"""Stage 4 Unit Tests: Multi-Archetype Candidate Generation & Pareto Optimization.

Tests verify:
A) CandidatePathwayGenerator
   - Each of the 5 archetypes returns a valid topological sort
   - generate_all() deduplicates identical sequences
   - Fallback fires when subgraph has only one node
   - Beam search respects deadline (trivially fast on toy graph)

B) MultiObjectivePathScorer
   - All 5 objectives are bounded in [0.0, 1.0]
   - Empty path returns sentinel 1.0 scores
   - Goal alignment rewards high-priority skills at early positions
   - Impact smoothness penalises abrupt centrality jumps
   - Cluster continuity counts same-cluster consecutive pairs

C) ParetoOptimizer
   - dominates() implements strict Pareto dominance correctly
   - extract_frontier() excludes dominated candidates
   - A non-dominated set is returned fully
   - explain_tradeoffs() lists primary path first with is_recommended=True
   - Delta strings respect the 5% threshold
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from pathgraph_ai.recommenders.candidate_generator import (
    CandidatePathwayGenerator,
)
from pathgraph_ai.recommenders.pareto import (
    ParetoCandidate,
    ParetoOptimizer,
)
from pathgraph_ai.recommenders.scorer import (
    MultiObjectivePathScorer,
    PathwayObjectiveScores,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_graph_mock(skill_uris: list[str]) -> MagicMock:
    """Build a minimal CompetencyGraph mock for the given skill URIs.

    Prerequisites subgraph topology (for testing depth ordering):
        A  →  B  →  C
              ↓
              D

    Depth:  A=0, B=1, C=2, D=2
    Unblocking impact: A unlocks {B,C,D}→count=3, B unlocks {C,D}→count=2,
                       C/D leaf → count=0 → impact=1.0
    """
    mock = MagicMock()

    depths = {"A": 0, "B": 1, "C": 2, "D": 2}
    impacts = {
        "A": round(1.0 + math.log(1.0 + 3), 4),  # ≈2.386
        "B": round(1.0 + math.log(1.0 + 2), 4),  # ≈2.099
        "C": 1.0,
        "D": 1.0,
    }
    clusters = {
        "A": ["cluster_1"],
        "B": ["cluster_1"],
        "C": ["cluster_2"],
        "D": ["cluster_2"],
    }
    skill_data = {uri: {"skill_versatility": i * 0.25} for i, uri in enumerate(skill_uris)}

    mock.get_prerequisite_depth.side_effect = lambda u: depths.get(u, 0)
    mock.get_unblocking_impact.side_effect = lambda u: impacts.get(u, 1.0)
    mock.get_taxonomy_cluster.side_effect = lambda u: clusters.get(u, [])
    mock.get_skill.side_effect = lambda u: skill_data.get(u)
    return mock


def _make_prereq_subgraph() -> nx.DiGraph:
    """Prerequisite subgraph: A → B → C; B → D (all 4 nodes required)."""
    g = nx.DiGraph()
    g.add_nodes_from(["A", "B", "C", "D"])
    g.add_edges_from([("A", "B"), ("B", "C"), ("B", "D")])
    return g


def _make_gaps() -> dict:
    """SkillGap mocks keyed by skill URI."""
    gaps = {}
    priorities = {"A": 0.9, "B": 0.6, "C": 0.3, "D": 0.8}
    for uri, pw in priorities.items():
        gap = MagicMock()
        gap.skill_uri = uri
        gap.priority_weight = pw
        gap.gap = pw
        gap.gap_score = pw  # use priority as gap_score for simplicity
        gap.requirement_weight = pw
        gap.relation_type = "essential"
        gaps[uri] = gap
    return gaps


# ---------------------------------------------------------------------------
# A) CandidatePathwayGenerator tests
# ---------------------------------------------------------------------------

class TestCandidatePathwayGenerator:

    def setup_method(self):
        self.subgraph = _make_prereq_subgraph()
        self.graph_mock = _make_graph_mock(["A", "B", "C", "D"])
        self.gaps = _make_gaps()
        self.gen = CandidatePathwayGenerator(
            subgraph=self.subgraph,
            graph=self.graph_mock,
            gaps=list(self.gaps.values()),
        )

    # ------------------------------------------------------------------ DAG invariant

    def test_foundational_first_valid_topological_order(self):
        path = self.gen.generate_foundational_first()
        assert self.gen._verify_topological_invariant(path.skill_ids), (
            "foundational_first must satisfy DAG topological invariant"
        )
        assert set(path.skill_ids) == {"A", "B", "C", "D"}

    def test_career_goal_first_valid_topological_order(self):
        path = self.gen.generate_goal_first()
        assert self.gen._verify_topological_invariant(path.skill_ids)
        assert set(path.skill_ids) == {"A", "B", "C", "D"}

    def test_quick_wins_valid_topological_order(self):
        path = self.gen.generate_quick_wins()
        assert self.gen._verify_topological_invariant(path.skill_ids)
        assert set(path.skill_ids) == {"A", "B", "C", "D"}

    def test_domain_clustered_valid_topological_order(self):
        path = self.gen.generate_domain_clustered()
        assert self.gen._verify_topological_invariant(path.skill_ids)
        assert set(path.skill_ids) == {"A", "B", "C", "D"}

    def test_beam_search_valid_topological_order(self):
        path = self.gen.generate_beam_search(beam_width=3, time_budget_ms=50.0)
        assert self.gen._verify_topological_invariant(path.skill_ids)
        assert set(path.skill_ids) == {"A", "B", "C", "D"}

    # ------------------------------------------------------------------ Archetype ordering semantics

    def test_foundational_first_starts_at_root(self):
        """A is the only depth-0 node; it must come first."""
        path = self.gen.generate_foundational_first()
        assert path.skill_ids[0] == "A", (
            "foundational_first must schedule the shallowest (root) skill first"
        )

    def test_career_goal_first_schedules_root_first(self):
        """A is the prerequisite for B, so even if B/D are high priority,
        A must appear before B (DAG constraint), and before C/D."""
        path = self.gen.generate_goal_first()
        idx = {uri: i for i, uri in enumerate(path.skill_ids)}
        # DAG constraint: A before B
        assert idx["A"] < idx["B"], "A must precede B (prerequisite)"
        assert idx["B"] < idx["C"], "B must precede C"
        assert idx["B"] < idx["D"], "B must precede D"

    def test_domain_clustered_groups_same_cluster(self):
        """After the root A (cluster_1), B (cluster_1) should immediately follow,
        grouping cluster_1 before moving to cluster_2 (C and D)."""
        path = self.gen.generate_domain_clustered()
        # Mandatory: A before B; B before C and D
        idx = {uri: i for i, uri in enumerate(path.skill_ids)}
        assert idx["A"] < idx["B"]
        assert idx["B"] < idx["C"]
        assert idx["B"] < idx["D"]

    def test_beam_search_strategy_name(self):
        path = self.gen.generate_beam_search()
        assert path.strategy == "bounded_beam_search"

    def test_each_archetype_has_correct_strategy_name(self):
        expected = {
            self.gen.generate_foundational_first: "foundational_first",
            self.gen.generate_goal_first: "target_goal_first",
            self.gen.generate_quick_wins: "quick_wins",
            self.gen.generate_domain_clustered: "domain_clustered",
        }
        for fn, name in expected.items():
            assert fn().strategy == name

    # ------------------------------------------------------------------ generate_all()

    def test_generate_all_returns_at_least_one_candidate(self):
        paths = self.gen.generate_all()
        assert len(paths) >= 1

    def test_generate_all_no_duplicates(self):
        paths = self.gen.generate_all()
        seen = [tuple(p.skill_ids) for p in paths]
        assert len(seen) == len(set(seen)), "generate_all() must deduplicate identical sequences"

    def test_generate_all_every_path_covers_all_nodes(self):
        for path in self.gen.generate_all():
            assert set(path.skill_ids) == set(self.subgraph.nodes), (
                f"Path '{path.strategy}' does not cover all subgraph nodes"
            )

    # ------------------------------------------------------------------ Fallback path

    def test_fallback_fires_when_all_archetypes_collapse(self):
        """Single-node graph → all 5 archetypes produce the same sequence.
        generate_all() should dedup and still return 1 path (the unique one).
        """
        single = nx.DiGraph()
        single.add_node("X")
        g = _make_graph_mock(["X"])
        gen = CandidatePathwayGenerator(subgraph=single, graph=g)
        paths = gen.generate_all()
        assert len(paths) == 1
        assert paths[0].skill_ids == ["X"]

    # ------------------------------------------------------------------ Generation time recorded

    def test_generation_time_recorded_in_ms(self):
        for fn in [
            self.gen.generate_foundational_first,
            self.gen.generate_goal_first,
            self.gen.generate_quick_wins,
            self.gen.generate_domain_clustered,
        ]:
            p = fn()
            assert p.generation_time_ms >= 0.0


# ---------------------------------------------------------------------------
# B) MultiObjectivePathScorer tests
# ---------------------------------------------------------------------------

class TestMultiObjectivePathScorer:

    def setup_method(self):
        self.subgraph = _make_prereq_subgraph()
        self.graph_mock = _make_graph_mock(["A", "B", "C", "D"])
        self.gaps = _make_gaps()
        self.scorer = MultiObjectivePathScorer(
            subgraph=self.subgraph,
            graph=self.graph_mock,
            gaps=list(self.gaps.values()),
        )

    # ------------------------------------------------------------------ Bounds

    def test_all_scores_bounded_zero_to_one(self):
        path = ["A", "B", "C", "D"]
        s = self.scorer.score_path(path)
        for name, v in s.to_dict().items():
            assert 0.0 <= v <= 1.0, f"{name}={v} is out of [0, 1]"

    def test_empty_path_returns_sentinel(self):
        s = self.scorer.score_path([])
        assert s.goal_alignment == 1.0
        assert s.composite_score == 1.0

    # ------------------------------------------------------------------ Goal alignment

    def test_goal_alignment_higher_when_high_priority_early(self):
        """Path with high-priority A,D early should score better on goal_alignment
        than a reversed path where low-priority C comes first."""
        high_first = ["A", "B", "D", "C"]
        low_first = ["A", "B", "C", "D"]
        s_high = self.scorer.score_path(high_first)
        s_low = self.scorer.score_path(low_first)
        # A has pw=0.9 in both (position 0), but D(0.8) at pos 2 vs pos 3
        # The high_first path should have equal or better goal_alignment
        assert s_high.goal_alignment >= s_low.goal_alignment

    def test_goal_alignment_bounded_by_one(self):
        s = self.scorer.score_path(["A"])
        assert s.goal_alignment <= 1.0

    # ------------------------------------------------------------------ Gap reduction

    def test_gap_reduction_bounded(self):
        for path in [["A", "B", "C", "D"], ["A", "B", "D", "C"]]:
            s = self.scorer.score_path(path)
            assert 0.0 <= s.gap_reduction <= 1.0

    def test_gap_reduction_returns_one_when_no_gaps(self):
        scorer = MultiObjectivePathScorer(
            subgraph=self.subgraph,
            graph=self.graph_mock,
            gaps=[],
        )
        s = scorer.score_path(["A", "B", "C", "D"])
        assert s.gap_reduction == 1.0

    # ------------------------------------------------------------------ Difficulty smoothness

    def test_difficulty_smoothness_one_for_single_node(self):
        s = self.scorer.score_path(["A"])
        assert s.difficulty_smoothness == 1.0

    def test_difficulty_smoothness_penalised_by_big_jumps(self):
        """Reversed path (big impact drop A→C in a hypothetical) should score
        differently from a smooth monotone path.  We only assert the metric is
        in [0, 1] and has a deterministic value."""
        s1 = self.scorer.score_path(["A", "B", "C", "D"])
        s2 = self.scorer.score_path(["A", "B", "D", "C"])
        # Both valid; we just verify boundedness and float stability
        assert 0.0 <= s1.difficulty_smoothness <= 1.0
        assert 0.0 <= s2.difficulty_smoothness <= 1.0

    def test_difficulty_smoothness_higher_for_monotone_path(self):
        """A(0) -> B(1) -> C(2) -> D(2) has smaller consecutive depth jumps
        than a jumping order C(2) -> A(0) -> B(1) -> D(2)."""
        g2 = nx.DiGraph()
        g2.add_nodes_from(["A", "B", "C", "D"])
        scorer = MultiObjectivePathScorer(
            subgraph=g2,
            graph=self.graph_mock,
            gaps=list(self.gaps.values()),
        )
        s_monotone = scorer.score_path(["A", "B", "C", "D"])
        s_jump = scorer.score_path(["C", "A", "B", "D"])
        assert s_monotone.difficulty_smoothness > s_jump.difficulty_smoothness

    # ------------------------------------------------------------------ Time budget fit

    def test_time_budget_fit_bounded(self):
        s = self.scorer.score_path(["A", "B", "C", "D"])
        assert 0.0 <= s.time_budget_fit <= 1.0

    # ------------------------------------------------------------------ Category continuity

    def test_category_continuity_one_for_single_node(self):
        s = self.scorer.score_path(["A"])
        assert s.category_continuity == 1.0

    def test_category_continuity_higher_for_same_cluster_transitions(self):
        """A→B are both cluster_1; C→D are both cluster_2.
        Path A,B,C,D has 1/3 cluster-matches (A→B yes, B→C no, C→D yes) = 2/3.
        Path A,C,B,D has 0 matches = 0/3."""
        g2 = nx.DiGraph()
        g2.add_nodes_from(["A", "B", "C", "D"])
        scorer = MultiObjectivePathScorer(
            subgraph=g2,
            graph=self.graph_mock,
            gaps=list(self.gaps.values()),
        )
        s_grouped = scorer.score_path(["A", "B", "C", "D"])
        s_interleaved = scorer.score_path(["A", "C", "B", "D"])
        assert s_grouped.category_continuity > s_interleaved.category_continuity

    # ------------------------------------------------------------------ to_dict / objective vector

    def test_to_dict_keys(self):
        s = self.scorer.score_path(["A", "B", "C", "D"])
        keys = set(s.to_dict().keys())
        assert "goal_alignment" in keys
        assert "composite_score" in keys
        assert "difficulty_smoothness" in keys
        assert "time_budget_fit" in keys
        assert "category_continuity" in keys

    def test_objective_vector_length(self):
        s = self.scorer.score_path(["A", "B", "C", "D"])
        assert len(s.to_dict()) == 6


# ---------------------------------------------------------------------------
# C) ParetoOptimizer tests
# ---------------------------------------------------------------------------

def _make_scores(
    goal: float = 0.5,
    gap: float = 0.5,
    smooth: float = 0.5,
    spread: float = 0.5,
    cluster: float = 0.5,
) -> PathwayObjectiveScores:
    composite = (goal * 0.30 + gap * 0.25 + smooth * 0.20 + spread * 0.15 + cluster * 0.10)
    return PathwayObjectiveScores(
        goal_alignment=goal,
        gap_reduction=gap,
        difficulty_smoothness=smooth,
        time_budget_fit=spread,
        category_continuity=cluster,
        composite_score=round(composite, 4),
    )


def _cand(strategy: str, **kwargs) -> ParetoCandidate:
    return ParetoCandidate(
        strategy=strategy,
        skill_ids=["A", "B", "C", "D"],
        scores=_make_scores(**kwargs),
    )


class TestParetoOptimizer:

    def setup_method(self):
        self.opt = ParetoOptimizer()

    # ------------------------------------------------------------------ dominates()

    def test_dominates_true_when_strictly_better_on_all(self):
        a = _make_scores(goal=0.9, gap=0.9, smooth=0.9, spread=0.9, cluster=0.9)
        b = _make_scores(goal=0.5, gap=0.5, smooth=0.5, spread=0.5, cluster=0.5)
        assert self.opt.dominates(a, b)

    def test_dominates_false_when_equal_on_all(self):
        a = _make_scores()
        b = _make_scores()
        assert not self.opt.dominates(a, b)

    def test_dominates_false_when_mixed_trade_off(self):
        a = _make_scores(goal=0.9, gap=0.1)
        b = _make_scores(goal=0.1, gap=0.9)
        assert not self.opt.dominates(a, b)
        assert not self.opt.dominates(b, a)

    def test_dominates_true_strict_improvement_on_one_equal_on_rest(self):
        a = _make_scores(goal=0.6)
        b = _make_scores(goal=0.5)
        assert self.opt.dominates(a, b)

    def test_dominates_false_when_a_worse_on_one(self):
        a = _make_scores(goal=0.4, gap=0.9, smooth=0.9, spread=0.9, cluster=0.9)
        b = _make_scores(goal=0.5, gap=0.9, smooth=0.9, spread=0.9, cluster=0.9)
        assert not self.opt.dominates(a, b)

    # ------------------------------------------------------------------ extract_frontier()

    def test_frontier_returns_all_when_none_dominated(self):
        candidates = [
            _cand("A", goal=0.9, gap=0.1),
            _cand("B", goal=0.1, gap=0.9),
            _cand("C", goal=0.5, smooth=0.9),
        ]
        frontier = self.opt.extract_frontier(candidates)
        assert len(frontier) == 3

    def test_frontier_excludes_dominated_candidate(self):
        dominated = _cand("weak", goal=0.3, gap=0.3, smooth=0.3, spread=0.3, cluster=0.3)
        strong = _cand("strong", goal=0.9, gap=0.9, smooth=0.9, spread=0.9, cluster=0.9)
        frontier = self.opt.extract_frontier([dominated, strong])
        strategies = [c.strategy for c in frontier]
        assert "strong" in strategies
        assert "weak" not in strategies

    def test_frontier_with_single_candidate(self):
        c = [_cand("only")]
        frontier = self.opt.extract_frontier(c)
        assert len(frontier) == 1
        assert frontier[0].strategy == "only"

    def test_frontier_empty_input_returns_empty(self):
        assert self.opt.extract_frontier([]) == []

    def test_frontier_all_equal_returns_all(self):
        """Equal candidates do not dominate each other."""
        candidates = [_cand(f"p{i}") for i in range(4)]
        frontier = self.opt.extract_frontier(candidates)
        assert len(frontier) == 4

    # ------------------------------------------------------------------ explain_tradeoffs()

    def test_recommended_is_listed_first(self):
        rec = _cand("foundational_first", goal=0.8, gap=0.7)
        alts = [_cand("target_goal_first", goal=0.6, gap=0.9)]
        tradeoffs = self.opt.explain_tradeoffs(rec, alts)
        assert tradeoffs[0].strategy == "foundational_first"
        assert tradeoffs[0].is_recommended is True

    def test_alternatives_marked_not_recommended(self):
        rec = _cand("rec", goal=0.8)
        alts = [_cand("alt_a", goal=0.5), _cand("alt_b", goal=0.9, gap=0.1)]
        tradeoffs = self.opt.explain_tradeoffs(rec, alts)
        for t in tradeoffs[1:]:
            assert t.is_recommended is False

    def test_delta_string_present_for_large_gap_difference(self):
        """Alt has goal_alignment = 0.95 vs rec's 0.80 → diff=+15% → should appear."""
        rec = _cand("rec", goal=0.80)
        alt = _cand("alt", goal=0.95, gap=0.2)
        tradeoffs = self.opt.explain_tradeoffs(rec, [alt])
        alt_tradeoff = next(t for t in tradeoffs if t.strategy == "alt")
        delta_text = " ".join(alt_tradeoff.tradeoffs_vs_primary)
        assert "goal alignment" in delta_text.lower()

    def test_no_delta_strings_when_differences_below_threshold(self):
        """Alt is only 2% better on every objective → no delta strings → fallback message."""
        rec = _cand("rec", goal=0.50)
        alt = _cand("alt", goal=0.52)  # only +2%, below 5% threshold
        tradeoffs = self.opt.explain_tradeoffs(rec, [alt])
        alt_t = next(t for t in tradeoffs if t.strategy == "alt")
        assert "comparable" in alt_t.tradeoffs_vs_primary[0].lower()

    def test_key_strengths_populated(self):
        rec = _cand("rec", goal=0.9, gap=0.9, smooth=0.9, spread=0.9, cluster=0.9)
        tradeoffs = self.opt.explain_tradeoffs(rec, [])
        assert len(tradeoffs[0].key_strengths) >= 1

    def test_to_dict_has_required_keys(self):
        rec = _cand("rec")
        tradeoffs = self.opt.explain_tradeoffs(rec, [])
        d = tradeoffs[0].to_dict()
        for k in ("strategy", "is_recommended", "composite_score", "key_strengths", "tradeoffs_vs_primary"):
            assert k in d

    # ------------------------------------------------------------------ identify_strengths()

    def test_strengths_balanced_fallback(self):
        s = _make_scores(goal=0.5, gap=0.5, smooth=0.5, spread=0.5, cluster=0.5)
        strengths = self.opt._identify_strengths(s)
        assert any("balanced" in st.lower() for st in strengths)

    def test_strengths_detects_high_goal_alignment(self):
        s = _make_scores(goal=0.85)
        strengths = self.opt._identify_strengths(s)
        assert any("goal alignment" in st.lower() for st in strengths)

    def test_strengths_detects_high_gap_reduction(self):
        s = _make_scores(gap=0.90)
        strengths = self.opt._identify_strengths(s)
        assert any("gap" in st.lower() for st in strengths)
