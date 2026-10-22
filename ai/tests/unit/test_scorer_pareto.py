"""Unit tests for MultiObjectivePathScorer and ParetoOptimizer."""
import networkx as nx
import pytest

from pathgraph_ai.domain.learner import LearningPreferences, PacePreference
from pathgraph_ai.recommenders.pareto import ParetoCandidate, ParetoOptimizer
from pathgraph_ai.recommenders.scorer import MultiObjectivePathScorer, PathwayObjectiveScores


@pytest.fixture
def mock_subgraph_and_prefs():
    graph = nx.DiGraph()
    graph.add_node("s1", difficulty=1, category="Foundations")
    graph.add_node("s2", difficulty=2, category="Core")
    graph.add_node("s3", difficulty=3, category="Core")
    graph.add_edges_from([("s1", "s2"), ("s2", "s3")])

    prefs = LearningPreferences(weekly_hours_budget=15, pace_preference=PacePreference.BALANCED)
    return graph, prefs


@pytest.mark.unit
def test_scorer_objective_bounds(mock_subgraph_and_prefs):
    graph, prefs = mock_subgraph_and_prefs
    scorer = MultiObjectivePathScorer(subgraph=graph, preferences=prefs)

    scores = scorer.score_path(["s1", "s2", "s3"])
    assert 0.0 <= scores.goal_alignment <= 1.0
    assert 0.0 <= scores.gap_reduction <= 1.0
    assert 0.0 <= scores.difficulty_smoothness <= 1.0
    assert 0.0 <= scores.time_budget_fit <= 1.0
    assert 0.0 <= scores.category_continuity <= 1.0
    assert 0.0 <= scores.composite_score <= 1.0


@pytest.mark.unit
def test_scorer_pacing_weight_adjustment(mock_subgraph_and_prefs):
    graph, _ = mock_subgraph_and_prefs
    prefs_fast = LearningPreferences(pace_preference=PacePreference.FAST_TRACK)
    prefs_found = LearningPreferences(pace_preference=PacePreference.FOUNDATIONAL_FIRST)

    scorer_fast = MultiObjectivePathScorer(subgraph=graph, preferences=prefs_fast)
    scorer_found = MultiObjectivePathScorer(subgraph=graph, preferences=prefs_found)

    w_fast = scorer_fast._get_weights()
    w_found = scorer_found._get_weights()

    assert w_fast["goal"] > w_found["goal"]
    assert w_found["smooth"] > w_fast["smooth"]


@pytest.mark.unit
def test_pareto_dominance_and_frontier():
    optimizer = ParetoOptimizer()

    cand_a = ParetoCandidate(
        strategy="A",
        skill_ids=["1", "2"],
        scores=PathwayObjectiveScores(0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
    )
    cand_b = ParetoCandidate(
        strategy="B",
        skill_ids=["2", "1"],
        scores=PathwayObjectiveScores(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    )
    cand_c = ParetoCandidate(
        strategy="C",
        skill_ids=["1", "2"],
        scores=PathwayObjectiveScores(0.4, 0.95, 0.95, 0.9, 0.9, 0.8),
    )

    # A strictly dominates B
    assert optimizer.dominates(cand_a.scores, cand_b.scores)
    # A does not dominate C (C has higher gap_reduction 0.95 > 0.9)
    assert not optimizer.dominates(cand_a.scores, cand_c.scores)

    frontier = optimizer.extract_frontier([cand_a, cand_b, cand_c])
    frontier_strategies = {c.strategy for c in frontier}
    assert "A" in frontier_strategies
    assert "C" in frontier_strategies
    assert "B" not in frontier_strategies


@pytest.mark.unit
def test_pareto_tradeoff_explanation():
    optimizer = ParetoOptimizer()

    cand_a = ParetoCandidate(
        strategy="goal_first",
        skill_ids=["s1", "s2"],
        scores=PathwayObjectiveScores(0.95, 0.80, 0.70, 0.90, 0.60, 0.85),
    )
    cand_b = ParetoCandidate(
        strategy="smooth_first",
        skill_ids=["s2", "s1"],
        scores=PathwayObjectiveScores(0.60, 0.80, 0.95, 0.90, 0.80, 0.78),
    )

    tradeoffs = optimizer.explain_tradeoffs(recommended=cand_a, alternatives=[cand_a, cand_b])
    assert len(tradeoffs) == 2

    primary = tradeoffs[0]
    assert primary.strategy == "goal_first"
    assert primary.is_recommended is True

    alt = tradeoffs[1]
    assert alt.strategy == "smooth_first"
    assert alt.is_recommended is False
    assert any("gradient" in t or "smoother" in t for t in alt.tradeoffs_vs_primary)
