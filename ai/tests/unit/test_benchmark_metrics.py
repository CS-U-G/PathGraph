"""Unit tests for recommendation quality benchmark metrics."""

import pytest

from pathgraph_ai.benchmarks.metrics import (
    average_precision,
    curriculum_bloat_ratio,
    deduplicate_preserve_order,
    essential_curriculum_coverage,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    prerequisite_violation_rate,
    recall_at_k,
    recommendation_leakage_rate,
)


class TestDeduplication:
    """Verifies order-preserving deduplication helper."""

    def test_preserves_order(self):
        items = ["a", "b", "c", "a", "d", "b"]
        assert deduplicate_preserve_order(items) == ["a", "b", "c", "d"]

    def test_empty_list(self):
        assert deduplicate_preserve_order([]) == []


class TestPrecisionAtK:
    """Verifies Precision@k calculation and edge cases."""

    def test_perfect_precision(self):
        rec = ["s1", "s2", "s3", "s4"]
        gt = {"s1", "s2", "s3", "s4"}
        assert precision_at_k(rec, gt, k=3) == 1.0

    def test_partial_precision(self):
        rec = ["s1", "other1", "s2", "other2"]
        gt = {"s1", "s2"}
        assert precision_at_k(rec, gt, k=4) == 0.5
        assert precision_at_k(rec, gt, k=2) == 0.5  # 1 hit / 2

    def test_zero_precision(self):
        rec = ["other1", "other2"]
        gt = {"s1", "s2"}
        assert precision_at_k(rec, gt, k=2) == 0.0

    def test_edge_cases(self):
        assert precision_at_k([], {"s1"}, k=5) == 0.0
        assert precision_at_k(["s1"], set(), k=5) == 0.0
        assert precision_at_k(["s1"], {"s1"}, k=0) == 0.0
        assert precision_at_k(["s1"], {"s1"}, k=-2) == 0.0

    def test_duplicate_recommendations_do_not_inflate_hits(self):
        # Even if "s1" appears twice, only 1 unique hit in top-2
        rec = ["s1", "s1"]
        gt = {"s1"}
        assert precision_at_k(rec, gt, k=2) == 0.5


class TestRecallAtK:
    """Verifies Recall@k calculation and edge cases."""

    def test_perfect_recall(self):
        rec = ["s1", "s2", "s3"]
        gt = {"s1", "s2"}
        assert recall_at_k(rec, gt, k=2) == 1.0

    def test_partial_recall(self):
        rec = ["s1", "other", "other2"]
        gt = {"s1", "s2", "s3", "s4"}
        assert recall_at_k(rec, gt, k=3) == 0.25

    def test_edge_cases(self):
        assert recall_at_k([], {"s1"}, k=5) == 0.0
        assert recall_at_k(["s1"], set(), k=5) == 0.0
        assert recall_at_k(["s1"], {"s1"}, k=0) == 0.0
        assert recall_at_k(["s1"], {"s1"}, k=-1) == 0.0


class TestNDCGAtK:
    """Verifies NDCG@k calculation with binary and graded relevance."""

    def test_perfect_ranking_binary(self):
        rec = ["s1", "s2", "other"]
        gt = {"s1", "s2"}
        # Both top items are relevant, so DCG == IDCG
        assert ndcg_at_k(rec, gt, k=2) == 1.0

    def test_imperfect_ranking_binary(self):
        rec = ["other", "s1"]
        gt = {"s1"}
        # Ideal: s1 at rank 1 -> IDCG = 1 / log2(2) = 1.0
        # Actual: s1 at rank 2 -> DCG = 1 / log2(3) = 0.6309
        score = ndcg_at_k(rec, gt, k=2)
        assert 0.60 < score < 0.65

    def test_graded_relevance(self):
        rec = ["s1", "s2"]
        gt = {"s1", "s2"}
        relevance = {"s1": 0.5, "s2": 1.0}
        # Ideal: s2 at rank 1, s1 at rank 2
        score = ndcg_at_k(rec, gt, k=2, relevance_scores=relevance)
        assert 0.0 < score < 1.0

    def test_edge_cases(self):
        assert ndcg_at_k([], {"s1"}, k=5) == 0.0
        assert ndcg_at_k(["s1"], set(), k=5) == 0.0
        assert ndcg_at_k(["s1"], {"s1"}, k=0) == 0.0
        assert ndcg_at_k(["s1"], {"s1"}, k=-5) == 0.0


class TestAveragePrecisionAndMAP:
    """Verifies Average Precision and Mean Average Precision."""

    def test_average_precision_perfect(self):
        rec = ["s1", "s2"]
        gt = {"s1", "s2"}
        assert average_precision(rec, gt) == 1.0

    def test_average_precision_partial(self):
        rec = ["s1", "other", "s2"]
        gt = {"s1", "s2"}
        # rank 1: P@1 = 1/1 = 1.0
        # rank 3: P@3 = 2/3 = 0.6667
        # AP = (1.0 + 0.6667) / 2 = 0.8333
        assert average_precision(rec, gt) == pytest.approx(0.8333, abs=1e-3)

    def test_mean_average_precision(self):
        recs = [["s1", "s2"], ["other", "s1"]]
        gts = [{"s1", "s2"}, {"s1"}]
        # AP1 = 1.0
        # AP2 = (1/2) / 1 = 0.5
        # MAP = 0.75
        assert mean_average_precision(recs, gts) == 0.75

    def test_map_mismatched_lengths_raises_error(self):
        with pytest.raises(ValueError, match="Mismatched query counts"):
            mean_average_precision([["s1"]], [{"s1"}, {"s2"}])

    def test_map_empty_inputs(self):
        assert mean_average_precision([], []) == 0.0


class TestEssentialCoverage:
    """Verifies Essential Curriculum Coverage."""

    def test_full_coverage(self):
        scheduled = ["s1", "s2", "s3", "optional1"]
        essential = {"s1", "s2", "s3"}
        assert essential_curriculum_coverage(scheduled, essential) == 1.0

    def test_partial_coverage(self):
        scheduled = ["s1", "optional1"]
        essential = {"s1", "s2"}
        assert essential_curriculum_coverage(scheduled, essential) == 0.5

    def test_empty_essential_returns_one(self):
        assert essential_curriculum_coverage(["s1"], set()) == 1.0


class TestPrerequisiteViolationRate:
    """Verifies prerequisite ordering and violation detection."""

    def test_valid_dag_order(self):
        scheduled = ["p1", "s1", "p2", "s2"]
        prereqs = [("p1", "s1"), ("p2", "s2")]
        res = prerequisite_violation_rate(scheduled, prereqs)
        assert res["total_checked"] == 2
        assert res["violations_count"] == 0
        assert res["violation_rate"] == 0.0
        assert res["is_valid_dag"] is True

    def test_inverted_order_detected(self):
        scheduled = ["s1", "p1"]  # s1 scheduled before its prerequisite p1
        prereqs = [("p1", "s1")]
        res = prerequisite_violation_rate(scheduled, prereqs)
        assert res["total_checked"] == 1
        assert res["violations_count"] == 1
        assert res["violation_rate"] == 1.0
        assert res["is_valid_dag"] is False
        assert res["violating_pairs"] == [("p1", "s1")]

    def test_unrelated_prerequisites_ignored(self):
        scheduled = ["a", "b"]
        prereqs = [("x", "y")]  # Neither x nor y are in scheduled
        res = prerequisite_violation_rate(scheduled, prereqs)
        assert res["total_checked"] == 0
        assert res["violations_count"] == 0
        assert res["violation_rate"] == 0.0
        assert res["is_valid_dag"] is True


class TestCurriculumBloatRatio:
    """Verifies curriculum bloat ratio calculation."""

    def test_exact_ratio(self):
        scheduled = ["s1", "s2", "s3"]
        essential = {"s1", "s2", "s3"}
        res = curriculum_bloat_ratio(scheduled, essential)
        assert res["numerator"] == 3
        assert res["denominator"] == 3
        assert res["bloat_ratio"] == 1.0
        assert res["net_expansion"] == 0

    def test_bloated_curriculum(self):
        scheduled = ["s1", "s2", "s3", "opt1", "opt2", "opt3"]
        essential = {"s1", "s2"}
        res = curriculum_bloat_ratio(scheduled, essential)
        assert res["numerator"] == 6
        assert res["denominator"] == 2
        assert res["bloat_ratio"] == 3.0
        assert res["net_expansion"] == 4

    def test_empty_baseline_does_not_divide_by_zero(self):
        scheduled = ["s1"]
        essential = set()
        res = curriculum_bloat_ratio(scheduled, essential)
        assert res["denominator"] == 1
        assert res["bloat_ratio"] == 1.0


class TestRecommendationLeakageRate:
    """Verifies negative control recommendation leakage rate."""

    def test_zero_leakage(self):
        rec = ["cs_skill_1", "cs_skill_2"]
        irrelevant = {"agriculture_1", "farming_2"}
        res = recommendation_leakage_rate(rec, irrelevant)
        assert res["leakage_count"] == 0
        assert res["leakage_rate"] == 0.0

    def test_detected_leakage(self):
        rec = ["cs_skill", "agriculture_1", "farming_2"]
        irrelevant = {"agriculture_1", "farming_2"}
        res = recommendation_leakage_rate(rec, irrelevant)
        assert res["leakage_count"] == 2
        assert res["leakage_rate"] == pytest.approx(0.6667, abs=1e-3)
        assert res["leaked_items"] == ["agriculture_1", "farming_2"]
