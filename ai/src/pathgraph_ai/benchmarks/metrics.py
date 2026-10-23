"""Evaluation metrics for recommendation quality benchmarking.

Defines mathematically rigorous, independently unit-tested metric calculations:
- Precision@k and Recall@k
- Normalized Discounted Cumulative Gain (NDCG@k)
- Mean Average Precision (MAP)
- Essential-Skill Curriculum Coverage
- Prerequisite Violation Rate
- Curriculum Bloat Ratio (with explicit numerator and denominator)
- Recommendation Leakage Rate (negative control sensitivity)

Handles edge cases (empty ground truth, duplicate items, k <= 0) consistently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence, Set
from typing import Any


def deduplicate_preserve_order(items: Sequence[str]) -> list[str]:
    """Removes duplicate items while strictly preserving initial occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def precision_at_k(
    recommended: Sequence[str],
    ground_truth: Set[str],
    k: int,
) -> float:
    """Calculates Precision@k.

    Precision@k = (Number of unique recommended items in top-k that are relevant) / k

    Args:
        recommended: Ordered list of recommended item identifiers.
        ground_truth: Set of relevant ground-truth item identifiers.
        k: Evaluation cut-off rank.

    Returns:
        Precision in [0.0, 1.0]. Returns 0.0 if k <= 0 or recommended is empty.
    """
    if k <= 0 or not recommended:
        return 0.0
    deduped = deduplicate_preserve_order(recommended)[:k]
    hits = sum(1 for item in deduped if item in ground_truth)
    return round(hits / k, 4)


def recall_at_k(
    recommended: Sequence[str],
    ground_truth: Set[str],
    k: int,
) -> float:
    """Calculates Recall@k.

    Recall@k = (Number of unique recommended items in top-k that are relevant) / |ground_truth|

    Args:
        recommended: Ordered list of recommended item identifiers.
        ground_truth: Set of relevant ground-truth item identifiers.
        k: Evaluation cut-off rank.

    Returns:
        Recall in [0.0, 1.0]. Returns 0.0 if ground_truth is empty, k <= 0, or recommended is empty.
    """
    if not ground_truth or k <= 0 or not recommended:
        return 0.0
    deduped = deduplicate_preserve_order(recommended)[:k]
    hits = sum(1 for item in deduped if item in ground_truth)
    return round(hits / len(ground_truth), 4)


def ndcg_at_k(
    recommended: Sequence[str],
    ground_truth: Set[str],
    k: int,
    relevance_scores: dict[str, float] | None = None,
) -> float:
    """Calculates Normalized Discounted Cumulative Gain at rank k (NDCG@k).

    Uses standard log2 discount formula:
        DCG@k = sum_{i=1}^{min(k, |R|)} rel_i / log2(i + 1)
        IDCG@k = sum_{i=1}^{min(k, |GT|)} ideal_rel_i / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k

    Args:
        recommended: Ordered list of recommended item identifiers.
        ground_truth: Set of relevant ground-truth item identifiers.
        k: Evaluation cut-off rank.
        relevance_scores: Optional mapping from item identifier to graded relevance score in [0.0, 1.0].
                         Defaults to binary relevance (1.0 if in ground_truth, 0.0 otherwise).

    Returns:
        NDCG score in [0.0, 1.0]. Returns 0.0 if ground_truth is empty, k <= 0, or IDCG == 0.
    """
    if not ground_truth or k <= 0 or not recommended:
        return 0.0

    deduped = deduplicate_preserve_order(recommended)[:k]

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, item in enumerate(deduped, start=1):
        if item in ground_truth:
            rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
            dcg += rel / math.log2(i + 1)

    # Calculate Ideal DCG (IDCG)
    if relevance_scores:
        ideal_rels = sorted(
            [relevance_scores.get(item, 1.0) for item in ground_truth],
            reverse=True,
        )[:k]
    else:
        ideal_rels = [1.0] * min(k, len(ground_truth))

    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))

    if idcg <= 0.0:
        return 0.0

    return round(min(dcg / idcg, 1.0), 4)


def average_precision(
    recommended: Sequence[str],
    ground_truth: Set[str],
) -> float:
    """Calculates Average Precision (AP) for a single ranked recommendation list.

    AP = (1 / |ground_truth|) * sum_{i=1}^{|R|} (Precision@i * Indicator(R[i] in ground_truth))

    Args:
        recommended: Ordered list of recommended item identifiers.
        ground_truth: Set of relevant ground-truth item identifiers.

    Returns:
        Average Precision in [0.0, 1.0]. Returns 0.0 if ground_truth or recommended is empty.
    """
    if not ground_truth or not recommended:
        return 0.0

    deduped = deduplicate_preserve_order(recommended)
    hits = 0
    sum_precisions = 0.0

    for i, item in enumerate(deduped, start=1):
        if item in ground_truth:
            hits += 1
            precision_at_i = hits / i
            sum_precisions += precision_at_i

    return round(sum_precisions / len(ground_truth), 4)


def mean_average_precision(
    recommendation_lists: Sequence[Sequence[str]],
    ground_truth_sets: Sequence[Set[str]],
) -> float:
    """Calculates Mean Average Precision (MAP) across multiple queries/personas.

    Args:
        recommendation_lists: Sequence of recommended item lists.
        ground_truth_sets: Sequence of corresponding ground-truth sets.

    Returns:
        MAP in [0.0, 1.0]. Returns 0.0 if input sequences are empty.
    """
    if not recommendation_lists or not ground_truth_sets:
        return 0.0
    if len(recommendation_lists) != len(ground_truth_sets):
        raise ValueError(
            f"Mismatched query counts: {len(recommendation_lists)} recommendation lists "
            f"vs {len(ground_truth_sets)} ground truth sets."
        )

    ap_scores = [
        average_precision(recs, gt)
        for recs, gt in zip(recommendation_lists, ground_truth_sets)
    ]
    return round(sum(ap_scores) / len(ap_scores), 4)


def essential_curriculum_coverage(
    scheduled_skills: Sequence[str],
    essential_ground_truth: Set[str],
) -> float:
    """Calculates the proportion of expected essential skills covered in the curriculum.

    Coverage = |Scheduled Skills intersect Essential Ground Truth| / |Essential Ground Truth|

    Args:
        scheduled_skills: Sequence of skill URIs present in the recommended path.
        essential_ground_truth: Set of expected essential target skill URIs.

    Returns:
        Coverage ratio in [0.0, 1.0]. If essential_ground_truth is empty, returns 1.0.
    """
    if not essential_ground_truth:
        return 1.0
    scheduled_set = set(scheduled_skills)
    covered = len(scheduled_set & essential_ground_truth)
    return round(covered / len(essential_ground_truth), 4)


def prerequisite_violation_rate(
    sequenced_skills: Sequence[str],
    prerequisite_pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Calculates the prerequisite violation rate over sequenced learning path steps.

    A violation occurs when both prerequisite (u) and dependent (v) are present in the
    sequence, but u is scheduled at a position AFTER v (index(u) > index(v)).

    Args:
        sequenced_skills: Ordered list of skill URIs in the generated learning path.
        prerequisite_pairs: Sequence of (prerequisite_uri, dependent_uri) tuples.

    Returns:
        Dictionary containing:
        - 'total_checked': Count of active prerequisite pairs where both skills are scheduled.
        - 'violations_count': Count of pairs where dependent precedes prerequisite.
        - 'violation_rate': violations_count / total_checked (or 0.0 if total_checked == 0).
        - 'is_valid_dag': True if violations_count == 0.
        - 'violating_pairs': List of violating (prereq, dependent) tuples.
    """
    position_map = {uri: idx for idx, uri in enumerate(sequenced_skills)}
    total_checked = 0
    violations: list[tuple[str, str]] = []

    for prereq, dependent in prerequisite_pairs:
        if prereq in position_map and dependent in position_map:
            total_checked += 1
            if position_map[prereq] > position_map[dependent]:
                violations.append((prereq, dependent))

    rate = (len(violations) / total_checked) if total_checked > 0 else 0.0
    return {
        "total_checked": total_checked,
        "violations_count": len(violations),
        "violation_rate": round(rate, 4),
        "is_valid_dag": len(violations) == 0,
        "violating_pairs": violations,
    }


def curriculum_bloat_ratio(
    scheduled_skills: Sequence[str],
    baseline_essential_skills: Set[str],
) -> dict[str, Any]:
    """Measures curriculum expansion beyond core verified essential requirements.

    Definition:
        Numerator: Total number of unique skill steps scheduled in the learning path.
        Denominator: Total number of baseline verified essential skills required (bounded below by 1).
        Bloat Ratio: Numerator / Denominator

    Interpretation:
        - Ratio == 1.0: Exact 1-to-1 curriculum matching essential requirements.
        - Ratio > 1.0: Curriculum incorporates optional, elective, or expansion skills.
        - Ratio < 1.0: Curriculum is under-covering the verified essential requirements.

    Args:
        scheduled_skills: Sequence of skill URIs scheduled in the learning path.
        baseline_essential_skills: Set of verified essential skill URIs required.

    Returns:
        Dictionary containing:
        - 'numerator': Number of scheduled unique skills.
        - 'denominator': Number of baseline essential skills (min 1).
        - 'bloat_ratio': Numerator / Denominator.
        - 'net_expansion': Number of skills scheduled beyond the baseline count.
    """
    unique_scheduled = set(scheduled_skills)
    numerator = len(unique_scheduled)
    denominator = max(1, len(baseline_essential_skills))
    ratio = numerator / denominator
    net_expansion = max(0, numerator - len(baseline_essential_skills))

    return {
        "numerator": numerator,
        "denominator": denominator,
        "bloat_ratio": round(ratio, 4),
        "net_expansion": net_expansion,
    }


def recommendation_leakage_rate(
    recommended: Sequence[str],
    known_irrelevant: Set[str],
) -> dict[str, Any]:
    """Quantifies recommendation leakage of negative-control or cross-domain irrelevant skills.

    Leakage occurs when skills deliberately identified as irrelevant to the target role
    (e.g., agricultural, culinary, healthcare skills in a software engineering pathway)
    are recommended or scheduled.

    Args:
        recommended: Sequence of recommended or scheduled skill identifiers.
        known_irrelevant: Set of known irrelevant negative control skill identifiers.

    Returns:
        Dictionary containing:
        - 'leaked_items': List of irrelevant skills present in recommendations.
        - 'leakage_count': Count of leaked items.
        - 'leakage_rate': leakage_count / max(1, len(recommended)).
    """
    if not recommended or not known_irrelevant:
        return {
            "leaked_items": [],
            "leakage_count": 0,
            "leakage_rate": 0.0,
        }

    unique_rec = deduplicate_preserve_order(recommended)
    leaked = [item for item in unique_rec if item in known_irrelevant]
    rate = len(leaked) / len(unique_rec)

    return {
        "leaked_items": leaked,
        "leakage_count": len(leaked),
        "leakage_rate": round(rate, 4),
    }
