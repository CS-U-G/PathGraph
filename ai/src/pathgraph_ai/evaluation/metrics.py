"""Path-level evaluation metrics for curriculum recommendation (R6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.explain.reasoning import verify_explanation_fidelity


@dataclass(frozen=True)
class PathMetrics:
    """Standard evaluation metrics computed over a generated learning path."""

    path_length: int
    parallel_levels_count: int
    prerequisite_violation_rate: float  # Must be 0.0
    target_coverage: float  # Share of required skills covered or satisfied
    redundancy_rate: float  # Share of scheduled skills already satisfied
    max_depth_jump: int  # Smoothness proxy: max consecutive jump in prerequisite depth
    mean_depth_change: float  # Smoothness proxy: mean consecutive change in depth
    explanation_fidelity_rate: float  # Share of items passing explanation verification
    kendall_tau: float | None = None  # Compared to reference path if available
    edit_distance: int | None = None  # Compared to reference path if available


def compute_kendall_tau(seq1: list[str], seq2: list[str]) -> float:
    """Computes normalized Kendall rank correlation tau between two permutations of shared elements."""
    shared = [x for x in seq1 if x in seq2]
    if len(shared) < 2:
        return 1.0

    pos2 = {x: i for i, x in enumerate(seq2)}
    shared_ranks2 = [pos2[x] for x in shared]

    concordant = 0
    discordant = 0
    n = len(shared_ranks2)
    for i in range(n):
        for j in range(i + 1, n):
            if shared_ranks2[i] < shared_ranks2[j]:
                concordant += 1
            elif shared_ranks2[i] > shared_ranks2[j]:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 1.0
    return round((concordant - discordant) / total_pairs, 4)


def compute_edit_distance(seq1: list[str], seq2: list[str]) -> int:
    """Computes Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def evaluate_path(
    path_items: list[dict[str, Any]],
    graph: CompetencyGraph,
    occupation_uri: str,
    satisfied_skill_uris: set[str],
    reference_path: list[str] | None = None,
) -> PathMetrics:
    """Computes comprehensive path-level metrics for a recommended sequence."""
    if not path_items:
        return PathMetrics(
            path_length=0,
            parallel_levels_count=0,
            prerequisite_violation_rate=0.0,
            target_coverage=0.0,
            redundancy_rate=0.0,
            max_depth_jump=0,
            mean_depth_change=0.0,
            explanation_fidelity_rate=1.0,
        )

    scheduled_uris = [item["skill_id"] for item in path_items]
    seen_or_satisfied = set(satisfied_skill_uris)
    violations = 0
    redundant = 0

    for s_uri in scheduled_uris:
        if s_uri in satisfied_skill_uris:
            redundant += 1

        direct_prereqs = graph.get_prerequisites(s_uri, transitive=False)
        if not direct_prereqs.issubset(seen_or_satisfied):
            violations += 1

        seen_or_satisfied.add(s_uri)

    violation_rate = round(violations / len(scheduled_uris), 4)
    redundancy_rate = round(redundant / len(scheduled_uris), 4)

    # Target Coverage
    req_skills = [
        r.skill_uri
        for r in graph.get_occupation_requirements(occupation_uri)
        if r.relation_type == "essential"
    ]
    all_addressed = set(scheduled_uris).union(satisfied_skill_uris)
    covered_count = sum(1 for s in req_skills if s in all_addressed)
    coverage = round(covered_count / len(req_skills), 4) if req_skills else 1.0

    # Difficulty Smoothness (prerequisite depths)
    depths = [graph.get_prerequisite_depth(s) for s in scheduled_uris]
    if len(depths) > 1:
        diffs = [abs(depths[i] - depths[i - 1]) for i in range(1, len(depths))]
        max_jump = max(diffs)
        mean_change = round(sum(diffs) / len(diffs), 4)
    else:
        max_jump = 0
        mean_change = 0.0

    # Parallel Levels Count
    # In induced DAG, calculate max level
    sub = graph.prerequisite_graph.subgraph(scheduled_uris)
    levels: dict[str, int] = {}
    if nx.is_directed_acyclic_graph(sub):
        for n in nx.topological_sort(sub):
            preds = [p for p in sub.predecessors(n) if p in levels]
            levels[n] = (1 + max(levels[p] for p in preds)) if preds else 0
    parallel_levels = (max(levels.values()) + 1) if levels else 1

    # Explanation Fidelity Rate
    fidelity_passes = 0
    for item in path_items:
        ok, _ = verify_explanation_fidelity(item, graph, occupation_uri)
        if ok:
            fidelity_passes += 1
    fidelity_rate = round(fidelity_passes / len(path_items), 4)

    # Reference Path Comparison
    kt: float | None = None
    ed: int | None = None
    if reference_path:
        kt = compute_kendall_tau(scheduled_uris, reference_path)
        ed = compute_edit_distance(scheduled_uris, reference_path)

    return PathMetrics(
        path_length=len(scheduled_uris),
        parallel_levels_count=parallel_levels,
        prerequisite_violation_rate=violation_rate,
        target_coverage=coverage,
        redundancy_rate=redundancy_rate,
        max_depth_jump=max_jump,
        mean_depth_change=mean_change,
        explanation_fidelity_rate=fidelity_rate,
        kendall_tau=kt,
        edit_distance=ed,
    )
