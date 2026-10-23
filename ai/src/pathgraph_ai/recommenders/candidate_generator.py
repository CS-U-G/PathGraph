"""Multi-Archetype Candidate Pathway Generator.

Generates diverse, pedagogical, 100% valid topological learning paths:
1. Foundational-First: Prioritizes prerequisite depth, foundational skills, and unblocking impact
2. Goal-First: Prioritizes directly required high-importance target occupation skills
3. Quick-Wins: Prioritizes small mastery gaps and high readiness for rapid progress
4. Domain-Clustered: Minimizes cognitive context-switching across taxonomy clusters
5. Bounded Beam Search: Priority-guided combinatorial exploration of topological sequence space

Every generated path satisfies the strict DAG invariant:
    forall (u, v) in E, index(u) < index(v)
Prerequisite violation rate is strictly 0.00%.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx

from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.domain.graph_metrics import PrerequisiteGraphMetrics, get_graph_metrics
from pathgraph_ai.domain.learner import LearningPreferences
from pathgraph_ai.recommenders.gap_analyser import SkillGap


@dataclass
class CandidatePath:
    """A generated candidate sequence of skill IDs with generator metadata."""
    strategy: str
    skill_ids: list[str]
    generation_time_ms: float


# Alias for backward compatibility
PathwayCandidate = CandidatePath


class CandidatePathwayGenerator:
    """Generates multiple diverse topological candidate sequences for an induced skill subgraph."""

    def __init__(
        self,
        subgraph: nx.DiGraph,
        target_occupation_uri: str | None = None,
        graph: CompetencyGraph | None = None,
        gaps: list[SkillGap] | None = None,
        preferences: LearningPreferences | None = None,
        metrics: PrerequisiteGraphMetrics | None = None,
    ) -> None:
        self.subgraph = subgraph
        self.target_occupation_uri = target_occupation_uri
        self.graph = graph
        self.gaps = gaps or []
        self.preferences = preferences or LearningPreferences()
        self.metrics = metrics or get_graph_metrics(subgraph)

        # Lookup tables
        self._gap_map = {g.skill_uri: g.gap for g in self.gaps}
        self._priority_map = {g.skill_uri: g.priority_weight for g in self.gaps}
        self._weight_map = {g.skill_uri: g.requirement_weight for g in self.gaps}
        self._relation_map = {g.skill_uri: g.relation_type for g in self.gaps}

    def _verify_topological_invariant(self, path: list[str]) -> bool:
        """Assert that every edge (u -> v) in subgraph satisfies index(u) < index(v)."""
        pos = {sid: idx for idx, sid in enumerate(path)}
        for u, v in self.subgraph.edges:
            if u in pos and v in pos:
                if pos[u] >= pos[v]:
                    return False
        return True

    def generate_all(self, beam_width: int = 5, time_budget_ms: float = 20.0) -> list[CandidatePath]:
        """Generate all distinct candidate pathways."""
        candidates: list[CandidatePath] = []
        strategies = [
            self.generate_foundational_first,
            self.generate_goal_first,
            self.generate_quick_wins,
            self.generate_domain_clustered,
            lambda: self.generate_beam_search(beam_width=beam_width, time_budget_ms=time_budget_ms),
        ]

        seen_paths: set[tuple[str, ...]] = set()
        for fn in strategies:
            cand = fn()
            tpl = tuple(cand.skill_ids)
            if tpl not in seen_paths and len(cand.skill_ids) == len(self.subgraph.nodes):
                seen_paths.add(tpl)
                candidates.append(cand)

        # Fallback to standard topological sort if none generated
        if not candidates:
            t0 = time.perf_counter()
            std = list(nx.topological_sort(self.subgraph))
            candidates.append(
                CandidatePath(
                    strategy="topological_fallback",
                    skill_ids=std,
                    generation_time_ms=(time.perf_counter() - t0) * 1000,
                )
            )

        return candidates

    def generate_foundational_first(self) -> CandidatePath:
        """Prioritizes lowest prerequisite depth, foundational skills, and high unblocking impact."""
        t0 = time.perf_counter()

        def key_fn(u: str) -> tuple[int, float, float, str]:
            depth = self.metrics.get_depth(u)
            impact = self.metrics.get_impact(u)
            unblocks = self.metrics.get_unblocks_count(u)
            return (depth, -impact, -unblocks, u)

        path = self._topological_priority_traversal(key_fn)
        elapsed = (time.perf_counter() - t0) * 1000
        assert self._verify_topological_invariant(path), "Foundational-first violated DAG invariant"
        return CandidatePath(strategy="foundational_first", skill_ids=path, generation_time_ms=elapsed)

    def generate_goal_first(self) -> CandidatePath:
        """Prioritizes highest requirement weight, gap priority, and target relevance."""
        t0 = time.perf_counter()

        def key_fn(u: str) -> tuple[float, float, int, str]:
            weight = self._weight_map.get(u, 0.4)
            p_weight = self._priority_map.get(u, 0.0)
            depth = self.metrics.get_depth(u)
            return (-weight, -p_weight, depth, u)

        path = self._topological_priority_traversal(key_fn)
        elapsed = (time.perf_counter() - t0) * 1000
        assert self._verify_topological_invariant(path), "Goal-first violated DAG invariant"
        return CandidatePath(strategy="target_goal_first", skill_ids=path, generation_time_ms=elapsed)

    def generate_quick_wins(self) -> CandidatePath:
        """Prioritizes small mastery gaps and high readiness for rapid early progress."""
        t0 = time.perf_counter()

        def key_fn(u: str) -> tuple[float, int, str]:
            gap = self._gap_map.get(u, 0.5)
            depth = self.metrics.get_depth(u)
            return (gap, depth, u)

        path = self._topological_priority_traversal(key_fn)
        elapsed = (time.perf_counter() - t0) * 1000
        assert self._verify_topological_invariant(path), "Quick-wins violated DAG invariant"
        return CandidatePath(strategy="quick_wins", skill_ids=path, generation_time_ms=elapsed)

    def generate_domain_clustered(self) -> CandidatePath:
        """Minimizes cognitive context-switching across taxonomy clusters."""
        t0 = time.perf_counter()
        in_degree = {u: self.subgraph.in_degree(u) for u in self.subgraph.nodes}
        ready: list[str] = [u for u in self.subgraph.nodes if in_degree[u] == 0]
        path: list[str] = []
        current_cluster: str | None = None

        def get_cluster(u: str) -> str:
            if self.graph:
                clusters = self.graph.get_taxonomy_cluster(u)
                if clusters:
                    return clusters[0]
            node_data = self.subgraph.nodes.get(u, {})
            return node_data.get("category", "")

        while ready:
            def sort_key(u: str) -> tuple[int, int, str]:
                c = get_cluster(u)
                match = 0 if current_cluster and c == current_cluster else 1
                depth = self.metrics.get_depth(u)
                return (match, depth, u)

            ready.sort(key=sort_key)
            u = ready.pop(0)
            path.append(u)
            current_cluster = get_cluster(u) or current_cluster

            for v in self.subgraph.successors(u):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    ready.append(v)

        elapsed = (time.perf_counter() - t0) * 1000
        assert self._verify_topological_invariant(path), "Domain-clustered violated DAG invariant"
        return CandidatePath(strategy="domain_clustered", skill_ids=path, generation_time_ms=elapsed)

    def generate_beam_search(self, beam_width: int = 5, time_budget_ms: float = 20.0) -> CandidatePath:
        """Explores topological sequence space with bounded beam search."""
        t0 = time.perf_counter()
        deadline = t0 + (time_budget_ms / 1000.0)
        n_nodes = len(self.subgraph.nodes)

        initial_in_degree = {u: self.subgraph.in_degree(u) for u in self.subgraph.nodes}
        initial_ready = frozenset(u for u in self.subgraph.nodes if initial_in_degree[u] == 0)

        # Beam item: (score, path_tuple, in_degrees_dict, ready_set)
        beam: list[tuple[float, tuple[str, ...], dict[str, int], frozenset[str]]] = [
            (0.0, (), dict(initial_in_degree), initial_ready)
        ]

        for _ in range(n_nodes):
            if time.perf_counter() > deadline:
                break

            candidates_pool: list[tuple[float, tuple[str, ...], dict[str, int], frozenset[str]]] = []

            for cum_score, cur_path, in_deg, ready_set in beam:
                for cand in ready_set:
                    weight = self._weight_map.get(cand, 0.4)
                    impact = self.metrics.get_impact(cand)

                    # Immediate step reward: high weight + unblocking impact
                    step_reward = (weight * 2.0) + (impact * 0.5)

                    new_path = cur_path + (cand,)
                    new_in_deg = dict(in_deg)
                    new_ready = set(ready_set)
                    new_ready.remove(cand)

                    for v in self.subgraph.successors(cand):
                        new_in_deg[v] -= 1
                        if new_in_deg[v] == 0:
                            new_ready.add(v)

                    candidates_pool.append((cum_score + step_reward, new_path, new_in_deg, frozenset(new_ready)))

            if not candidates_pool:
                break

            candidates_pool.sort(key=lambda x: -x[0])
            beam = candidates_pool[:beam_width]

        beam.sort(key=lambda x: (len(x[1]) == n_nodes, x[0]), reverse=True)
        best_path = list(beam[0][1])

        if len(best_path) < n_nodes:
            remaining = set(self.subgraph.nodes) - set(best_path)
            sub_rem = self.subgraph.subgraph(remaining)
            best_path.extend(list(nx.topological_sort(sub_rem)))

        elapsed = (time.perf_counter() - t0) * 1000
        assert self._verify_topological_invariant(best_path), "Beam search violated DAG invariant"
        return CandidatePath(strategy="bounded_beam_search", skill_ids=best_path, generation_time_ms=elapsed)

    def _topological_priority_traversal(self, key_fn: Callable[[str], Any]) -> list[str]:
        """Generic priority-guided Kahn's topological sort."""
        in_degree = {u: self.subgraph.in_degree(u) for u in self.subgraph.nodes}
        ready = [u for u in self.subgraph.nodes if in_degree[u] == 0]
        path: list[str] = []

        while ready:
            ready.sort(key=key_fn)
            u = ready.pop(0)
            path.append(u)
            for v in self.subgraph.successors(u):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    ready.append(v)

        return path
