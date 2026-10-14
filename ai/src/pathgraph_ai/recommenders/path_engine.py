"""Hybrid deterministic learning path recommendation engine (Kahn DAG + priority queue)."""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import networkx as nx

from pathgraph_ai.api.schemas import (
    PathSkillRecommendation,
    ReasonCode,
    RecommendationMetadata,
    RecommendationRequest,
    RecommendationResponse,
    TargetType,
)
from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.explain.reasoning import generate_skill_explanation
from pathgraph_ai.recommenders.gap_analyser import GapAnalyser, LearnerCompetency, SkillGap
from pathgraph_ai.settings import settings


class OrderingStrategy(str, Enum):
    """Named topological tie-breaking ordering strategies."""

    GAP_FIRST = "gap_first"  # Default: higher weighted gap first
    UNLOCK_FIRST = "unlock_first"  # More dependents unlocked first
    VERSATILITY_FIRST = "versatility_first"  # Higher versatility first
    BLOCKING = "blocking"  # Cluster-adjacent: finish taxonomy cluster first
    INTERLEAVED = "interleaved"  # Cluster-alternating: rotate between clusters


class InvalidTargetOccupationError(ValueError):
    """Raised when the requested target occupation does not exist in Tier 1 computing graph."""


class PrerequisiteCycleError(RuntimeError):
    """Raised when an induced prerequisite subgraph contains a cycle."""


class UnsupportedTargetTypeError(ValueError):
    """Raised when target_type is not 'occupation'."""


@dataclass
class ScheduledStep:
    """Internal representation of a sequenced learning milestone."""

    skill_uri: str
    label: str
    is_directly_required: bool
    relation_type: str | None
    gap: float
    relevance: float
    demand: float
    level: int
    prerequisite_depth: int
    taxonomy_cluster: str
    dependent_uri: str | None = None
    prerequisites_unknown: bool = False


class PathEngine:
    """Deterministic, graph-grounded recommendation engine."""

    def __init__(
        self,
        graph: CompetencyGraph,
        gap_analyser: GapAnalyser | None = None,
        default_strategy: OrderingStrategy = OrderingStrategy.GAP_FIRST,
        prereq_satisfaction_threshold: float = 0.50,
        relevance_decay: float = 0.80,
        max_path_length: int = 50,
        max_depth: int = 10,
        enable_soft_fallback: bool = False,
    ) -> None:
        self.graph = graph
        self.gap_analyser = gap_analyser or GapAnalyser(graph)
        self.default_strategy = default_strategy
        self.prereq_satisfaction_threshold = prereq_satisfaction_threshold
        self.relevance_decay = relevance_decay
        self.max_path_length = max_path_length
        self.max_depth = max_depth
        self.enable_soft_fallback = enable_soft_fallback

    def recommend(
        self,
        request: RecommendationRequest,
        unresolved_skills: list[str] | None = None,
        strategy: OrderingStrategy | None = None,
    ) -> RecommendationResponse:
        """Executes the deterministic path generation pipeline."""
        target = request.learner.target

        # 1. Target Validation
        if target.target_type == TargetType.CUSTOM_GOAL:
            raise UnsupportedTargetTypeError(
                "target_type 'custom_goal' is not currently supported in v1.0.0; reserved for v1.1.0."
            )

        occ_uri = target.occupation_id.strip()
        if occ_uri not in self.graph.occupations:
            raise InvalidTargetOccupationError(
                f"Target occupation '{occ_uri}' is not a valid Tier 1 core computing occupation."
            )

        active_strategy = strategy or self.default_strategy
        unresolved = list(unresolved_skills or [])

        # 2. Parse Learner Competencies
        learner_competencies: dict[str, LearnerCompetency] = {}
        satisfied_skill_uris: set[str] = set()

        for s in request.learner.skills:
            uri = s.skill_id.strip()
            # If skill not in graph, add to unresolved
            if uri not in self.graph.skills:
                if s.label and s.label not in unresolved:
                    unresolved.append(s.label)
                continue

            prof = s.proficiency  # defaulted to 0.5 by schema if omitted
            learner_competencies[uri] = LearnerCompetency(
                skill_uri=uri,
                proficiency=prof,
                evidence_source="self_reported",
                evidence_confidence=settings.self_reported_evidence_confidence,
            )
            if prof is not None and prof >= self.prereq_satisfaction_threshold:
                satisfied_skill_uris.add(uri)

        # 3. Gap Analysis
        gaps = self.gap_analyser.analyse_target(occ_uri, learner_competencies)
        gaps_by_uri = {g.skill_uri: g for g in gaps}

        # Identify unmet required skills (missing, partially satisfied, or unassessed)
        unmet_required_uris = {g.skill_uri for g in gaps if g.is_unmet}

        # 4. Prerequisite Closure over Strict Edges
        # Determine stepping stone skills required to unlock unmet skills
        needed_skills: set[str] = set(unmet_required_uris)
        direct_to_prereq_map: dict[str, str] = {}  # prereq_uri -> dependent_uri

        for target_skill in unmet_required_uris:
            ancestors = self.graph.get_prerequisites(target_skill, transitive=True)
            for anc in ancestors:
                # Skip prerequisites the learner already satisfies
                if anc not in satisfied_skill_uris:
                    needed_skills.add(anc)
                    if anc not in direct_to_prereq_map:
                        direct_to_prereq_map[anc] = target_skill

        # 5. Build Induced Prerequisite Subgraph & Cycle Check
        induced_subgraph = nx.DiGraph()
        for s in needed_skills:
            induced_subgraph.add_node(s)

        for u in needed_skills:
            for v in self.graph.get_dependents(u, transitive=False):
                if v in needed_skills:
                    induced_subgraph.add_edge(u, v)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(induced_subgraph):
            sccs = [list(c) for c in nx.strongly_connected_components(induced_subgraph) if len(c) > 1]
            raise PrerequisiteCycleError(
                f"Prerequisite subgraph for target '{occ_uri}' contains cyclic dependencies: {sccs}"
            )

        # 6. Topological Sequencing with Priority Queue (Kahn-style)
        in_degrees = {n: induced_subgraph.in_degree(n) for n in induced_subgraph.nodes()}
        ready_queue: list[tuple[Any, ...]] = []

        # Helper to compute priority key based on strategy
        def make_priority_key(skill_uri: str, cluster_tracker: dict[str, int]) -> tuple[Any, ...]:
            gap_rec = gaps_by_uri.get(skill_uri)
            p_weight = gap_rec.priority_weight if gap_rec else 0.50
            unlocked = len(self.graph.get_dependents(skill_uri, transitive=False))
            skill_info = self.graph.get_skill(skill_uri) or {}
            vers = float(skill_info.get("skill_versatility", 0.0))
            cluster = self._get_primary_cluster(skill_uri)

            if active_strategy == OrderingStrategy.GAP_FIRST:
                return (-p_weight, -unlocked, -vers, skill_uri)
            elif active_strategy == OrderingStrategy.UNLOCK_FIRST:
                return (-unlocked, -p_weight, -vers, skill_uri)
            elif active_strategy == OrderingStrategy.VERSATILITY_FIRST:
                return (-vers, -p_weight, -unlocked, skill_uri)
            elif active_strategy == OrderingStrategy.BLOCKING:
                # Prefer same cluster as last scheduled
                current_cluster = cluster_tracker.get("last_cluster", "")
                is_same_cluster = 0 if (cluster == current_cluster and current_cluster != "") else 1
                return (is_same_cluster, -p_weight, -unlocked, skill_uri)
            elif active_strategy == OrderingStrategy.INTERLEAVED:
                # Prefer different cluster than last scheduled
                current_cluster = cluster_tracker.get("last_cluster", "")
                is_diff_cluster = 0 if (cluster != current_cluster) else 1
                return (is_diff_cluster, -p_weight, -unlocked, skill_uri)
            return (-p_weight, -unlocked, -vers, skill_uri)

        cluster_tracker: dict[str, Any] = {"last_cluster": ""}

        # Initial available skills (in-degree == 0)
        for n, deg in in_degrees.items():
            if deg == 0:
                key = make_priority_key(n, cluster_tracker)
                heapq.heappush(ready_queue, (key, n))

        scheduled_sequence: list[str] = []
        node_levels: dict[str, int] = {}

        while ready_queue and len(scheduled_sequence) < self.max_path_length:
            _, chosen = heapq.heappop(ready_queue)
            scheduled_sequence.append(chosen)

            # Compute longest-path level in scheduled subgraph
            preds = [p for p in induced_subgraph.predecessors(chosen) if p in node_levels]
            level = (max(node_levels[p] for p in preds) + 1) if preds else 0
            node_levels[chosen] = level
            cluster_tracker["last_cluster"] = self._get_primary_cluster(chosen)

            # Decrement in-degree of successors
            for succ in induced_subgraph.successors(chosen):
                in_degrees[succ] -= 1
                if in_degrees[succ] == 0:
                    succ_key = make_priority_key(succ, cluster_tracker)
                    heapq.heappush(ready_queue, (succ_key, succ))

        # Soft fallback (R4, off by default): cluster-adjacent within the same level
        if self.enable_soft_fallback:
            scheduled_sequence = self._apply_soft_fallback(scheduled_sequence, node_levels)

        # 7. Build Output PathSkillRecommendation list
        recommendation_items: list[PathSkillRecommendation] = []
        already_scheduled: set[str] = set(satisfied_skill_uris)

        for priority_idx, s_uri in enumerate(scheduled_sequence, start=1):
            skill_info = self.graph.get_skill(s_uri) or {}
            lbl = skill_info.get("preferredLabel", s_uri)
            vers = float(skill_info.get("skill_versatility", 0.0))
            reuse = skill_info.get("reuseLevel")

            is_direct = s_uri in unmet_required_uris
            gap_rec = gaps_by_uri.get(s_uri)

            if is_direct and gap_rec:
                gap_val = gap_rec.gap
                rel_val = gap_rec.requirement_weight  # 1.0 or 0.4
                rel_type = gap_rec.relation_type
                dep_lbl = None
            else:
                # Stepping stone prerequisite
                gap_val = self.prereq_satisfaction_threshold  # default gap against prerequisite threshold
                dep_uri = direct_to_prereq_map.get(s_uri)
                dep_info = self.graph.get_skill(dep_uri) if dep_uri else {}
                dep_lbl = dep_info.get("preferredLabel") if dep_info else None
                dep_gap = gaps_by_uri.get(dep_uri) if dep_uri else None
                dep_weight = dep_gap.requirement_weight if dep_gap else 1.0
                # Relevance decay: max(r of dependent) * (0.8 ^ distance)
                rel_val = round(dep_weight * self.relevance_decay, 4)
                rel_type = None

            # Prerequisite satisfaction check
            direct_prereqs = self.graph.get_prerequisites(s_uri, transitive=False)
            prereqs_satisfied = direct_prereqs.issubset(already_scheduled)
            already_scheduled.add(s_uri)

            # Templated reason & codes
            reason, codes = generate_skill_explanation(
                skill_uri=s_uri,
                skill_label=lbl,
                is_directly_required=is_direct,
                relation_type=rel_type,
                gap=gap_val,
                dependent_label=dep_lbl,
                all_prereqs_met=prereqs_satisfied,
                reuse_level=reuse,
                relevance=rel_val,
                versatility=vers,
            )

            item = PathSkillRecommendation(
                skill_id=s_uri,
                label=lbl,
                priority=priority_idx,
                gap=round(gap_val, 4),
                prerequisite_satisfied=prereqs_satisfied,
                relevance=round(rel_val, 4),
                demand=round(vers, 4),
                estimated_cost=None,  # Stays null until resource catalog exists
                reason=reason,
                reason_codes=codes,
            )
            recommendation_items.append(item)

        meta = RecommendationMetadata(
            target_occupation_id=occ_uri,
            unresolved_skills=sorted(unresolved),
            generated_at=datetime.now(timezone.utc),
        )

        return RecommendationResponse(
            learning_path=recommendation_items,
            meta=meta,
        )

    def _get_primary_cluster(self, skill_uri: str) -> str:
        """Finds primary broader taxonomy cluster for a skill."""
        clusters = self.graph.get_taxonomy_cluster(skill_uri)
        return clusters[0] if clusters else "unclustered"

    def _apply_soft_fallback(
        self,
        sequence: list[str],
        levels: dict[str, int],
    ) -> list[str]:
        """Reorders skills within the exact same topological level to group clusters (R4)."""
        by_level: dict[int, list[str]] = defaultdict(list)
        for s in sequence:
            by_level[levels[s]].append(s)

        reordered: list[str] = []
        for lvl in sorted(by_level.keys()):
            level_skills = by_level[lvl]
            # Group by taxonomy cluster within same level
            level_skills.sort(key=lambda s: (self._get_primary_cluster(s), s))
            reordered.extend(level_skills)

        return reordered
