"""Multi-Objective Pathway Scorer.

Evaluates candidate pathways along 5 normalized pedagogical objectives in [0.0, 1.0]:
- f_goal: Early target-occupation alignment (discounted cumulative requirement weight)
- f_gap: Mastery gap reduction efficiency
- f_smooth: Cognitive prerequisite depth & difficulty gradient smoothness
- f_time: Pacing efficiency and time budget compliance
- f_cluster: Taxonomy and domain context continuity
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.domain.learner import LearningPreferences, PacePreference
from pathgraph_ai.recommenders.gap_analyser import SkillGap


@dataclass
class PathwayObjectiveScores:
    """Normalized multi-objective scores in [0.0, 1.0]."""
    goal_alignment: float
    gap_reduction: float
    difficulty_smoothness: float
    time_budget_fit: float
    category_continuity: float
    composite_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "goal_alignment": self.goal_alignment,
            "gap_reduction": self.gap_reduction,
            "difficulty_smoothness": self.difficulty_smoothness,
            "time_budget_fit": self.time_budget_fit,
            "category_continuity": self.category_continuity,
            "composite_score": self.composite_score,
        }


class MultiObjectivePathScorer:
    """Evaluates and scores candidate topological pathways."""

    def __init__(
        self,
        subgraph: nx.DiGraph,
        target_occupation_uri: str | None = None,
        graph: CompetencyGraph | None = None,
        gaps: list[SkillGap] | None = None,
        preferences: LearningPreferences | None = None,
        discount_factor: float = 0.95,
    ) -> None:
        self.subgraph = subgraph
        self.target_occupation_uri = target_occupation_uri
        self.graph = graph
        self.preferences = preferences or LearningPreferences()
        self.discount_factor = discount_factor

        self._gap_map = {g.skill_uri: g.gap for g in (gaps or [])}
        self._weight_map = {g.skill_uri: g.requirement_weight for g in (gaps or [])}

    def score_path(self, skill_ids: list[str]) -> PathwayObjectiveScores:
        """Calculate normalized objective scores and weighted composite score for a sequence."""
        m = len(skill_ids)
        if m == 0:
            return PathwayObjectiveScores(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        # 1. Goal Alignment (Discounted cumulative target requirement weight)
        f_goal = self._calc_goal_alignment(skill_ids)

        # 2. Gap Reduction Efficiency
        f_gap = self._calc_gap_reduction(skill_ids)

        # 3. Difficulty Smoothness
        f_smooth = self._calc_difficulty_smoothness(skill_ids)

        # 4. Time Budget Fit
        f_time = self._calc_time_fit(skill_ids)

        # 5. Category Continuity
        f_cluster = self._calc_category_continuity(skill_ids)

        # Weighted Composite Score
        weights = self._get_weights()
        composite = (
            weights["goal"] * f_goal
            + weights["gap"] * f_gap
            + weights["smooth"] * f_smooth
            + weights["time"] * f_time
            + weights["cluster"] * f_cluster
        )

        return PathwayObjectiveScores(
            goal_alignment=round(f_goal, 4),
            gap_reduction=round(f_gap, 4),
            difficulty_smoothness=round(f_smooth, 4),
            time_budget_fit=round(f_time, 4),
            category_continuity=round(f_cluster, 4),
            composite_score=round(composite, 4),
        )

    def _calc_goal_alignment(self, skill_ids: list[str]) -> float:
        discount_sum = 0.0
        weighted_score = 0.0
        for i, sid in enumerate(skill_ids):
            gamma = self.discount_factor ** i
            weight = self._weight_map.get(sid, 0.4)  # 1.0 for essential, 0.4 for optional, default 0.4
            weighted_score += gamma * min(1.0, weight)
            discount_sum += gamma
        return weighted_score / discount_sum if discount_sum > 0 else 0.0

    def _calc_gap_reduction(self, skill_ids: list[str]) -> float:
        if not self._gap_map:
            return 1.0
        max_gap = max(self._gap_map.values(), default=1.0) or 1.0
        discount_sum = 0.0
        weighted_score = 0.0
        for i, sid in enumerate(skill_ids):
            gamma = self.discount_factor ** i
            raw_gap = self._gap_map.get(sid, 0.0) / max_gap
            weighted_score += gamma * raw_gap
            discount_sum += gamma
        return min(1.0, weighted_score / discount_sum) if discount_sum > 0 else 0.0

    def _calc_difficulty_smoothness(self, skill_ids: list[str]) -> float:
        m = len(skill_ids)
        if m <= 1:
            return 1.0
        total_jump = 0
        for i in range(m - 1):
            if self.graph:
                d1 = self.graph.get_prerequisite_depth(skill_ids[i])
                d2 = self.graph.get_prerequisite_depth(skill_ids[i + 1])
            else:
                d1 = int(self.subgraph.nodes.get(skill_ids[i], {}).get("difficulty", 1))
                d2 = int(self.subgraph.nodes.get(skill_ids[i + 1], {}).get("difficulty", 1))
            total_jump += abs(d2 - d1)
        mean_jump = total_jump / (m - 1)
        return max(0.0, min(1.0, 1.0 - (mean_jump / 4.0)))

    def _calc_time_fit(self, skill_ids: list[str]) -> float:
        m = len(skill_ids)
        max_hours = self.preferences.max_total_hours
        # Estimate ~15h per skill unit
        total_hours = m * 15
        if max_hours is not None and max_hours > 0:
            if total_hours <= max_hours:
                return 1.0
            excess = total_hours - max_hours
            return max(0.2, 1.0 - (excess / max_hours))

        weekly = self.preferences.weekly_hours_budget
        estimated_weeks = total_hours / weekly if weekly > 0 else 10
        if estimated_weeks <= 24:
            return 1.0
        return max(0.4, 1.0 - ((estimated_weeks - 24) / 50.0))

    def _calc_category_continuity(self, skill_ids: list[str]) -> float:
        m = len(skill_ids)
        if m <= 1:
            return 1.0
        same_cat_count = 0

        def get_cat(sid: str) -> str:
            if self.graph:
                clusters = self.graph.get_taxonomy_cluster(sid)
                if clusters:
                    return clusters[0]
            return self.subgraph.nodes.get(sid, {}).get("category", "")

        for i in range(m - 1):
            c1 = get_cat(skill_ids[i])
            c2 = get_cat(skill_ids[i + 1])
            if c1 and c1 == c2:
                same_cat_count += 1
        return same_cat_count / (m - 1)

    def _get_weights(self) -> dict[str, float]:
        pref = self.preferences.pace_preference
        if pref == PacePreference.FOUNDATIONAL_FIRST:
            return {"goal": 0.20, "gap": 0.20, "smooth": 0.35, "time": 0.15, "cluster": 0.10}
        elif pref == PacePreference.FAST_TRACK:
            return {"goal": 0.40, "gap": 0.20, "smooth": 0.10, "time": 0.25, "cluster": 0.05}
        elif pref == PacePreference.DOMAIN_FOCUSED:
            return {"goal": 0.20, "gap": 0.15, "smooth": 0.15, "time": 0.15, "cluster": 0.35}
        # BALANCED
        return {"goal": 0.30, "gap": 0.25, "smooth": 0.20, "time": 0.15, "cluster": 0.10}


# Alias for backward compatibility
Scorer = MultiObjectivePathScorer

