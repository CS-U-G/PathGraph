"""Deterministic competency gap analyser grounded in ESCO occupation requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pathgraph_ai.domain.competency_graph import CompetencyGraph, SkillRequirement
from pathgraph_ai.settings import settings


@dataclass(frozen=True)
class LearnerCompetency:
    """Proficiency and evidence record for a learner's skill."""

    skill_uri: str
    proficiency: float | None = None  # None indicates unassessed / unknown
    evidence_source: str = "self_reported"  # 'assessment' | 'project' | 'self_reported' | 'assumed'
    evidence_confidence: float = 0.5


@dataclass(frozen=True)
class SkillGap:
    """Evaluated gap for a required skill."""

    occupation_uri: str
    skill_uri: str
    preferred_label: str
    relation_type: str  # 'essential' | 'optional'
    requirement_weight: float  # r (1.0 or 0.4)
    target_proficiency: float  # t (0.75 or 0.50)
    learner_proficiency: float | None  # p
    evidence_confidence: float  # q
    gap: float  # max(0, t - p)
    priority_weight: float  # G = r * gap * q * impact, normalised to [0, 1]
    unblocking_impact: float  # 1.0 + ln(1 + |transitive_descendants|)
    status: str  # 'satisfied' | 'partially_satisfied' | 'missing_or_below_target' | 'unassessed'
    is_unmet: bool  # True if requires learning or assessment


class GapAnalyser:
    """Computes proficiency deficits, priority weights, and skill statuses."""

    def __init__(
        self,
        graph: CompetencyGraph,
        essential_target: float = settings.essential_target_proficiency,
        optional_target: float = settings.optional_target_proficiency,
        essential_weight: float = settings.essential_requirement_weight,
        optional_weight: float = settings.optional_requirement_weight,
        include_unassessed: bool = True,
        use_unblocking_impact: bool = True,
    ) -> None:
        self.graph = graph
        self.essential_target = essential_target
        self.optional_target = optional_target
        self.essential_weight = essential_weight
        self.optional_weight = optional_weight
        self.include_unassessed = include_unassessed
        self.use_unblocking_impact = use_unblocking_impact

    def get_evidence_confidence(self, source: str) -> float:
        """Maps an evidence source to its configured confidence score."""
        s = source.lower().strip()
        if s == "assessment":
            return settings.assessed_evidence_confidence
        elif s == "project":
            return settings.project_evidence_confidence
        elif s == "self_reported":
            return settings.self_reported_evidence_confidence
        elif s == "assumed":
            return settings.assumed_evidence_confidence
        return 0.50

    def analyse_target(
        self,
        occupation_uri: str,
        learner_skills: dict[str, LearnerCompetency],
    ) -> list[SkillGap]:
        """Analyses all requirements of the target occupation against learner competencies."""
        requirements = self.graph.get_occupation_requirements(occupation_uri)
        gaps: list[SkillGap] = []

        # The maximum possible impact factor across all skills in this occupation
        # is bounded by the skill with the most descendants; use it to normalise.
        # Impact_cap is computed once per call to avoid repeated graph traversals.
        if self.use_unblocking_impact:
            req_uris = [req.skill_uri for req in requirements]
            impact_values = [self.graph.get_unblocking_impact(u) for u in req_uris] or [1.0]
            impact_cap = max(impact_values)
        else:
            impact_cap = 1.0

        # Find max possible priority weight for normalisation
        max_possible_g = self.essential_weight * 1.0 * 1.0 * impact_cap

        for req in requirements:
            s_uri = req.skill_uri
            skill_info = self.graph.get_skill(s_uri) or {}
            lbl = skill_info.get("preferredLabel", s_uri)

            r = self.essential_weight if req.relation_type == "essential" else self.optional_weight
            t = self.essential_target if req.relation_type == "essential" else self.optional_target

            learner_rec = learner_skills.get(s_uri)
            if learner_rec is None or learner_rec.proficiency is None:
                # Unassessed skill: proficiency is unknown
                p = None
                q = settings.assumed_evidence_confidence
                raw_gap = t
                status = "unassessed"
                is_unmet = self.include_unassessed
            else:
                p = min(max(float(learner_rec.proficiency), 0.0), 1.0)
                q = learner_rec.evidence_confidence
                raw_gap = max(0.0, t - p)

                if raw_gap <= 1e-4:
                    status = "satisfied"
                    is_unmet = False
                elif p > 0.0:
                    status = "partially_satisfied"
                    is_unmet = True
                else:
                    status = "missing_or_below_target"
                    is_unmet = True

            # Unblocking impact centrality: 1.0 + ln(1 + |transitive_descendants|)
            impact = self.graph.get_unblocking_impact(s_uri) if self.use_unblocking_impact else 1.0

            # G = r * gap * q * impact, normalised to [0, 1]
            g_raw = r * raw_gap * q * impact
            g_norm = min(max(round(g_raw / max_possible_g, 4), 0.0), 1.0) if max_possible_g > 0 else 0.0

            gaps.append(
                SkillGap(
                    occupation_uri=occupation_uri,
                    skill_uri=s_uri,
                    preferred_label=lbl,
                    relation_type=req.relation_type,
                    requirement_weight=round(r, 4),
                    target_proficiency=round(t, 4),
                    learner_proficiency=round(p, 4) if p is not None else None,
                    evidence_confidence=round(q, 4),
                    gap=round(raw_gap, 4),
                    priority_weight=g_norm,
                    unblocking_impact=round(impact, 4),
                    status=status,
                    is_unmet=is_unmet,
                )
            )

        return gaps
