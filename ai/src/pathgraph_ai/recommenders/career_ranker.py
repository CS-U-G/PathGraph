"""Deterministic career target ranker grounded in ESCO competency coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.recommenders.gap_analyser import LearnerCompetency
from pathgraph_ai.settings import settings


@dataclass(frozen=True)
class CareerMatchResult:
    """Evaluated match score and coverage for an occupation target."""

    occupation_uri: str
    preferred_label: str
    isco_group: str
    match_score: float | None  # M(o) in [0, 1] or None if insufficient evidence
    coverage: float  # C(o) in [0, 1]
    evidence_skills_count: int
    total_required_skills_count: int
    has_sufficient_evidence: bool


class CareerRanker:
    """Ranks target occupations deterministically without probabilistic or employability claims."""

    def __init__(
        self,
        graph: CompetencyGraph,
        essential_weight: float = settings.essential_requirement_weight,
        optional_weight: float = settings.optional_requirement_weight,
        min_coverage_threshold: float = 0.05,
    ) -> None:
        self.graph = graph
        self.essential_weight = essential_weight
        self.optional_weight = optional_weight
        self.min_coverage_threshold = min_coverage_threshold

    def evaluate_target(
        self,
        occupation_uri: str,
        learner_skills: dict[str, LearnerCompetency],
    ) -> CareerMatchResult:
        """Computes M(o) and C(o) for a single occupation target."""
        occ_data = self.graph.get_occupation(occupation_uri) or {}
        lbl = occ_data.get("preferredLabel", occupation_uri)
        isco = occ_data.get("iscoGroup", "")

        requirements = self.graph.get_occupation_requirements(occupation_uri)
        if not requirements:
            return CareerMatchResult(
                occupation_uri=occupation_uri,
                preferred_label=lbl,
                isco_group=isco,
                match_score=None,
                coverage=0.0,
                evidence_skills_count=0,
                total_required_skills_count=0,
                has_sufficient_evidence=False,
            )

        sum_rq_p = 0.0
        sum_rq = 0.0
        sum_r_evidence = 0.0
        sum_r_total = 0.0
        evidence_count = 0

        for req in requirements:
            r = self.essential_weight if req.relation_type == "essential" else self.optional_weight
            sum_r_total += r

            learner_rec = learner_skills.get(req.skill_uri)
            if learner_rec is not None and learner_rec.proficiency is not None:
                # Skill has measured evidence
                p = min(max(float(learner_rec.proficiency), 0.0), 1.0)
                q = min(max(float(learner_rec.evidence_confidence), 0.0), 1.0)

                sum_rq_p += r * q * p
                sum_rq += r * q
                sum_r_evidence += r
                evidence_count += 1

        coverage = round(sum_r_evidence / max(sum_r_total, 1e-6), 4)

        if sum_rq <= 1e-6 or coverage < self.min_coverage_threshold:
            # Zero denominator or below minimum coverage -> insufficient evidence
            match_score = None
            has_sufficient_evidence = False
        else:
            match_score = round(sum_rq_p / sum_rq, 4)
            has_sufficient_evidence = True

        return CareerMatchResult(
            occupation_uri=occupation_uri,
            preferred_label=lbl,
            isco_group=isco,
            match_score=match_score,
            coverage=coverage,
            evidence_skills_count=evidence_count,
            total_required_skills_count=len(requirements),
            has_sufficient_evidence=has_sufficient_evidence,
        )

    def rank_all_targets(
        self,
        learner_skills: dict[str, LearnerCompetency],
        target_uris: list[str] | None = None,
    ) -> list[CareerMatchResult]:
        """Evaluates and ranks occupations by match score descending, then coverage descending."""
        candidates = target_uris if target_uris is not None else list(self.graph.occupations.keys())
        results: list[CareerMatchResult] = []

        for uri in candidates:
            res = self.evaluate_target(uri, learner_skills)
            results.append(res)

        # Sort: sufficient evidence first, then match_score desc, coverage desc, preferred_label asc
        results.sort(
            key=lambda r: (
                1 if r.has_sufficient_evidence else 0,
                r.match_score if r.match_score is not None else -1.0,
                r.coverage,
                r.preferred_label,
            ),
            reverse=True,
        )

        return results
