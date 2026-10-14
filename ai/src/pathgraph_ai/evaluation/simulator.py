"""Item Response Theory (IRT) learner simulator for curriculum evaluation (R7).

NOTE: This simulator encodes pedagogical assumptions (e.g. prerequisite mastery
scaling learning gains) and serves as an offline test harness for strategy comparison,
regression, and sensitivity testing. It is never evidence of real student learning.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pathgraph_ai.api.schemas import LearnerProfile, LearnerSkill, RecommendationRequest, RecommendationTarget
from pathgraph_ai.domain.competency_graph import CompetencyGraph


class LearnerArchetype(str, Enum):
    """Synthetic learner archetypes for simulation experiments."""

    NOVICE = "novice"
    CAREER_SWITCHER = "career_switcher"
    NEAR_COMPLETE = "near_complete"
    NOISY_SELF_REPORT = "noisy_self_report"


@dataclass
class SimulatedLearner:
    """Synthetic learner with true latent proficiency and explicit self-reported claims."""

    learner_id: str
    archetype: LearnerArchetype
    latent_proficiencies: dict[str, float]  # True unobserved mastery in [0.0, 1.0]
    reported_proficiencies: dict[str, float]  # Claimed/observed scores in [0.0, 1.0]

    def to_recommendation_request(self, target_occupation_id: str) -> RecommendationRequest:
        """Converts reported competencies into a contract-valid RecommendationRequest."""
        skills = [
            LearnerSkill(skill_id=uri, label=uri, proficiency=prof)
            for uri, prof in self.reported_proficiencies.items()
        ]
        return RecommendationRequest(
            learner=LearnerProfile(
                skills=skills,
                target=RecommendationTarget(occupation_id=target_occupation_id),
            )
        )


@dataclass
class SimulationOutcome:
    """Results from simulating a learner progressing along a recommended learning path."""

    learner_id: str
    archetype: LearnerArchetype
    path_length: int
    final_target_proficiency: float
    target_satisfaction_rate: float
    steps_with_no_gain: int
    steps_to_full_satisfaction: int | None
    total_learning_gain: float


class LearnerSimulator:
    """Fast, seeded IRT-style curriculum simulator."""

    def __init__(
        self,
        graph: CompetencyGraph,
        guess_prob: float = 0.25,
        base_learning_gain: float = 0.35,
        temperature: float = 0.20,
        scale_by_prerequisites: bool = True,
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.guess_prob = guess_prob
        self.base_learning_gain = base_learning_gain
        self.temperature = temperature
        self.scale_by_prerequisites = scale_by_prerequisites
        self.rng = random.Random(seed)

    def generate_learner(
        self,
        target_occupation_uri: str,
        archetype: LearnerArchetype,
        learner_id: str | None = None,
    ) -> SimulatedLearner:
        """Generates a seeded synthetic learner profile."""
        lid = learner_id or f"learner-{self.rng.randint(10000, 99999)}"
        requirements = self.graph.get_occupation_requirements(target_occupation_uri)
        target_skill_uris = [r.skill_uri for r in requirements]

        latent: dict[str, float] = {}
        reported: dict[str, float] = {}

        if archetype == LearnerArchetype.NOVICE:
            # Low baseline across all skills
            for s in target_skill_uris:
                val = round(self.rng.uniform(0.0, 0.15), 3)
                latent[s] = val
                reported[s] = val

        elif archetype == LearnerArchetype.CAREER_SWITCHER:
            # Transferable skills satisfied, technical core partial
            for idx, s in enumerate(target_skill_uris):
                if idx % 2 == 0:
                    val = round(self.rng.uniform(0.40, 0.70), 3)
                else:
                    val = round(self.rng.uniform(0.05, 0.25), 3)
                latent[s] = val
                reported[s] = val

        elif archetype == LearnerArchetype.NEAR_COMPLETE:
            # Almost all satisfied (>= 0.75), exactly 2 gaps
            for idx, s in enumerate(target_skill_uris):
                if idx < 2:
                    val = round(self.rng.uniform(0.20, 0.40), 3)
                else:
                    val = round(self.rng.uniform(0.75, 0.95), 3)
                latent[s] = val
                reported[s] = val

        elif archetype == LearnerArchetype.NOISY_SELF_REPORT:
            # True mastery is moderate, self-report is inflated/noisy
            for s in target_skill_uris:
                true_val = round(self.rng.uniform(0.10, 0.45), 3)
                # Over-estimate by 0.2 - 0.4
                noise = self.rng.uniform(0.20, 0.40)
                rep_val = min(1.0, round(true_val + noise, 3))
                latent[s] = true_val
                reported[s] = rep_val

        return SimulatedLearner(
            learner_id=lid,
            archetype=archetype,
            latent_proficiencies=latent,
            reported_proficiencies=reported,
        )

    def simulate_path_execution(
        self,
        learner: SimulatedLearner,
        recommended_skill_uris: list[str],
        target_occupation_uri: str,
        target_proficiency: float = 0.75,
    ) -> SimulationOutcome:
        """Simulates step-by-step learning along the recommended path."""
        current_state = dict(learner.latent_proficiencies)
        target_requirements = self.graph.get_occupation_requirements(target_occupation_uri)
        essential_target_uris = [
            r.skill_uri for r in target_requirements if r.relation_type == "essential"
        ]

        steps_with_no_gain = 0
        total_gain = 0.0
        step_when_satisfied: int | None = None

        for step_idx, s_uri in enumerate(recommended_skill_uris, start=1):
            current_p = current_state.get(s_uri, 0.0)

            # Determine prerequisite mastery factor
            if self.scale_by_prerequisites:
                direct_prereqs = self.graph.get_prerequisites(s_uri, transitive=False)
                if direct_prereqs:
                    prereq_masteries = [current_state.get(p, 0.0) for p in direct_prereqs]
                    # Penalty if prerequisites are unmet
                    prereq_factor = sum(prereq_masteries) / len(prereq_masteries)
                    prereq_multiplier = max(0.15, prereq_factor)
                else:
                    prereq_multiplier = 1.0
            else:
                prereq_multiplier = 1.0

            # Diminishing returns gain function
            headroom = max(0.0, 1.0 - current_p)
            gain = round(self.base_learning_gain * headroom * prereq_multiplier, 4)

            if gain < 0.01:
                steps_with_no_gain += 1

            new_p = min(1.0, current_p + gain)
            current_state[s_uri] = new_p
            total_gain += gain

            # Check if all essential target requirements are met
            if step_when_satisfied is None:
                all_met = all(
                    current_state.get(t, 0.0) >= target_proficiency
                    for t in essential_target_uris
                )
                if all_met:
                    step_when_satisfied = step_idx

        # Calculate final metrics over essential target skills
        target_profs = [current_state.get(t, 0.0) for t in essential_target_uris]
        mean_final = sum(target_profs) / len(target_profs) if target_profs else 0.0
        met_count = sum(1 for p in target_profs if p >= target_proficiency)
        satisfaction_rate = met_count / len(target_profs) if target_profs else 1.0

        return SimulationOutcome(
            learner_id=learner.learner_id,
            archetype=learner.archetype,
            path_length=len(recommended_skill_uris),
            final_target_proficiency=round(mean_final, 4),
            target_satisfaction_rate=round(satisfaction_rate, 4),
            steps_with_no_gain=steps_with_no_gain,
            steps_to_full_satisfaction=step_when_satisfied,
            total_learning_gain=round(total_gain, 4),
        )
