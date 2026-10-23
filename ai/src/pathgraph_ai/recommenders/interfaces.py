"""Interfaces and pluggable abstractions for PathGraph AI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


from pathgraph_ai.learner.bkt import BKTModel, BKTParameters


class ProficiencyUpdater(ABC):
    """Interface for updating learner skill proficiency based on interaction outcomes."""

    @abstractmethod
    def update(
        self,
        current_proficiencies: dict[str, float],
        skill_uri: str,
        outcome: bool,  # True = correct / success, False = incorrect / failure
    ) -> dict[str, float]:
        """Returns updated proficiency map after the interaction."""
        ...


@dataclass
class BKTProficiencyUpdater(ProficiencyUpdater):
    """Bayesian Knowledge Tracing (BKT) updater backed by BKTModel.

    Standard BKT parameters:
    - P(L0): Prior probability of mastery (default: current proficiency)
    - P(T): Transition probability (learning rate)
    - P(G): Guess probability (correct response despite non-mastery)
    - P(S): Slip probability (incorrect response despite mastery)
    """

    p_transit: float = 0.15
    p_guess: float = 0.20
    p_slip: float = 0.10
    bkt_model: BKTModel | None = None

    def __post_init__(self) -> None:
        if self.bkt_model is None:
            params = BKTParameters(
                p_t=self.p_transit,
                p_g=self.p_guess,
                p_s=self.p_slip,
            )
            self.bkt_model = BKTModel(params=params)

    def update(
        self,
        current_proficiencies: dict[str, float],
        skill_uri: str,
        outcome: bool,
    ) -> dict[str, float]:
        updated = dict(current_proficiencies)
        p_l = updated.get(skill_uri, 0.10)
        assert self.bkt_model is not None
        new_mastery, _ = self.bkt_model.update(
            current_mastery=p_l,
            evidence=outcome,
            evidence_count=0,
        )
        updated[skill_uri] = round(min(max(new_mastery, 0.0), 1.0), 4)
        return updated


class FeedbackStore(ABC):
    """Interface for logging learner path interactions and ratings."""

    @abstractmethod
    def record_feedback(
        self,
        recommendation_id: str,
        learner_id: str | None,
        feedback_type: str,
        payload: dict[str, Any],
    ) -> None:
        ...


class NoOpFeedbackStore(FeedbackStore):
    """No-op feedback store for stateless Phase 3-4 execution."""

    def record_feedback(
        self,
        recommendation_id: str,
        learner_id: str | None,
        feedback_type: str,
        payload: dict[str, Any],
    ) -> None:
        pass


@dataclass
class UserPreferences:
    """User preferences for recommendation filtering and path tailoring."""

    max_study_hours_per_week: float = 10.0
    preferred_language: str = "en"
    max_budget: float | None = None
    learning_resource_types: list[str] = field(default_factory=lambda: ["interactive", "video", "text"])


class PreferenceProvider(ABC):
    """Interface for loading user preferences."""

    @abstractmethod
    def get_preferences(self, learner_id: str | None = None) -> UserPreferences:
        ...


class DefaultPreferenceProvider(PreferenceProvider):
    """Returns default system preferences."""

    def get_preferences(self, learner_id: str | None = None) -> UserPreferences:
        return UserPreferences()
