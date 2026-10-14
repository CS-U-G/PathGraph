"""Interfaces and pluggable abstractions for PathGraph AI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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
    """Bayesian Knowledge Tracing (BKT) updater for future database integration.

    Standard BKT parameters:
    - P(L0): Prior probability of mastery (default: current proficiency)
    - P(T): Transition probability (learning rate)
    - P(G): Guess probability (correct response despite non-mastery)
    - P(S): Slip probability (incorrect response despite mastery)
    """

    p_transit: float = 0.15
    p_guess: float = 0.20
    p_slip: float = 0.10

    def update(
        self,
        current_proficiencies: dict[str, float],
        skill_uri: str,
        outcome: bool,
    ) -> dict[str, float]:
        updated = dict(current_proficiencies)
        p_l = updated.get(skill_uri, 0.10)

        # Posterior probability P(L | observation)
        if outcome:
            numerator = p_l * (1.0 - self.p_slip)
            denominator = numerator + (1.0 - p_l) * self.p_guess
        else:
            numerator = p_l * self.p_slip
            denominator = numerator + (1.0 - p_l) * (1.0 - self.p_guess)

        p_l_given_obs = numerator / max(denominator, 1e-6)

        # Transition to next state: P(L_next) = P(L | obs) + (1 - P(L | obs)) * P(T)
        p_l_next = p_l_given_obs + (1.0 - p_l_given_obs) * self.p_transit
        updated[skill_uri] = round(min(max(p_l_next, 0.0), 1.0), 4)
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
