"""Comprehensive property-style invariant tests for PathGraph recommendation engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathgraph_ai.api.schemas import (
    LearnerProfile,
    LearnerSkill,
    ReasonCode,
    RecommendationRequest,
    RecommendationTarget,
    TargetType,
)
from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.explain.reasoning import verify_explanation_fidelity
from pathgraph_ai.recommenders.career_ranker import CareerRanker
from pathgraph_ai.recommenders.gap_analyser import GapAnalyser, LearnerCompetency
from pathgraph_ai.recommenders.path_engine import (
    InvalidTargetOccupationError,
    OrderingStrategy,
    PathEngine,
    PrerequisiteCycleError,
    UnsupportedTargetTypeError,
)

AI_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = AI_DIR / "data" / "processed"
SEEDS_DIR = AI_DIR.parent / "knowledge" / "seeds"

SW_DEV_URI = "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1"
WEB_DEV_URI = "http://data.europa.eu/esco/occupation/c40a2919-48a9-40ea-b506-1f34f693496d"
DATA_ENG_URI = "http://data.europa.eu/esco/occupation/2079755f-d809-49e6-8037-4de6180e54c0"


@pytest.fixture(scope="session")
def graph() -> CompetencyGraph:
    return CompetencyGraph.from_processed_data(
        processed_dir=PROCESSED_DIR,
        prerequisites_csv_path=SEEDS_DIR / "prerequisites.csv",
    )


@pytest.fixture(scope="session")
def engine(graph: CompetencyGraph) -> PathEngine:
    return PathEngine(graph)


class TestEngineInvariants:
    """Verifies that all generated paths strictly uphold domain and contract invariants."""

    def test_invariant_prerequisite_admissibility(self, engine: PathEngine, graph: CompetencyGraph) -> None:
        """Every strict prerequisite is scheduled earlier or already satisfied."""
        req = RecommendationRequest(
            learner=LearnerProfile(skills=[], target=RecommendationTarget(occupation_id=SW_DEV_URI))
        )
        resp = engine.recommend(req)
        assert len(resp.learning_path) > 0

        scheduled: set[str] = set()
        for item in resp.learning_path:
            direct_prereqs = graph.get_prerequisites(item.skill_id, transitive=False)
            assert direct_prereqs.issubset(scheduled), (
                f"Prerequisite violation: {item.skill_id} scheduled before {direct_prereqs - scheduled}"
            )
            assert item.prerequisite_satisfied is True
            scheduled.add(item.skill_id)

    def test_invariant_score_bounds(self, engine: PathEngine) -> None:
        """All emitted scores (gap, relevance, demand) must lie in [0, 1]."""
        req = RecommendationRequest(
            learner=LearnerProfile(skills=[], target=RecommendationTarget(occupation_id=WEB_DEV_URI))
        )
        resp = engine.recommend(req)
        for item in resp.learning_path:
            assert 0.0 <= item.gap <= 1.0
            assert 0.0 <= item.relevance <= 1.0
            assert 0.0 <= item.demand <= 1.0
            assert item.estimated_cost is None
            assert item.priority >= 1

    def test_invariant_reason_codes_validity(self, engine: PathEngine) -> None:
        """Every recommendation item contains at least one valid ReasonCode."""
        req = RecommendationRequest(
            learner=LearnerProfile(skills=[], target=RecommendationTarget(occupation_id=DATA_ENG_URI))
        )
        resp = engine.recommend(req)
        valid_codes = set(ReasonCode)
        for item in resp.learning_path:
            assert len(item.reason_codes) >= 1
            for code in item.reason_codes:
                assert code in valid_codes
            assert len(item.reason.strip()) > 0

    def test_invariant_strict_determinism(self, engine: PathEngine) -> None:
        """Identical input yields byte-identical JSON outputs."""
        req = RecommendationRequest(
            learner=LearnerProfile(
                skills=[
                    LearnerSkill(
                        skill_id="http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97",
                        label="computer programming",
                        proficiency=0.8,
                    )
                ],
                target=RecommendationTarget(occupation_id=SW_DEV_URI),
            )
        )
        resp1 = engine.recommend(req)
        resp2 = engine.recommend(req)

        # Freeze generated_at timestamp for byte comparison
        dump1 = resp1.model_dump(mode="json")
        dump2 = resp2.model_dump(mode="json")
        dump1["meta"]["generated_at"] = "2026-10-14T12:00:00Z"
        dump2["meta"]["generated_at"] = "2026-10-14T12:00:00Z"

        assert json.dumps(dump1, sort_keys=True) == json.dumps(dump2, sort_keys=True)

    def test_invariant_unassessed_vs_measured(self, graph: CompetencyGraph) -> None:
        """Unknown or unassessed proficiency is never treated as measured 0.0."""
        analyser = GapAnalyser(graph)
        gaps = analyser.analyse_target(SW_DEV_URI, learner_skills={})
        for g in gaps:
            assert g.status == "unassessed"
            assert g.learner_proficiency is None

    def test_invariant_zero_denominator_handling(self, graph: CompetencyGraph) -> None:
        """Zero evidence in career ranking returns has_sufficient_evidence=False and score=None."""
        ranker = CareerRanker(graph)
        res = ranker.evaluate_target(SW_DEV_URI, learner_skills={})
        assert res.has_sufficient_evidence is False
        assert res.match_score is None
        assert res.coverage == 0.0

    def test_invariant_invalid_target_occupation_rejected(self, engine: PathEngine) -> None:
        """Invalid occupation URI raises InvalidTargetOccupationError."""
        req = RecommendationRequest(
            learner=LearnerProfile(
                skills=[],
                target=RecommendationTarget(occupation_id="http://data.europa.eu/esco/occupation/invalid-id"),
            )
        )
        with pytest.raises(InvalidTargetOccupationError):
            engine.recommend(req)

    def test_invariant_custom_goal_rejected(self, engine: PathEngine) -> None:
        """target_type 'custom_goal' raises UnsupportedTargetTypeError."""
        req = RecommendationRequest(
            learner=LearnerProfile(
                skills=[],
                target=RecommendationTarget(
                    occupation_id=SW_DEV_URI,
                    target_type=TargetType.CUSTOM_GOAL,
                    custom_goal="Become an AI Engineer",
                ),
            )
        )
        with pytest.raises(UnsupportedTargetTypeError):
            engine.recommend(req)

    def test_invariant_soft_fallback_never_violates_prerequisites(
        self, graph: CompetencyGraph
    ) -> None:
        """Soft fallback preserves prerequisite ordering admissibility."""
        engine_fb = PathEngine(graph, enable_soft_fallback=True)
        req = RecommendationRequest(
            learner=LearnerProfile(skills=[], target=RecommendationTarget(occupation_id=SW_DEV_URI))
        )
        resp = engine_fb.recommend(req)

        scheduled: set[str] = set()
        for item in resp.learning_path:
            direct_prereqs = graph.get_prerequisites(item.skill_id, transitive=False)
            assert direct_prereqs.issubset(scheduled)
            scheduled.add(item.skill_id)

    def test_explanation_fidelity_100_percent(
        self, engine: PathEngine, graph: CompetencyGraph
    ) -> None:
        """Every generated recommendation item has 100% reproducible explanation fidelity (R10)."""
        req = RecommendationRequest(
            learner=LearnerProfile(skills=[], target=RecommendationTarget(occupation_id=SW_DEV_URI))
        )
        resp = engine.recommend(req)
        for item in resp.learning_path:
            ok, msg = verify_explanation_fidelity(item.model_dump(mode="json"), graph, SW_DEV_URI)
            assert ok is True, f"Fidelity violation on {item.skill_id}: {msg}"
