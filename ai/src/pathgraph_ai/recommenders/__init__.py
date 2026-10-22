"""Recommenders subpackage for PathGraph AI."""

from pathgraph_ai.recommenders.candidate_generator import (
    CandidatePathwayGenerator,
    PathwayCandidate,
)
from pathgraph_ai.recommenders.career_ranker import CareerRanker
from pathgraph_ai.recommenders.gap_analyser import (
    GapAnalyser,
    LearnerCompetency,
    SkillGap,
)
from pathgraph_ai.recommenders.interfaces import (
    BKTProficiencyUpdater,
    ProficiencyUpdater,
)
from pathgraph_ai.recommenders.pareto import ParetoFrontier
from pathgraph_ai.recommenders.path_engine import (
    OrderingStrategy,
    PathEngine,
)
from pathgraph_ai.recommenders.path_generator import (
    LearningPath,
    LearningPathGenerator,
    LearningStep,
    PrerequisiteGraph,
)
from pathgraph_ai.recommenders.scorer import Scorer
from pathgraph_ai.recommenders.skill_gap import (
    MatchedSkill,
    MissingSkill,
    SemanticCandidate,
    SkillGapAnalyzer,
    SkillGapReport,
)

__all__ = [
    "GapAnalyser",
    "LearnerCompetency",
    "SkillGap",
    "SkillGapAnalyzer",
    "MatchedSkill",
    "MissingSkill",
    "SemanticCandidate",
    "SkillGapReport",
    "PathEngine",
    "OrderingStrategy",
    "LearningPathGenerator",
    "LearningPath",
    "LearningStep",
    "PrerequisiteGraph",
    "CandidatePathwayGenerator",
    "PathwayCandidate",
    "Scorer",
    "ParetoFrontier",
    "CareerRanker",
    "ProficiencyUpdater",
    "BKTProficiencyUpdater",
]
