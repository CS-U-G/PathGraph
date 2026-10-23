"""Learning Path Generator for PathGraph.

Sequences missing skills into a personalized curriculum path based on
verified prerequisite dependencies, essentiality, and target competencies.

Design and Modeling Assumptions:
--------------------------------
1. ESCO dataset relations ('skillsHierarchy' and 'skillSkillRelations') do NOT
   naturally define strict learning prerequisites. 'skillsHierarchy' is a
   broad taxonomy (Levels 0-3), and 'skillSkillRelations' represents associations.
2. Pedagogical prerequisites must be maintained in a verified Dependency Graph (DAG).
3. Where explicit prerequisite edges exist, skills must be ordered topologically.
4. Independent skills without mutual dependencies are ordered prioritizing
   'essential' requirements over 'optional' competencies.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearningStep:
    step_number: int
    skill_uri: str
    skill_label: str
    relation_type: str  # 'essential' | 'optional'
    prerequisites: list[str] = field(default_factory=list)
    estimated_level: str | None = None
    notes: str | None = None


@dataclass
class LearningPath:
    target_occupation_uri: str
    target_occupation_label: str
    steps: list[LearningStep] = field(default_factory=list)
    total_steps: int = 0
    has_prerequisite_cycles: bool = False


class PrerequisiteGraph:
    """Directed Acyclic Graph (DAG) representing verified skill prerequisites."""

    def __init__(self) -> None:
        # skill_uri -> set of skills that directly depend on it
        self.adj: dict[str, set[str]] = defaultdict(set)
        # skill_uri -> set of skills that must be learned before it
        self.prereqs: dict[str, set[str]] = defaultdict(set)

    def add_prerequisite(self, prerequisite_uri: str, target_skill_uri: str) -> None:
        """Adds a directed edge: prerequisite_uri -> target_skill_uri."""
        self.adj[prerequisite_uri].add(target_skill_uri)
        self.prereqs[target_skill_uri].add(prerequisite_uri)

    def get_prerequisites(self, skill_uri: str) -> set[str]:
        """Returns direct prerequisites for a skill."""
        return self.prereqs.get(skill_uri, set())


class LearningPathGenerator:
    """Generates structured learning sequences from missing skills and verified DAGs."""

    def __init__(self, prerequisite_graph: PrerequisiteGraph | None = None) -> None:
        self.prereq_graph = prerequisite_graph or PrerequisiteGraph()

    def generate_path(
        self,
        target_occupation_uri: str,
        target_occupation_label: str,
        missing_skills: list[dict[str, str]],
    ) -> LearningPath:
        """Generates a sequenced learning curriculum.

        Args:
            target_occupation_uri: URI of the target occupation.
            target_occupation_label: Label of the target occupation.
            missing_skills: List of dicts with 'skillUri', 'skillLabel', 'relationType'.

        Returns:
            LearningPath with sequenced learning steps.
        """
        path = LearningPath(
            target_occupation_uri=target_occupation_uri,
            target_occupation_label=target_occupation_label,
        )

        skill_map = {s["skillUri"]: s for s in missing_skills}
        all_uris = set(skill_map.keys())

        # Build in-degree counts strictly within the subset of missing skills
        in_degree: dict[str, int] = {uri: 0 for uri in all_uris}
        subgraph_adj: dict[str, list[str]] = defaultdict(list)

        for uri in all_uris:
            direct_prereqs = self.prereq_graph.get_prerequisites(uri)
            for prereq in direct_prereqs:
                if prereq in all_uris:
                    subgraph_adj[prereq].append(uri)
                    in_degree[uri] += 1

        # Kahn's algorithm for topological sorting, prioritizing essential skills
        queue = deque(
            sorted(
                [u for u in all_uris if in_degree[u] == 0],
                key=lambda u: (
                    0 if skill_map[u].get("relationType") == "essential" else 1,
                    skill_map[u].get("skillLabel", ""),
                ),
            )
        )

        ordered_uris: list[str] = []
        while queue:
            curr = queue.popleft()
            ordered_uris.append(curr)

            for neighbor in subgraph_adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(ordered_uris) < len(all_uris):
            path.has_prerequisite_cycles = True
            remaining = [u for u in all_uris if u not in ordered_uris]
            ordered_uris.extend(remaining)

        # Assemble learning steps
        for step_idx, uri in enumerate(ordered_uris, start=1):
            s_info = skill_map[uri]
            step = LearningStep(
                step_number=step_idx,
                skill_uri=uri,
                skill_label=s_info.get("skillLabel", ""),
                relation_type=s_info.get("relationType", "optional"),
                prerequisites=list(self.prereq_graph.get_prerequisites(uri).intersection(all_uris)),
            )
            path.steps.append(step)

        path.total_steps = len(path.steps)
        return path
