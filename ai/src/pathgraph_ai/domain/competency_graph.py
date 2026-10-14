"""Four-layer Competency Graph architecture for PathGraph AI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from pathgraph_ai.domain.prerequisites import (
    PrerequisiteEdge,
    compute_prerequisite_closure,
    load_prerequisites_csv,
)
from pathgraph_ai.settings import settings


@dataclass(frozen=True)
class SkillRequirement:
    """Occupation requirement for a specific skill."""

    occupation_uri: str
    skill_uri: str
    relation_type: str  # 'essential' | 'optional'
    target_proficiency: float
    requirement_weight: float


class CompetencyGraph:
    """Multi-layer competency and curriculum dependency graph grounded in ESCO.

    Features:
    - 4 distinct NetworkX layers (Requirement, Taxonomy, Association, Prerequisite).
    - Internal integer indexing for graph traversal performance, with canonical URIs at boundaries.
    - Read-only helper queries for learning path recommendations.
    """

    def __init__(self) -> None:
        # Entities & Indexing
        self.occupations: dict[str, dict[str, Any]] = {}
        self.skills: dict[str, dict[str, Any]] = {}
        self.uri_to_int: dict[str, int] = {}
        self.int_to_uri: dict[int, str] = {}

        # 4 Separate Graph Layers
        self.requirement_graph = nx.DiGraph()  # Bipartite: occupation_uri -> skill_uri
        self.taxonomy_graph = nx.DiGraph()  # Skill hierarchy DAG: narrower_uri -> broader_uri
        self.association_graph = nx.Graph()  # Associative relations: skill_uri <-> skill_uri
        self.prerequisite_graph = nx.DiGraph()  # Strict prerequisite DAG: prereq_uri -> target_uri

        # Prerequisite unknown tracking & precomputed depths
        self.unknown_prerequisite_skills: set[str] = set()
        self.prerequisite_depths: dict[str, int] = {}
        self._is_frozen: bool = False

    @classmethod
    def from_processed_data(
        cls,
        processed_dir: Path,
        prerequisites_csv_path: Path,
    ) -> CompetencyGraph:
        """Constructs and populates all 4 graph layers from processed JSON artifacts."""
        graph = cls()
        current_idx = 0

        # 1. Load 96 Tier 1 Occupations
        occs_file = processed_dir / "occupations.json"
        with open(occs_file, "r", encoding="utf-8") as f:
            for occ in json.load(f):
                uri = occ["conceptUri"]
                graph.occupations[uri] = occ
                if uri not in graph.uri_to_int:
                    graph.uri_to_int[uri] = current_idx
                    graph.int_to_uri[current_idx] = uri
                    current_idx += 1

                graph.requirement_graph.add_node(
                    uri,
                    node_id=graph.uri_to_int[uri],
                    node_type="occupation",
                    preferredLabel=occ.get("preferredLabel", ""),
                    iscoGroup=occ.get("iscoGroup", ""),
                )

        # 2. Load 13,939 Unique Skills
        skills_file = processed_dir / "skills.json"
        with open(skills_file, "r", encoding="utf-8") as f:
            for skill in json.load(f):
                uri = skill["conceptUri"]
                graph.skills[uri] = skill
                if uri not in graph.uri_to_int:
                    graph.uri_to_int[uri] = current_idx
                    graph.int_to_uri[current_idx] = uri
                    current_idx += 1

                idx = graph.uri_to_int[uri]
                versatility = float(skill.get("skill_versatility", 0.0))
                lbl = skill.get("preferredLabel", "")

                graph.requirement_graph.add_node(
                    uri,
                    node_id=idx,
                    node_type="skill",
                    preferredLabel=lbl,
                    skill_versatility=versatility,
                )
                graph.taxonomy_graph.add_node(
                    uri,
                    node_id=idx,
                    node_type="skill",
                    preferredLabel=lbl,
                )
                graph.association_graph.add_node(
                    uri,
                    node_id=idx,
                    node_type="skill",
                    preferredLabel=lbl,
                )
                graph.prerequisite_graph.add_node(
                    uri,
                    node_id=idx,
                    node_type="skill",
                    preferredLabel=lbl,
                    skill_versatility=versatility,
                )

        # 3. Layer 1: Occupation-Skill Requirements (G_req)
        req_file = processed_dir / "occupation_skill_edges.json"
        with open(req_file, "r", encoding="utf-8") as f:
            for edge in json.load(f):
                occ_uri = edge["occupationUri"]
                skill_uri = edge["skillUri"]
                rel_type = edge.get("relationType", "optional").strip().lower()

                target_p = (
                    settings.essential_target_proficiency
                    if rel_type == "essential"
                    else settings.optional_target_proficiency
                )
                weight = (
                    settings.essential_requirement_weight
                    if rel_type == "essential"
                    else settings.optional_requirement_weight
                )

                if occ_uri in graph.occupations and skill_uri in graph.skills:
                    graph.requirement_graph.add_edge(
                        occ_uri,
                        skill_uri,
                        relation_type=rel_type,
                        target_proficiency=target_p,
                        requirement_weight=weight,
                    )

        # 4. Layer 2: Taxonomy Hierarchy (G_tax)
        tax_file = processed_dir / "skill_hierarchy.json"
        if tax_file.exists():
            with open(tax_file, "r", encoding="utf-8") as f:
                for edge in json.load(f):
                    concept_uri = edge["conceptUri"]
                    broader_uri = edge["broaderUri"]
                    if broader_uri not in graph.uri_to_int:
                        graph.uri_to_int[broader_uri] = current_idx
                        graph.int_to_uri[current_idx] = broader_uri
                        current_idx += 1

                    graph.taxonomy_graph.add_node(
                        broader_uri,
                        node_id=graph.uri_to_int[broader_uri],
                        node_type=edge.get("broaderType", "SkillGroup"),
                        preferredLabel=edge.get("broaderLabel", ""),
                    )
                    graph.taxonomy_graph.add_edge(
                        concept_uri,
                        broader_uri,
                        concept_type=edge.get("conceptType", ""),
                        broader_type=edge.get("broaderType", ""),
                    )

        # 5. Layer 3: Skill Associations (G_assoc)
        assoc_file = processed_dir / "skill_associations.json"
        if assoc_file.exists():
            with open(assoc_file, "r", encoding="utf-8") as f:
                for edge in json.load(f):
                    u = edge["originalSkillUri"]
                    v = edge["relatedSkillUri"]
                    rel_type = edge.get("relationType", "optional")
                    if u in graph.skills and v in graph.skills:
                        graph.association_graph.add_edge(u, v, relation_type=rel_type)

        # 6. Layer 4: Prerequisite Dependencies (G_prereq)
        if prerequisites_csv_path.exists():
            valid_skill_uris = set(graph.skills.keys())
            prereq_edges = load_prerequisites_csv(prerequisites_csv_path, valid_skill_uris)
            for e in prereq_edges:
                graph.prerequisite_graph.add_edge(
                    e.prerequisite_skill_uri,
                    e.target_skill_uri,
                    relation_type=e.relation_type,
                    provenance=e.provenance,
                    confidence=e.confidence,
                    evidence_source=e.evidence_source,
                    notes=e.notes,
                )

        # 7. Identify skills with unknown prerequisite status
        active_prereq_nodes = {
            n
            for n, d in graph.prerequisite_graph.degree()
            if d > 0
        }
        for s_uri in graph.skills:
            if s_uri not in active_prereq_nodes:
                graph.unknown_prerequisite_skills.add(s_uri)
                graph.prerequisite_graph.nodes[s_uri]["prerequisites_unknown"] = True
            else:
                graph.prerequisite_graph.nodes[s_uri]["prerequisites_unknown"] = False

        # 8. Precompute depths in prerequisite DAG
        if nx.is_directed_acyclic_graph(graph.prerequisite_graph) and active_prereq_nodes:
            sub = graph.prerequisite_graph.subgraph(active_prereq_nodes)
            for n in nx.topological_sort(sub):
                preds = list(sub.predecessors(n))
                if not preds:
                    graph.prerequisite_depths[n] = 0
                else:
                    graph.prerequisite_depths[n] = 1 + max(graph.prerequisite_depths[p] for p in preds)

        graph._is_frozen = True
        return graph

    def get_occupation(self, uri: str) -> dict[str, Any] | None:
        return self.occupations.get(uri)

    def get_skill(self, uri: str) -> dict[str, Any] | None:
        return self.skills.get(uri)

    def get_occupation_requirements(self, occupation_uri: str) -> list[SkillRequirement]:
        """Returns all skill requirements for the specified occupation target."""
        if occupation_uri not in self.requirement_graph:
            return []

        requirements: list[SkillRequirement] = []
        for _, skill_uri, data in self.requirement_graph.out_edges(occupation_uri, data=True):
            requirements.append(
                SkillRequirement(
                    occupation_uri=occupation_uri,
                    skill_uri=skill_uri,
                    relation_type=data.get("relation_type", "optional"),
                    target_proficiency=data.get("target_proficiency", 0.50),
                    requirement_weight=data.get("requirement_weight", 0.40),
                )
            )
        return requirements

    def get_prerequisites(self, skill_uri: str, transitive: bool = False) -> set[str]:
        """Returns direct or transitive ancestors (prerequisites) for a skill."""
        if skill_uri not in self.prerequisite_graph:
            return set()
        if transitive:
            return nx.ancestors(self.prerequisite_graph, skill_uri)
        return set(self.prerequisite_graph.predecessors(skill_uri))

    def get_dependents(self, skill_uri: str, transitive: bool = False) -> set[str]:
        """Returns direct or transitive descendants (dependents) unlocked by a skill."""
        if skill_uri not in self.prerequisite_graph:
            return set()
        if transitive:
            return nx.descendants(self.prerequisite_graph, skill_uri)
        return set(self.prerequisite_graph.successors(skill_uri))

    def get_prerequisite_depth(self, skill_uri: str) -> int:
        """Returns the prerequisite depth of a skill (0 for roots or unknown)."""
        return self.prerequisite_depths.get(skill_uri, 0)

    def get_taxonomy_cluster(self, skill_uri: str) -> list[str]:
        """Returns taxonomy ancestors up to level 0 (SkillGroups) for clustering."""
        if skill_uri not in self.taxonomy_graph:
            return []
        try:
            return list(nx.descendants(self.taxonomy_graph, skill_uri))
        except Exception:
            return []

    def get_taxonomy_siblings(self, skill_uri: str) -> set[str]:
        """Returns skills sharing immediate broader parents in the taxonomy."""
        if skill_uri not in self.taxonomy_graph:
            return set()
        parents = set(self.taxonomy_graph.successors(skill_uri))
        siblings: set[str] = set()
        for p in parents:
            for child in self.taxonomy_graph.predecessors(p):
                if child != skill_uri and child in self.skills:
                    siblings.add(child)
        return siblings

    def get_associations(self, skill_uri: str) -> set[str]:
        """Returns associated skills in the association graph."""
        if skill_uri not in self.association_graph:
            return set()
        return set(self.association_graph.neighbors(skill_uri))

    def is_prerequisite_satisfied(self, skill_uri: str, learner_skill_uris: set[str]) -> bool:
        """Checks if all immediate prerequisites for a skill are present in learner skills."""
        direct_prereqs = self.get_prerequisites(skill_uri, transitive=False)
        if not direct_prereqs:
            return True
        return direct_prereqs.issubset(learner_skill_uris)

    def is_prerequisites_unknown(self, skill_uri: str) -> bool:
        """Returns True if the skill has no documented prerequisite edges."""
        return skill_uri in self.unknown_prerequisite_skills

    def check_prerequisite_acyclicity(self) -> tuple[bool, list[list[str]]]:
        """Verifies that the prerequisite layer is a DAG.

        On failure, returns False and the list of strongly connected components
        with size > 1. Never silently deletes edges.
        """
        is_dag = nx.is_directed_acyclic_graph(self.prerequisite_graph)
        if is_dag:
            return True, []

        sccs = [
            list(c)
            for c in nx.strongly_connected_components(self.prerequisite_graph)
            if len(c) > 1
        ]
        return False, sccs
