"""ESCO Knowledge Graph Builder, Validator, and Query Interface for PathGraph.

Constructs a bipartite / heterogeneous knowledge graph linking Occupations and Skills
using verified ESCO URIs and relations.

Important note on prerequisites and hierarchy:
1. ESCO relationship types in 'occupationSkillRelations' indicate 'essential' or
   'optional' requirements for an occupation.
2. In 'skillSkillRelations', relationship types ('essential' | 'optional') represent
   competence-knowledge associations and co-occurrences between skills, NOT verified
   pedagogical or educational prerequisite order.
3. In 'skillsHierarchy', the 4-level taxonomy (Levels 0-3) represents categorization
   and broader/narrower classification groups, NOT educational prerequisites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pathgraph_ai.pipelines.data_loader import ESCODataLoader, VALID_RELATION_TYPES

# Lazy/optional dependency for NetworkX
try:
    import networkx as nx
except ImportError:
    nx = None

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Node and Edge Data Structures
# -----------------------------------------------------------------------------


@dataclass
class OccupationNode:
    """Represents an occupation node in the ESCO knowledge graph."""

    uri: str
    label: str
    isco_group: str | None = None
    code: str | None = None
    description: str | None = None
    node_type: str = "occupation"
    alt_labels: str | None = None


@dataclass
class SkillNode:
    """Represents a skill or competence node in the ESCO knowledge graph."""

    uri: str
    label: str
    skill_type: str | None = None  # 'knowledge' | 'skill/competence'
    reuse_level: str | None = None
    description: str | None = None
    node_type: str = "skill"
    alt_labels: str | None = None


@dataclass
class HierarchyNode:
    """Represents a taxonomy group node from skillsHierarchy (Level 0-3)."""

    uri: str
    label: str
    level: int  # 0, 1, 2, or 3
    code: str | None = None
    description: str | None = None
    scope_note: str | None = None
    node_type: str = "skill_group"


@dataclass
class OccupationSkillRelation:
    """Represents a directed edge from an occupation to a required skill."""

    occupation_uri: str
    skill_uri: str
    relation_type: str  # 'essential' | 'optional'
    skill_type: str | None = None  # 'knowledge' | 'skill/competence'
    skill_label: str | None = None
    occupation_label: str | None = None
    edge_type: str = "occupation_skill"


# Retain SkillRelation name for backwards compatibility
SkillRelation = OccupationSkillRelation


@dataclass
class SkillSkillRelation:
    """Represents a directed association edge between two skills in ESCO.

    Note: These represent competence-knowledge association or co-occurrence,
    NOT verified pedagogical prerequisite order.
    """

    original_skill_uri: str
    related_skill_uri: str
    relation_type: str  # 'essential' | 'optional'
    original_skill_type: str | None = None
    related_skill_type: str | None = None
    edge_type: str = "skill_skill"


@dataclass
class HierarchyEdge:
    """Represents a broader-to-narrower taxonomy edge between hierarchy levels.

    Note: Represents taxonomic categorization, NOT educational prerequisites.
    """

    parent_uri: str
    child_uri: str
    relation_type: str = "broader_narrower"
    edge_type: str = "hierarchy"


@dataclass
class GraphValidationReport:
    """Diagnostic report capturing graph construction metrics and anomaly detection."""

    occupation_node_count: int = 0
    skill_node_count: int = 0
    hierarchy_node_count: int = 0
    total_node_count: int = 0
    occupation_skill_edge_count: int = 0
    occupation_skill_by_relation: dict[str, int] = field(default_factory=dict)
    skill_skill_edge_count: int = 0
    skill_skill_by_relation: dict[str, int] = field(default_factory=dict)
    hierarchy_edge_count: int = 0
    total_edge_count: int = 0
    missing_node_references: list[str] = field(default_factory=list)
    duplicate_edges: list[str] = field(default_factory=list)
    unexpected_relation_types: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True

    @property
    def status_label(self) -> str:
        """Returns PASS, WARN, or FAIL status label."""
        if not self.is_valid or self.errors:
            return "FAIL"
        if self.warnings or self.missing_node_references or self.duplicate_edges or self.unexpected_relation_types:
            return "WARN"
        return "PASS"


# -----------------------------------------------------------------------------
# Knowledge Graph Core Class
# -----------------------------------------------------------------------------


class ESCOKnowledgeGraph:
    """Manages the graph representation of ESCO occupations, skills, and hierarchies.

    Provides pure-Python in-memory graph representation, anomaly detection,
    relational queries, and optional NetworkX export.
    """

    def __init__(self) -> None:
        # Nodes
        self.occupations: dict[str, OccupationNode] = {}
        self.skills: dict[str, SkillNode] = {}
        self.hierarchy_nodes: dict[str, HierarchyNode] = {}

        # Edges: Occupation -> Skill
        self.occupation_to_skills: dict[str, list[tuple[str, str]]] = {}
        self.skill_to_occupations: dict[str, list[tuple[str, str]]] = {}
        self.occupation_skill_relations: list[OccupationSkillRelation] = []

        # Edges: Skill -> Skill
        self.skill_associations: dict[str, set[str]] = {}
        self.skill_to_skills: dict[str, list[tuple[str, str]]] = {}
        self.skill_skill_relations: list[SkillSkillRelation] = []

        # Edges: Hierarchy
        self.hierarchy_parent_to_children: dict[str, list[str]] = {}
        self.hierarchy_child_to_parents: dict[str, list[str]] = {}
        self.hierarchy_edges: list[HierarchyEdge] = []

        # Construction report
        self.build_report: GraphValidationReport | None = None

    def add_occupation(
        self,
        uri: str,
        label: str,
        isco_group: str | None = None,
        code: str | None = None,
        description: str | None = None,
        alt_labels: str | None = None,
    ) -> OccupationNode:
        """Adds an occupation node."""
        node = OccupationNode(
            uri=uri,
            label=label,
            isco_group=isco_group,
            code=code,
            description=description,
            alt_labels=alt_labels,
        )
        self.occupations[uri] = node
        if uri not in self.occupation_to_skills:
            self.occupation_to_skills[uri] = []
        return node

    def add_skill(
        self,
        uri: str,
        label: str,
        skill_type: str | None = None,
        reuse_level: str | None = None,
        description: str | None = None,
        alt_labels: str | None = None,
    ) -> SkillNode:
        """Adds a skill node."""
        node = SkillNode(
            uri=uri,
            label=label,
            skill_type=skill_type,
            reuse_level=reuse_level,
            description=description,
            alt_labels=alt_labels,
        )
        self.skills[uri] = node
        if uri not in self.skill_to_occupations:
            self.skill_to_occupations[uri] = []
        if uri not in self.skill_associations:
            self.skill_associations[uri] = set()
        if uri not in self.skill_to_skills:
            self.skill_to_skills[uri] = []
        return node

    def add_hierarchy_node(
        self,
        uri: str,
        label: str,
        level: int,
        code: str | None = None,
        description: str | None = None,
        scope_note: str | None = None,
    ) -> HierarchyNode:
        """Adds a skills hierarchy category node."""
        node = HierarchyNode(
            uri=uri,
            label=label,
            level=level,
            code=code,
            description=description,
            scope_note=scope_note,
        )
        self.hierarchy_nodes[uri] = node
        if uri not in self.hierarchy_parent_to_children:
            self.hierarchy_parent_to_children[uri] = []
        if uri not in self.hierarchy_child_to_parents:
            self.hierarchy_child_to_parents[uri] = []
        return node

    def add_occupation_skill_relation(
        self,
        occupation_uri: str,
        skill_uri: str,
        relation_type: str,
        skill_type: str | None = None,
        skill_label: str | None = None,
        occupation_label: str | None = None,
    ) -> OccupationSkillRelation:
        """Links an occupation to a skill with relation_type ('essential' | 'optional')."""
        rel = OccupationSkillRelation(
            occupation_uri=occupation_uri,
            skill_uri=skill_uri,
            relation_type=relation_type,
            skill_type=skill_type,
            skill_label=skill_label,
            occupation_label=occupation_label,
        )
        self.occupation_skill_relations.append(rel)

        if occupation_uri not in self.occupation_to_skills:
            self.occupation_to_skills[occupation_uri] = []
        self.occupation_to_skills[occupation_uri].append((skill_uri, relation_type))

        if skill_uri not in self.skill_to_occupations:
            self.skill_to_occupations[skill_uri] = []
        self.skill_to_occupations[skill_uri].append((occupation_uri, relation_type))
        return rel

    def add_skill_skill_relation(
        self,
        original_skill_uri: str,
        related_skill_uri: str,
        relation_type: str = "optional",
        original_skill_type: str | None = None,
        related_skill_type: str | None = None,
    ) -> SkillSkillRelation:
        """Records an ESCO inter-skill relation/association with explicit relationType."""
        rel = SkillSkillRelation(
            original_skill_uri=original_skill_uri,
            related_skill_uri=related_skill_uri,
            relation_type=relation_type,
            original_skill_type=original_skill_type,
            related_skill_type=related_skill_type,
        )
        self.skill_skill_relations.append(rel)

        if original_skill_uri not in self.skill_associations:
            self.skill_associations[original_skill_uri] = set()
        self.skill_associations[original_skill_uri].add(related_skill_uri)

        if original_skill_uri not in self.skill_to_skills:
            self.skill_to_skills[original_skill_uri] = []
        self.skill_to_skills[original_skill_uri].append((related_skill_uri, relation_type))
        return rel

    def add_skill_association(
        self,
        original_skill_uri: str,
        related_skill_uri: str,
        relation_type: str = "optional",
        original_skill_type: str | None = None,
        related_skill_type: str | None = None,
    ) -> SkillSkillRelation:
        """Alias for add_skill_skill_relation for backwards compatibility."""
        return self.add_skill_skill_relation(
            original_skill_uri=original_skill_uri,
            related_skill_uri=related_skill_uri,
            relation_type=relation_type,
            original_skill_type=original_skill_type,
            related_skill_type=related_skill_type,
        )

    def add_hierarchy_edge(
        self,
        parent_uri: str,
        child_uri: str,
        relation_type: str = "broader_narrower",
    ) -> HierarchyEdge:
        """Adds a parent-child taxonomy edge between hierarchy category levels."""
        edge = HierarchyEdge(
            parent_uri=parent_uri,
            child_uri=child_uri,
            relation_type=relation_type,
        )
        self.hierarchy_edges.append(edge)

        if parent_uri not in self.hierarchy_parent_to_children:
            self.hierarchy_parent_to_children[parent_uri] = []
        self.hierarchy_parent_to_children[parent_uri].append(child_uri)

        if child_uri not in self.hierarchy_child_to_parents:
            self.hierarchy_child_to_parents[child_uri] = []
        self.hierarchy_child_to_parents[child_uri].append(parent_uri)
        return edge

    def build_from_loader(
        self,
        loader: ESCODataLoader | None = None,
        load_hierarchy: bool = True,
        strict: bool = False,
    ) -> GraphValidationReport:
        """Loads and integrates ESCO dataset into the knowledge graph with full validation."""
        if loader is None:
            loader = ESCODataLoader()

        report = GraphValidationReport()

        occ_records = loader.load_occupations(deduplicate=True)
        for row in occ_records:
            uri = row.get("conceptUri", "").strip()
            label = row.get("preferredLabel", "").strip()
            if not uri:
                report.errors.append("Empty conceptUri in occupation record.")
                continue
            self.add_occupation(
                uri=uri,
                label=label,
                isco_group=row.get("iscoGroup"),
                code=row.get("code"),
                description=row.get("description"),
                alt_labels=row.get("altLabels"),
            )
        report.occupation_node_count = len(self.occupations)

        skill_records = loader.load_skills(deduplicate=True)
        for row in skill_records:
            uri = row.get("conceptUri", "").strip()
            label = row.get("preferredLabel", "").strip()
            if not uri:
                report.errors.append("Empty conceptUri in skill record.")
                continue
            self.add_skill(
                uri=uri,
                label=label,
                skill_type=row.get("skillType"),
                reuse_level=row.get("reuseLevel"),
                description=row.get("description"),
                alt_labels=row.get("altLabels"),
            )
        report.skill_node_count = len(self.skills)

        occ_skill_records = loader.load_occupation_skill_relations()
        seen_occ_skill_edges: set[tuple[str, str]] = set()

        for row_idx, row in enumerate(occ_skill_records, start=1):
            occ_uri = row.get("occupationUri", "").strip()
            skill_uri = row.get("skillUri", "").strip()
            rel_type = row.get("relationType", "").strip()
            skill_type = row.get("skillType")
            skill_label = row.get("skillLabel")
            occ_label = row.get("occupationLabel")

            if occ_uri not in self.occupations:
                msg = f"Referenced occupationUri '{occ_uri}' at row {row_idx} not found in occupation nodes."
                report.missing_node_references.append(msg)
                report.errors.append(msg)
            if skill_uri not in self.skills:
                msg = f"Referenced skillUri '{skill_uri}' at row {row_idx} not found in skill nodes."
                report.missing_node_references.append(msg)
                report.errors.append(msg)

            if rel_type not in VALID_RELATION_TYPES:
                msg = f"Unexpected occupationSkillRelation relationType '{rel_type}' at row {row_idx}: ({occ_uri}, {skill_uri})"
                report.unexpected_relation_types.append(msg)
                report.errors.append(msg)

            edge_key = (occ_uri, skill_uri)
            if edge_key in seen_occ_skill_edges:
                report.duplicate_edges.append(f"Duplicate occupation-skill edge at row {row_idx}: {edge_key}")
            else:
                seen_occ_skill_edges.add(edge_key)

            self.add_occupation_skill_relation(
                occupation_uri=occ_uri,
                skill_uri=skill_uri,
                relation_type=rel_type,
                skill_type=skill_type,
                skill_label=skill_label,
                occupation_label=occ_label,
            )
            report.occupation_skill_by_relation[rel_type] = (
                report.occupation_skill_by_relation.get(rel_type, 0) + 1
            )

        report.occupation_skill_edge_count = len(self.occupation_skill_relations)

        skill_skill_records = loader.load_skill_skill_relations()
        seen_skill_skill_edges: set[tuple[str, str]] = set()

        for row_idx, row in enumerate(skill_skill_records, start=1):
            orig_uri = row.get("originalSkillUri", "").strip()
            rel_uri = row.get("relatedSkillUri", "").strip()
            rel_type = row.get("relationType", "optional").strip()
            orig_type = row.get("originalSkillType")
            rel_skill_type = row.get("relatedSkillType")

            if orig_uri not in self.skills:
                msg = f"Referenced originalSkillUri '{orig_uri}' at row {row_idx} not found in skill nodes."
                report.missing_node_references.append(msg)
                report.errors.append(msg)
            if rel_uri not in self.skills:
                msg = f"Referenced relatedSkillUri '{rel_uri}' at row {row_idx} not found in skill nodes."
                report.missing_node_references.append(msg)
                report.errors.append(msg)

            if rel_type not in VALID_RELATION_TYPES:
                msg = f"Unexpected skillSkillRelation relationType '{rel_type}' at row {row_idx}: ({orig_uri}, {rel_uri})"
                report.unexpected_relation_types.append(msg)
                report.errors.append(msg)

            edge_key = (orig_uri, rel_uri)
            if edge_key in seen_skill_skill_edges:
                report.duplicate_edges.append(f"Duplicate skill-skill edge at row {row_idx}: {edge_key}")
            else:
                seen_skill_skill_edges.add(edge_key)

            self.add_skill_skill_relation(
                original_skill_uri=orig_uri,
                related_skill_uri=rel_uri,
                relation_type=rel_type,
                original_skill_type=orig_type,
                related_skill_type=rel_skill_type,
            )
            report.skill_skill_by_relation[rel_type] = (
                report.skill_skill_by_relation.get(rel_type, 0) + 1
            )

        report.skill_skill_edge_count = len(self.skill_skill_relations)

        if load_hierarchy:
            hierarchy_records = loader.load_skills_hierarchy()
            seen_hierarchy_edges: set[tuple[str, str]] = set()

            for row in hierarchy_records:
                path: list[tuple[int, str]] = []
                for lvl in range(4):
                    u = (row.get(f"Level {lvl} URI") or "").strip()
                    term = (row.get(f"Level {lvl} preferred term") or "").strip()
                    code = (row.get(f"Level {lvl} code") or "").strip()
                    desc = (row.get("Description") or "").strip() if lvl == 3 else None
                    scope = (row.get("Scope note") or "").strip() if lvl == 3 else None

                    if u:
                        if u not in self.hierarchy_nodes:
                            self.add_hierarchy_node(
                                uri=u,
                                label=term,
                                level=lvl,
                                code=code if code else None,
                                description=desc if desc else None,
                                scope_note=scope if scope else None,
                            )
                        path.append((lvl, u))

                for i in range(len(path) - 1):
                    p_uri = path[i][1]
                    c_uri = path[i + 1][1]
                    h_edge = (p_uri, c_uri)
                    if h_edge not in seen_hierarchy_edges:
                        seen_hierarchy_edges.add(h_edge)
                        self.add_hierarchy_edge(parent_uri=p_uri, child_uri=c_uri)

            report.hierarchy_node_count = len(self.hierarchy_nodes)
            report.hierarchy_edge_count = len(self.hierarchy_edges)

        report.total_node_count = (
            report.occupation_node_count
            + report.skill_node_count
            + report.hierarchy_node_count
        )
        report.total_edge_count = (
            report.occupation_skill_edge_count
            + report.skill_skill_edge_count
            + report.hierarchy_edge_count
        )

        report.is_valid = len(report.errors) == 0
        self.build_report = report

        if strict and not report.is_valid:
            raise ValueError(
                f"Knowledge graph build validation failed in strict mode with {len(report.errors)} errors."
            )

        return report

    @classmethod
    def from_loader(
        cls,
        loader: ESCODataLoader | None = None,
        load_hierarchy: bool = True,
        strict: bool = False,
    ) -> ESCOKnowledgeGraph:
        """Factory method to construct and populate an ESCOKnowledgeGraph."""
        kg = cls()
        kg.build_from_loader(loader=loader, load_hierarchy=load_hierarchy, strict=strict)
        return kg

    def get_node(self, uri: str) -> OccupationNode | SkillNode | HierarchyNode | None:
        """Retrieves any node (occupation, skill, or hierarchy group) by URI."""
        if uri in self.occupations:
            return self.occupations[uri]
        if uri in self.skills:
            return self.skills[uri]
        if uri in self.hierarchy_nodes:
            return self.hierarchy_nodes[uri]
        return None

    def get_skills_for_occupation(
        self, occupation_uri: str, relation_filter: str | None = None
    ) -> list[tuple[str, str]]:
        """Retrieves (skill_uri, relation_type) pairs required for an occupation."""
        relations = self.occupation_to_skills.get(occupation_uri, [])
        if relation_filter:
            filter_lower = relation_filter.lower()
            return [(s, r) for s, r in relations if r.lower() == filter_lower]
        return relations

    def get_occupation_skills_detailed(
        self, occupation_uri: str, relation_filter: str | None = None
    ) -> list[OccupationSkillRelation]:
        """Retrieves detailed OccupationSkillRelation objects for an occupation."""
        results = [
            rel for rel in self.occupation_skill_relations
            if rel.occupation_uri == occupation_uri
        ]
        if relation_filter:
            filter_lower = relation_filter.lower()
            return [rel for rel in results if rel.relation_type.lower() == filter_lower]
        return results

    def get_occupations_for_skill(
        self, skill_uri: str, relation_filter: str | None = None
    ) -> list[tuple[str, str]]:
        """Retrieves (occupation_uri, relation_type) pairs that require a skill."""
        relations = self.skill_to_occupations.get(skill_uri, [])
        if relation_filter:
            filter_lower = relation_filter.lower()
            return [(o, r) for o, r in relations if r.lower() == filter_lower]
        return relations

    def get_related_skills(
        self, skill_uri: str, relation_filter: str | None = None
    ) -> list[tuple[str, str]]:
        """Retrieves (related_skill_uri, relation_type) associated with a skill."""
        relations = self.skill_to_skills.get(skill_uri, [])
        if relation_filter:
            filter_lower = relation_filter.lower()
            return [(s, r) for s, r in relations if r.lower() == filter_lower]
        return relations

    def get_skill_skill_relations_detailed(
        self, skill_uri: str, relation_filter: str | None = None
    ) -> list[SkillSkillRelation]:
        """Retrieves detailed SkillSkillRelation objects originated by a skill."""
        results = [
            rel for rel in self.skill_skill_relations
            if rel.original_skill_uri == skill_uri
        ]
        if relation_filter:
            filter_lower = relation_filter.lower()
            return [rel for rel in results if rel.relation_type.lower() == filter_lower]
        return results

    def get_hierarchy_children(self, parent_uri: str) -> list[str]:
        """Returns direct child category URIs in the skills hierarchy."""
        return self.hierarchy_parent_to_children.get(parent_uri, [])

    def get_hierarchy_parents(self, child_uri: str) -> list[str]:
        """Returns direct parent category URIs in the skills hierarchy."""
        return self.hierarchy_child_to_parents.get(child_uri, [])

    @property
    def num_occupations(self) -> int:
        return len(self.occupations)

    @property
    def num_skills(self) -> int:
        return len(self.skills)

    @property
    def num_hierarchy_nodes(self) -> int:
        return len(self.hierarchy_nodes)

    @property
    def num_nodes(self) -> int:
        return self.num_occupations + self.num_skills + self.num_hierarchy_nodes

    @property
    def num_edges(self) -> int:
        return (
            len(self.occupation_skill_relations)
            + len(self.skill_skill_relations)
            + len(self.hierarchy_edges)
        )

    def summary(self) -> dict[str, Any]:
        """Returns a summary dictionary of graph dimensions and relation breakdowns."""
        occ_skill_by_rel: dict[str, int] = {}
        for rel in self.occupation_skill_relations:
            occ_skill_by_rel[rel.relation_type] = occ_skill_by_rel.get(rel.relation_type, 0) + 1

        skill_skill_by_rel: dict[str, int] = {}
        for rel in self.skill_skill_relations:
            skill_skill_by_rel[rel.relation_type] = skill_skill_by_rel.get(rel.relation_type, 0) + 1

        return {
            "num_occupations": self.num_occupations,
            "num_skills": self.num_skills,
            "num_hierarchy_nodes": self.num_hierarchy_nodes,
            "total_nodes": self.num_nodes,
            "occupation_skill_edges": len(self.occupation_skill_relations),
            "occupation_skill_by_relation": occ_skill_by_rel,
            "skill_skill_edges": len(self.skill_skill_relations),
            "skill_skill_by_relation": skill_skill_by_rel,
            "hierarchy_edges": len(self.hierarchy_edges),
            "total_edges": self.num_edges,
        }

    def to_networkx(self, include_hierarchy: bool = False) -> Any:
        """Builds and returns a NetworkX DiGraph representation if networkx is installed."""
        if nx is None:
            raise ImportError(
                "NetworkX is required to build a NetworkX graph. Please install networkx."
            )

        graph = nx.DiGraph()

        for occ_uri, occ in self.occupations.items():
            graph.add_node(
                occ_uri,
                type="occupation",
                node_type="occupation",
                label=occ.label,
                isco_group=occ.isco_group,
                code=occ.code,
            )

        for skill_uri, skill in self.skills.items():
            graph.add_node(
                skill_uri,
                type="skill",
                node_type="skill",
                label=skill.label,
                skill_type=skill.skill_type,
                reuse_level=skill.reuse_level,
            )

        for rel in self.occupation_skill_relations:
            graph.add_edge(
                rel.occupation_uri,
                rel.skill_uri,
                relation=rel.relation_type,
                relation_type=rel.relation_type,
                edge_type="occupation_skill",
                skill_type=rel.skill_type,
            )

        for rel in self.skill_skill_relations:
            graph.add_edge(
                rel.original_skill_uri,
                rel.related_skill_uri,
                relation=rel.relation_type,
                relation_type=rel.relation_type,
                edge_type="skill_skill",
                original_skill_type=rel.original_skill_type,
                related_skill_type=rel.related_skill_type,
            )

        if include_hierarchy:
            for h_uri, h_node in self.hierarchy_nodes.items():
                graph.add_node(
                    h_uri,
                    type="skill_group",
                    node_type="skill_group",
                    label=h_node.label,
                    level=h_node.level,
                    code=h_node.code,
                )
            for h_edge in self.hierarchy_edges:
                graph.add_edge(
                    h_edge.parent_uri,
                    h_edge.child_uri,
                    relation=h_edge.relation_type,
                    relation_type=h_edge.relation_type,
                    edge_type="hierarchy",
                )

        return graph

    def __repr__(self) -> str:
        return (
            f"<ESCOKnowledgeGraph: {self.num_occupations} occupations, "
            f"{self.num_skills} skills, {self.num_hierarchy_nodes} hierarchy groups | "
            f"{len(self.occupation_skill_relations)} occ-skill edges, "
            f"{len(self.skill_skill_relations)} skill-skill edges>"
        )
