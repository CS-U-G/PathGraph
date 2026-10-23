"""Comprehensive Automated Test Suite for ESCOKnowledgeGraph.

Verifies:
1. Node counts and distinct node types (occupations, skills, hierarchy groups).
2. Edge counts and edge attributes across all relation types.
3. Occupation-skill relationships: essential vs optional attribute preservation.
4. Skill-skill relationships: relation types (optional, essential) and taxonomy semantics.
5. Skills hierarchy: structure and representation without prerequisite interpretation.
6. Detection and reporting of missing node references, duplicate edges, and unexpected relation types.
7. Strict mode vs permissive diagnostic reporting.
8. Query interface and NetworkX export behavior.
"""

from unittest.mock import MagicMock, patch
import pytest

from pathgraph_ai.pipelines.data_loader import ESCODataLoader
from pathgraph_ai.domain.knowledge_graph import (
    ESCOKnowledgeGraph,
    GraphValidationReport,
    HierarchyEdge,
    HierarchyNode,
    OccupationNode,
    OccupationSkillRelation,
    SkillNode,
    SkillSkillRelation,
)


@pytest.fixture(scope="session")
def esco_kg() -> ESCOKnowledgeGraph:
    """Session-scoped fixture providing the fully built ESCO Knowledge Graph."""
    return ESCOKnowledgeGraph.from_loader(load_hierarchy=True, strict=False)


# -----------------------------------------------------------------------------
# 1. Node Counts and Node Types
# -----------------------------------------------------------------------------


def test_node_counts_match_esco_spec(esco_kg: ESCOKnowledgeGraph):
    """Verifies that the graph contains the exact deduplicated ESCO node counts."""
    assert esco_kg.num_occupations == 3039
    assert esco_kg.num_skills == 13939
    assert esco_kg.num_hierarchy_nodes == 640
    assert esco_kg.num_nodes == 3039 + 13939 + 640


def test_distinct_node_types(esco_kg: ESCOKnowledgeGraph):
    """Verifies that occupation, skill, and hierarchy nodes maintain distinct types."""
    for occ_uri, occ_node in esco_kg.occupations.items():
        assert occ_node.node_type == "occupation"
        assert occ_node.uri == occ_uri
        assert isinstance(occ_node.label, str)
        assert len(occ_node.label) > 0

    for skill_uri, skill_node in esco_kg.skills.items():
        assert skill_node.node_type == "skill"
        assert skill_node.uri == skill_uri
        assert isinstance(skill_node.label, str)
        assert len(skill_node.label) > 0

    for h_uri, h_node in esco_kg.hierarchy_nodes.items():
        assert h_node.node_type == "skill_group"
        assert h_node.uri == h_uri
        assert h_node.level in (0, 1, 2, 3)
        assert isinstance(h_node.label, str)

    occ_uris = set(esco_kg.occupations.keys())
    skill_uris = set(esco_kg.skills.keys())
    hierarchy_uris = set(esco_kg.hierarchy_nodes.keys())

    assert len(occ_uris.intersection(skill_uris)) == 0, "Occupation and skill URIs must not overlap"
    assert len(skill_uris.intersection(hierarchy_uris)) == 0, "Skill and hierarchy URIs must not overlap"


def test_node_attribute_richness(esco_kg: ESCOKnowledgeGraph):
    """Verifies that ESCO metadata attributes are preserved on nodes."""
    sample_occ = next(iter(esco_kg.occupations.values()))
    assert hasattr(sample_occ, "isco_group")
    assert hasattr(sample_occ, "code")
    assert hasattr(sample_occ, "description")

    sample_skill = next(iter(esco_kg.skills.values()))
    assert hasattr(sample_skill, "skill_type")
    assert hasattr(sample_skill, "reuse_level")
    assert hasattr(sample_skill, "description")
    assert sample_skill.skill_type in ("knowledge", "skill/competence", None)


# -----------------------------------------------------------------------------
# 2. Edge Counts and Edge Attributes
# -----------------------------------------------------------------------------


def test_occupation_skill_edge_counts_and_relation_types(esco_kg: ESCOKnowledgeGraph):
    """Verifies occupation-skill edge totals and exact relation type breakdown."""
    report = esco_kg.build_report
    assert report is not None

    total_occ_skill_edges = len(esco_kg.occupation_skill_relations)
    assert total_occ_skill_edges == 126051
    assert report.occupation_skill_edge_count == 126051

    assert report.occupation_skill_by_relation["essential"] == 67600
    assert report.occupation_skill_by_relation["optional"] == 58451
    assert (
        report.occupation_skill_by_relation["essential"]
        + report.occupation_skill_by_relation["optional"]
        == 126051
    )


def test_occupation_skill_edge_attribute_preservation(esco_kg: ESCOKnowledgeGraph):
    """Verifies that occupation-skill edges preserve relation_type and metadata."""
    sample_edge = esco_kg.occupation_skill_relations[0]
    assert sample_edge.edge_type == "occupation_skill"
    assert sample_edge.relation_type in ("essential", "optional")
    assert sample_edge.occupation_uri in esco_kg.occupations
    assert sample_edge.skill_uri in esco_kg.skills


def test_skill_skill_edge_counts_and_relation_types(esco_kg: ESCOKnowledgeGraph):
    """Verifies skill-skill relationship totals and actual relation type distribution."""
    report = esco_kg.build_report
    assert report is not None

    total_skill_skill_edges = len(esco_kg.skill_skill_relations)
    assert total_skill_skill_edges == 5818
    assert report.skill_skill_edge_count == 5818

    assert report.skill_skill_by_relation["optional"] == 5629
    assert report.skill_skill_by_relation["essential"] == 189
    assert (
        report.skill_skill_by_relation["optional"]
        + report.skill_skill_by_relation["essential"]
        == 5818
    )


def test_skill_skill_edge_attributes_and_semantics(esco_kg: ESCOKnowledgeGraph):
    """Verifies that skill-skill edges store relation_type, skill types, and associations."""
    sample_edge = esco_kg.skill_skill_relations[0]
    assert sample_edge.edge_type == "skill_skill"
    assert sample_edge.relation_type in ("essential", "optional")
    assert sample_edge.original_skill_uri in esco_kg.skills
    assert sample_edge.related_skill_uri in esco_kg.skills

    orig_uri = sample_edge.original_skill_uri
    assert sample_edge.related_skill_uri in esco_kg.skill_associations[orig_uri]


def test_skills_hierarchy_structure_and_no_prerequisite_interpretation(esco_kg: ESCOKnowledgeGraph):
    """Verifies skills hierarchy tree structure and non-prerequisite classification semantics."""
    assert len(esco_kg.hierarchy_edges) == 636
    assert esco_kg.num_hierarchy_nodes == 640

    for edge in esco_kg.hierarchy_edges:
        assert edge.edge_type == "hierarchy"
        assert edge.relation_type == "broader_narrower"
        assert edge.parent_uri in esco_kg.hierarchy_nodes
        assert edge.child_uri in esco_kg.hierarchy_nodes
        p_node = esco_kg.hierarchy_nodes[edge.parent_uri]
        c_node = esco_kg.hierarchy_nodes[edge.child_uri]
        assert p_node.level < c_node.level

    root_nodes = [node for node in esco_kg.hierarchy_nodes.values() if node.level == 0]
    assert len(root_nodes) == 4
    root_codes = {node.code for node in root_nodes}
    assert root_codes == {"S", "K", "L", "T"}


# -----------------------------------------------------------------------------
# 3. Graph Validation and Anomaly Reporting
# -----------------------------------------------------------------------------


def test_full_esco_dataset_validation_zero_anomalies(esco_kg: ESCOKnowledgeGraph):
    """Verifies that the full official ESCO dataset builds with zero validation anomalies."""
    report = esco_kg.build_report
    assert report is not None
    assert report.is_valid is True
    assert report.status_label == "PASS"
    assert len(report.missing_node_references) == 0
    assert len(report.duplicate_edges) == 0
    assert len(report.unexpected_relation_types) == 0
    assert len(report.errors) == 0


def test_detect_missing_occupation_node_reference():
    """Verifies that an edge referencing a non-existent occupation URI is detected and reported."""
    mock_loader = MagicMock(spec=ESCODataLoader)
    mock_loader.load_occupations.return_value = [
        {"conceptUri": "http://data.europa.eu/esco/occupation/valid-occ", "preferredLabel": "Valid Occ"}
    ]
    mock_loader.load_skills.return_value = [
        {"conceptUri": "http://data.europa.eu/esco/skill/valid-skill", "preferredLabel": "Valid Skill"}
    ]
    mock_loader.load_occupation_skill_relations.return_value = [
        {
            "occupationUri": "http://data.europa.eu/esco/occupation/ghost-occ",
            "skillUri": "http://data.europa.eu/esco/skill/valid-skill",
            "relationType": "essential",
        }
    ]
    mock_loader.load_skill_skill_relations.return_value = []
    mock_loader.load_skills_hierarchy.return_value = []

    kg = ESCOKnowledgeGraph()
    report = kg.build_from_loader(loader=mock_loader, load_hierarchy=False, strict=False)

    assert report.is_valid is False
    assert report.status_label == "FAIL"
    assert len(report.missing_node_references) == 1
    assert "ghost-occ" in report.missing_node_references[0]


def test_detect_missing_skill_node_reference():
    """Verifies that an edge referencing a non-existent skill URI is detected and reported."""
    mock_loader = MagicMock(spec=ESCODataLoader)
    mock_loader.load_occupations.return_value = [
        {"conceptUri": "http://data.europa.eu/esco/occupation/valid-occ", "preferredLabel": "Valid Occ"}
    ]
    mock_loader.load_skills.return_value = [
        {"conceptUri": "http://data.europa.eu/esco/skill/valid-skill", "preferredLabel": "Valid Skill"}
    ]
    mock_loader.load_occupation_skill_relations.return_value = [
        {
            "occupationUri": "http://data.europa.eu/esco/occupation/valid-occ",
            "skillUri": "http://data.europa.eu/esco/skill/ghost-skill",
            "relationType": "essential",
        }
    ]
    mock_loader.load_skill_skill_relations.return_value = []
    mock_loader.load_skills_hierarchy.return_value = []

    kg = ESCOKnowledgeGraph()
    report = kg.build_from_loader(loader=mock_loader, load_hierarchy=False, strict=False)

    assert report.is_valid is False
    assert report.status_label == "FAIL"
    assert len(report.missing_node_references) == 1
    assert "ghost-skill" in report.missing_node_references[0]


def test_detect_duplicate_edges():
    """Verifies that duplicate edges in relationship files are detected and reported."""
    mock_loader = MagicMock(spec=ESCODataLoader)
    mock_loader.load_occupations.return_value = [
        {"conceptUri": "http://data.europa.eu/esco/occupation/occ-1", "preferredLabel": "Occ 1"}
    ]
    mock_loader.load_skills.return_value = [
        {"conceptUri": "http://data.europa.eu/esco/skill/skill-1", "preferredLabel": "Skill 1"},
        {"conceptUri": "http://data.europa.eu/esco/skill/skill-2", "preferredLabel": "Skill 2"},
    ]
    mock_loader.load_occupation_skill_relations.return_value = [
        {"occupationUri": "http://data.europa.eu/esco/occupation/occ-1", "skillUri": "http://data.europa.eu/esco/skill/skill-1", "relationType": "essential"},
        {"occupationUri": "http://data.europa.eu/esco/occupation/occ-1", "skillUri": "http://data.europa.eu/esco/skill/skill-1", "relationType": "essential"},
    ]
    mock_loader.load_skill_skill_relations.return_value = [
        {"originalSkillUri": "http://data.europa.eu/esco/skill/skill-1", "relatedSkillUri": "http://data.europa.eu/esco/skill/skill-2", "relationType": "optional"},
        {"originalSkillUri": "http://data.europa.eu/esco/skill/skill-1", "relatedSkillUri": "http://data.europa.eu/esco/skill/skill-2", "relationType": "optional"},
    ]
    mock_loader.load_skills_hierarchy.return_value = []

    kg = ESCOKnowledgeGraph()
    report = kg.build_from_loader(loader=mock_loader, load_hierarchy=False, strict=False)

    assert len(report.duplicate_edges) == 2
    assert any("Duplicate occupation-skill edge" in d for d in report.duplicate_edges)
    assert any("Duplicate skill-skill edge" in d for d in report.duplicate_edges)


def test_detect_unexpected_relation_type():
    """Verifies that relation types outside ('essential', 'optional') trigger anomalies."""
    mock_loader = MagicMock(spec=ESCODataLoader)
    mock_loader.load_occupations.return_value = [
        {"conceptUri": "http://data.europa.eu/esco/occupation/occ-1", "preferredLabel": "Occ 1"}
    ]
    mock_loader.load_skills.return_value = [
        {"conceptUri": "http://data.europa.eu/esco/skill/skill-1", "preferredLabel": "Skill 1"}
    ]
    mock_loader.load_occupation_skill_relations.return_value = [
        {
            "occupationUri": "http://data.europa.eu/esco/occupation/occ-1",
            "skillUri": "http://data.europa.eu/esco/skill/skill-1",
            "relationType": "mandatory",
        }
    ]
    mock_loader.load_skill_skill_relations.return_value = []
    mock_loader.load_skills_hierarchy.return_value = []

    kg = ESCOKnowledgeGraph()
    report = kg.build_from_loader(loader=mock_loader, load_hierarchy=False, strict=False)

    assert report.is_valid is False
    assert len(report.unexpected_relation_types) == 1
    assert "Unexpected occupationSkillRelation relationType 'mandatory'" in report.unexpected_relation_types[0]


def test_strict_mode_raises_value_error_on_anomalies():
    """Verifies that strict=True causes build_from_loader to raise ValueError on invalid relations."""
    mock_loader = MagicMock(spec=ESCODataLoader)
    mock_loader.load_occupations.return_value = []
    mock_loader.load_skills.return_value = []
    mock_loader.load_occupation_skill_relations.return_value = [
        {"occupationUri": "unknown-occ", "skillUri": "unknown-skill", "relationType": "invalid"}
    ]
    mock_loader.load_skill_skill_relations.return_value = []
    mock_loader.load_skills_hierarchy.return_value = []

    kg = ESCOKnowledgeGraph()
    with pytest.raises(ValueError, match="Knowledge graph build validation failed in strict mode"):
        kg.build_from_loader(loader=mock_loader, load_hierarchy=False, strict=True)


# -----------------------------------------------------------------------------
# 4. Query Interface and Retrieval
# -----------------------------------------------------------------------------


def test_query_methods(esco_kg: ESCOKnowledgeGraph):
    """Verifies node lookup and filtered edge retrieval queries."""
    sample_occ_uri = next(iter(esco_kg.occupations.keys()))
    node = esco_kg.get_node(sample_occ_uri)
    assert isinstance(node, OccupationNode)
    assert node.uri == sample_occ_uri

    sample_skill_uri = next(iter(esco_kg.skills.keys()))
    skill_node = esco_kg.get_node(sample_skill_uri)
    assert isinstance(skill_node, SkillNode)

    assert esco_kg.get_node("http://invalid-uri") is None

    all_skills = esco_kg.get_skills_for_occupation(sample_occ_uri)
    essential_skills = esco_kg.get_skills_for_occupation(sample_occ_uri, relation_filter="essential")
    optional_skills = esco_kg.get_skills_for_occupation(sample_occ_uri, relation_filter="optional")

    assert len(all_skills) == len(essential_skills) + len(optional_skills)
    assert all(r == "essential" for _, r in essential_skills)
    assert all(r == "optional" for _, r in optional_skills)


def test_summary_dictionary(esco_kg: ESCOKnowledgeGraph):
    """Verifies that summary() contains all metrics matching the graph state."""
    summary = esco_kg.summary()
    assert summary["num_occupations"] == 3039
    assert summary["num_skills"] == 13939
    assert summary["num_hierarchy_nodes"] == 640
    assert summary["total_nodes"] == 17618
    assert summary["occupation_skill_edges"] == 126051
    assert summary["occupation_skill_by_relation"]["essential"] == 67600
    assert summary["occupation_skill_by_relation"]["optional"] == 58451
    assert summary["skill_skill_edges"] == 5818
    assert summary["skill_skill_by_relation"]["optional"] == 5629
    assert summary["skill_skill_by_relation"]["essential"] == 189
    assert summary["hierarchy_edges"] == 636
    assert summary["total_edges"] == 126051 + 5818 + 636


# -----------------------------------------------------------------------------
# 5. NetworkX Export Behavior
# -----------------------------------------------------------------------------


def test_to_networkx_without_networkx_raises_importerror(esco_kg: ESCOKnowledgeGraph):
    """Verifies that to_networkx raises an informative ImportError when networkx is absent."""
    with patch("pathgraph_ai.domain.knowledge_graph.nx", None):
        with pytest.raises(ImportError, match="NetworkX is required to build a NetworkX graph"):
            esco_kg.to_networkx()


def test_to_networkx_with_mocked_networkx():
    """Verifies that to_networkx populates nodes and edges with attributes when networkx is available."""
    mock_nx = MagicMock()
    mock_digraph = MagicMock()
    mock_nx.DiGraph.return_value = mock_digraph

    kg = ESCOKnowledgeGraph()
    kg.add_occupation(uri="occ-1", label="Software Developer", isco_group="2512")
    kg.add_skill(uri="skill-1", label="Python", skill_type="skill/competence")
    kg.add_occupation_skill_relation("occ-1", "skill-1", relation_type="essential")
    kg.add_skill_skill_relation("skill-1", "skill-2", relation_type="optional")

    with patch("pathgraph_ai.domain.knowledge_graph.nx", mock_nx):
        g = kg.to_networkx(include_hierarchy=False)
        assert g == mock_digraph
        assert mock_digraph.add_node.call_count >= 2
        assert mock_digraph.add_edge.call_count >= 2
