"""Unit tests for CandidatePathwayGenerator and strict topological validity."""
import networkx as nx
import pytest

from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.domain.graph_metrics import get_graph_metrics
from pathgraph_ai.domain.learner import LearningPreferences
from pathgraph_ai.recommenders.candidate_generator import CandidatePathwayGenerator
from pathgraph_ai.recommenders.gap_analyser import SkillGap


@pytest.fixture
def mock_subgraph():
    # DAG:
    # math -> data_structures
    # prog -> data_structures
    # data_structures -> algorithms
    # prog -> web_basics
    # web_basics -> react
    # data_structures -> backend
    # algorithms -> system_design
    # backend -> system_design
    dag = nx.DiGraph()
    skills = [
        ("math", 1, "Foundations"),
        ("prog", 1, "Foundations"),
        ("data_structures", 2, "CS Core"),
        ("algorithms", 3, "CS Core"),
        ("web_basics", 1, "Frontend"),
        ("react", 2, "Frontend"),
        ("backend", 2, "Backend"),
        ("system_design", 4, "Architecture"),
    ]
    for s_id, diff, cat in skills:
        dag.add_node(s_id, difficulty=diff, category=cat)

    edges = [
        ("math", "data_structures"),
        ("prog", "data_structures"),
        ("data_structures", "algorithms"),
        ("prog", "web_basics"),
        ("web_basics", "react"),
        ("data_structures", "backend"),
        ("algorithms", "system_design"),
        ("backend", "system_design"),
    ]
    dag.add_edges_from(edges)
    return dag


@pytest.mark.unit
def test_all_candidates_satisfy_topological_invariants(mock_subgraph):
    generator = CandidatePathwayGenerator(subgraph=mock_subgraph)
    candidates = generator.generate_all(beam_width=5)
    assert len(candidates) >= 3

    for cand in candidates:
        path = cand.skill_ids
        assert len(path) == len(mock_subgraph.nodes)
        pos = {sid: idx for idx, sid in enumerate(path)}

        # Strict invariant assertion: for all u -> v, pos[u] < pos[v]
        for u, v in mock_subgraph.edges:
            assert pos[u] < pos[v], (
                f"Strategy '{cand.strategy}' violated invariant: {u} (pos {pos[u]}) >= {v} (pos {pos[v]})"
            )


@pytest.mark.unit
def test_candidate_archetype_diversity(mock_subgraph):
    generator = CandidatePathwayGenerator(subgraph=mock_subgraph)

    foundational = generator.generate_foundational_first()
    quick_wins = generator.generate_quick_wins()
    clustered = generator.generate_domain_clustered()
    beam = generator.generate_beam_search(beam_width=3)

    # Foundational first starts with roots
    assert foundational.skill_ids[0] in {"math", "prog"}
    # All are 100% valid DAG sequences
    for cand in [foundational, quick_wins, clustered, beam]:
        pos = {sid: idx for idx, sid in enumerate(cand.skill_ids)}
        for u, v in mock_subgraph.edges:
            assert pos[u] < pos[v]


@pytest.mark.unit
def test_candidate_generator_on_real_esco_subgraph(graph: CompetencyGraph):
    sw_dev_uri = "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1"
    reqs = graph.get_occupation_requirements(sw_dev_uri)
    needed = {r.skill_uri for r in reqs[:8]}

    # Induce subgraph from prerequisite graph
    sub = nx.DiGraph()
    for u in needed:
        sub.add_node(u)
    for u in needed:
        for v in graph.get_dependents(u, transitive=False):
            if v in needed:
                sub.add_edge(u, v)

    generator = CandidatePathwayGenerator(subgraph=sub, graph=graph)
    candidates = generator.generate_all()

    for cand in candidates:
        pos = {sid: idx for idx, sid in enumerate(cand.skill_ids)}
        for u, v in sub.edges:
            assert pos[u] < pos[v]
