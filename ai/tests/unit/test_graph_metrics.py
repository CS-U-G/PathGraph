"""Unit tests for graph metrics, centrality, and logarithmic unblocking impact factor."""
import math
import networkx as nx
import pytest

from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.domain.graph_metrics import PrerequisiteGraphMetrics, get_graph_metrics
from pathgraph_ai.recommenders.gap_analyser import GapAnalyser


@pytest.mark.unit
def test_prerequisite_graph_metrics_calculation():
    # Build a test DAG: A -> B -> C, and A -> D
    dag = nx.DiGraph()
    dag.add_edge("skill_a", "skill_b")
    dag.add_edge("skill_b", "skill_c")
    dag.add_edge("skill_a", "skill_d")

    metrics = PrerequisiteGraphMetrics(dag)

    # Depth verification
    assert metrics.get_depth("skill_a") == 0
    assert metrics.get_depth("skill_b") == 1
    assert metrics.get_depth("skill_c") == 2
    assert metrics.get_depth("skill_d") == 1

    # Transitive descendants
    assert metrics.get_descendants("skill_a") == {"skill_b", "skill_c", "skill_d"}
    assert metrics.get_descendants("skill_b") == {"skill_c"}
    assert metrics.get_descendants("skill_c") == set()

    # Logarithmic unblocking impact: 1 + ln(1 + |Descendants|)
    assert metrics.get_impact("skill_c") == 1.0
    assert metrics.get_impact("skill_b") == round(1.0 + math.log(1.0 + 1), 4)
    assert metrics.get_impact("skill_a") == round(1.0 + math.log(1.0 + 3), 4)

    # Unblocks count with target filtering
    assert metrics.get_unblocks_count("skill_a") == 3
    assert metrics.get_unblocks_count("skill_a", target_subgraph={"skill_c", "skill_e"}) == 1


@pytest.mark.unit
def test_metrics_cache_idempotence():
    dag = nx.DiGraph()
    dag.add_edge("u", "v")
    m1 = get_graph_metrics(dag)
    m2 = get_graph_metrics(dag)
    assert m1 is m2


@pytest.mark.unit
def test_competency_graph_unblocking_impact_populated(graph: CompetencyGraph):
    # Every skill in graph must have unblocking impact >= 1.0
    for s_uri in list(graph.skills.keys())[:20]:
        impact = graph.get_unblocking_impact(s_uri)
        assert impact >= 1.0
        assert graph.get_prerequisite_impact(s_uri) == impact


@pytest.mark.unit
def test_gap_analyser_incorporates_unblocking_impact(graph: CompetencyGraph):
    sw_dev_uri = "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1"
    analyser = GapAnalyser(graph, use_unblocking_impact=True)
    gaps = analyser.analyse_target(sw_dev_uri, learner_skills={})

    assert len(gaps) > 0
    for g in gaps:
        assert g.unblocking_impact >= 1.0
        assert 0.0 <= g.priority_weight <= 1.0
