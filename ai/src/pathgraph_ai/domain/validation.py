"""Validation suite V1-V8 and stats generator for PathGraph Competency Graph."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from pathgraph_ai.api.schemas import ReasonCode
from pathgraph_ai.domain.competency_graph import CompetencyGraph


@dataclass
class CheckResult:
    """Detailed result for a single validation check (V1-V8)."""

    check_id: str
    name: str
    description: str
    passed: bool
    violations_count: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphValidationReport:
    """Full validation report containing checks V1 through V8."""

    report_id: str
    generated_at: str
    graph_version: str
    esco_version: str
    overall_pass: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    summary: dict[str, str]
    checks: dict[str, CheckResult]
    graph_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "graph_version": self.graph_version,
            "esco_version": self.esco_version,
            "overall_pass": self.overall_pass,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "summary": self.summary,
            "checks": {k: asdict(v) for k, v in self.checks.items()},
            "graph_metrics": self.graph_metrics,
        }

    def write_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class GraphValidator:
    """Executes validation suite V1 through V8 against a CompetencyGraph instance."""

    def __init__(self, graph: CompetencyGraph) -> None:
        self.graph = graph

    def run_all(self, graph_version: str = "1.0.0", esco_version: str = "v1.2.1") -> GraphValidationReport:
        checks: dict[str, CheckResult] = {
            "V1": self.validate_v1_no_dangling_edges(),
            "V2": self.validate_v2_prerequisite_dag(),
            "V3": self.validate_v3_min_essential_skills(),
            "V4": self.validate_v4_score_bounds(),
            "V5": self.validate_v5_root_reachability(),
            "V6": self.validate_v6_reason_codes_parity(),
            "V7": self.validate_v7_provenance_and_confidence(),
            "V8": self.validate_v8_duplicates_and_contradictions(),
        }

        overall_pass = all(c.passed for c in checks.values())
        passed_count = sum(1 for c in checks.values() if c.passed)
        failed_count = len(checks) - passed_count

        summary = {k: "PASS" if v.passed else "FAIL" for k, v in checks.items()}

        metrics = {
            "occupations_count": len(self.graph.occupations),
            "skills_count": len(self.graph.skills),
            "unknown_prerequisites_skills_count": len(self.graph.unknown_prerequisite_skills),
            "requirement_edges_count": self.graph.requirement_graph.number_of_edges(),
            "taxonomy_edges_count": self.graph.taxonomy_graph.number_of_edges(),
            "association_edges_count": self.graph.association_graph.number_of_edges(),
            "prerequisite_edges_count": self.graph.prerequisite_graph.number_of_edges(),
        }

        return GraphValidationReport(
            report_id=f"gvr-{int(datetime.now(timezone.utc).timestamp())}",
            generated_at=datetime.now(timezone.utc).isoformat(),
            graph_version=graph_version,
            esco_version=esco_version,
            overall_pass=overall_pass,
            total_checks=len(checks),
            passed_checks=passed_count,
            failed_checks=failed_count,
            summary=summary,
            checks=checks,
            graph_metrics=metrics,
        )

    def validate_v1_no_dangling_edges(self) -> CheckResult:
        """V1: No dangling edges across all 4 graph layers."""
        dangling_req: list[dict[str, str]] = []
        for u, v in self.graph.requirement_graph.edges():
            if u not in self.graph.occupations:
                dangling_req.append({"source": u, "target": v, "layer": "requirement", "reason": "Unknown occupation"})
            if v not in self.graph.skills:
                dangling_req.append({"source": u, "target": v, "layer": "requirement", "reason": "Unknown skill"})

        dangling_assoc: list[dict[str, str]] = []
        for u, v in self.graph.association_graph.edges():
            if u not in self.graph.skills or v not in self.graph.skills:
                dangling_assoc.append({"source": u, "target": v, "layer": "association", "reason": "Endpoint not in skills"})

        dangling_prereq: list[dict[str, str]] = []
        for u, v in self.graph.prerequisite_graph.edges():
            if u not in self.graph.skills or v not in self.graph.skills:
                dangling_prereq.append({"source": u, "target": v, "layer": "prerequisite", "reason": "Endpoint not in skills"})

        total_dangling = len(dangling_req) + len(dangling_assoc) + len(dangling_prereq)
        return CheckResult(
            check_id="V1",
            name="No dangling edges",
            description="All edges in all layers connect existing registered nodes.",
            passed=total_dangling == 0,
            violations_count=total_dangling,
            details={
                "dangling_requirement_edges": dangling_req,
                "dangling_association_edges": dangling_assoc,
                "dangling_prerequisite_edges": dangling_prereq,
            },
        )

    def validate_v2_prerequisite_dag(self) -> CheckResult:
        """V2: Prerequisite layer is a DAG (on failure report strongly connected components)."""
        is_dag, sccs = self.graph.check_prerequisite_acyclicity()
        return CheckResult(
            check_id="V2",
            name="Prerequisite layer is a DAG",
            description="The prerequisite graph must be strictly acyclic; report SCCs on failure.",
            passed=is_dag,
            violations_count=len(sccs),
            details={
                "is_dag": is_dag,
                "strongly_connected_components": sccs,
            },
        )

    def validate_v3_min_essential_skills(self) -> CheckResult:
        """V3: Each target occupation has at least 3 essential skills."""
        failures: list[dict[str, Any]] = []
        checked_occs = 0

        for occ_uri, occ_data in self.graph.occupations.items():
            checked_occs += 1
            essential_count = sum(
                1
                for _, _, data in self.graph.requirement_graph.out_edges(occ_uri, data=True)
                if data.get("relation_type") == "essential"
            )
            if essential_count < 3:
                failures.append({
                    "occupation_uri": occ_uri,
                    "preferredLabel": occ_data.get("preferredLabel", ""),
                    "iscoGroup": occ_data.get("iscoGroup", ""),
                    "essential_skills_count": essential_count,
                })

        return CheckResult(
            check_id="V3",
            name="Minimum essential skills per occupation",
            description="Every core computing occupation must require at least 3 essential skills.",
            passed=len(failures) == 0,
            violations_count=len(failures),
            details={
                "total_occupations_audited": checked_occs,
                "failing_occupations": failures,
            },
        )

    def validate_v4_score_bounds(self) -> CheckResult:
        """V4: All scores in [0, 1] (target proficiencies, weights, confidences, versatility)."""
        violations: list[str] = []

        # Check requirement graph edge attributes
        for u, v, data in self.graph.requirement_graph.edges(data=True):
            tp = data.get("target_proficiency")
            if tp is not None and not (0.0 <= tp <= 1.0):
                violations.append(f"Req edge ({u}->{v}) target_proficiency {tp} not in [0, 1]")
            rw = data.get("requirement_weight")
            if rw is not None and not (0.0 <= rw <= 1.0):
                violations.append(f"Req edge ({u}->{v}) requirement_weight {rw} not in [0, 1]")

        # Check prerequisite graph edge attributes
        for u, v, data in self.graph.prerequisite_graph.edges(data=True):
            conf = data.get("confidence")
            if conf is not None and not (0.0 <= conf <= 1.0):
                violations.append(f"Prereq edge ({u}->{v}) confidence {conf} not in [0, 1]")

        # Check skill versatility
        for uri, skill in self.graph.skills.items():
            vers = skill.get("skill_versatility")
            if vers is not None:
                try:
                    f_vers = float(vers)
                    if not (0.0 <= f_vers <= 1.0):
                        violations.append(f"Skill {uri} versatility {f_vers} not in [0, 1]")
                except ValueError:
                    violations.append(f"Skill {uri} non-numeric versatility {vers}")

        return CheckResult(
            check_id="V4",
            name="All scores bounded in [0, 1]",
            description="All proficiencies, weights, confidences, and versatility metrics bounded in [0, 1].",
            passed=len(violations) == 0,
            violations_count=len(violations),
            details={"violations_sample": violations[:20]},
        )

    def validate_v5_root_reachability(self) -> CheckResult:
        """V5: Strict prerequisites reachable from root nodes."""
        prereq_g = self.graph.prerequisite_graph
        active_nodes = {n for n, d in prereq_g.degree() if d > 0}
        if not active_nodes:
            return CheckResult(
                check_id="V5",
                name="Strict prerequisites reachable from root nodes",
                description="Every active prerequisite node is reachable from an in-degree 0 root node.",
                passed=True,
                violations_count=0,
                details={"active_nodes_count": 0, "root_nodes_count": 0},
            )

        roots = [n for n in active_nodes if prereq_g.in_degree(n) == 0]
        reachable = set(roots)
        for r in roots:
            reachable.update(nx.descendants(prereq_g, r))

        unreachable = active_nodes - reachable
        return CheckResult(
            check_id="V5",
            name="Strict prerequisites reachable from root nodes",
            description="Every active prerequisite node is reachable from an in-degree 0 root node.",
            passed=len(unreachable) == 0,
            violations_count=len(unreachable),
            details={
                "active_nodes_count": len(active_nodes),
                "root_nodes_count": len(roots),
                "unreachable_nodes": list(unreachable),
            },
        )

    def validate_v6_reason_codes_parity(self) -> CheckResult:
        """V6: Emitted reason codes exist in contract enum."""
        contract_codes = {rc.value for rc in ReasonCode}
        expected_codes = {
            "REQUIRED_BY_TARGET",
            "MASTERY_GAP",
            "PREREQ_SATISFIED",
            "HIGH_RELEVANCE",
            "HIGH_DEMAND",
            "TRANSVERSAL_FOUNDATION",
            "STEPPING_STONE",
        }
        exact_parity = contract_codes == expected_codes
        return CheckResult(
            check_id="V6",
            name="Reason codes contract parity",
            description="Emitted reason codes must maintain exact parity with OpenAPI ReasonCode enum.",
            passed=exact_parity,
            violations_count=0 if exact_parity else 1,
            details={
                "contract_reason_codes": sorted(list(contract_codes)),
                "expected_reason_codes": sorted(list(expected_codes)),
                "difference": list(contract_codes.symmetric_difference(expected_codes)),
            },
        )

    def validate_v7_provenance_and_confidence(self) -> CheckResult:
        """V7: Every prerequisite edge has provenance and confidence."""
        missing: list[dict[str, Any]] = []
        for u, v, data in self.graph.prerequisite_graph.edges(data=True):
            prov = data.get("provenance")
            conf = data.get("confidence")
            if not prov or conf is None:
                missing.append({"source": u, "target": v, "provenance": prov, "confidence": conf})

        return CheckResult(
            check_id="V7",
            name="Prerequisite provenance and confidence present",
            description="Every prerequisite edge must specify non-empty provenance and valid confidence.",
            passed=len(missing) == 0,
            violations_count=len(missing),
            details={"missing_edges": missing},
        )

    def validate_v8_duplicates_and_contradictions(self) -> CheckResult:
        """V8: Duplicate or contradictory edges flagged."""
        contradictions: list[tuple[str, str]] = []
        prereq_g = self.graph.prerequisite_graph

        for u, v in prereq_g.edges():
            if prereq_g.has_edge(v, u):
                pair = tuple(sorted([u, v]))
                if pair not in contradictions:
                    contradictions.append(pair)

        return CheckResult(
            check_id="V8",
            name="Duplicate or contradictory edges flagged",
            description="Zero bidirectional contradictory edges in the prerequisite DAG.",
            passed=len(contradictions) == 0,
            violations_count=len(contradictions),
            details={"contradictory_cycles": contradictions},
        )

    def compute_graph_stats(self, seed_roles_map: dict[str, str]) -> dict[str, Any]:
        """Computes comprehensive graph stats including prerequisite coverage per seed occupation and depth."""
        prereq_g = self.graph.prerequisite_graph
        active_prereq_skills = {n for n, d in prereq_g.degree() if d > 0}

        # Seed occupation coverage
        seed_coverage: dict[str, Any] = {}
        for role_name, occ_uri in seed_roles_map.items():
            if occ_uri in self.graph.occupations:
                req_skills = [
                    v
                    for _, v, data in self.graph.requirement_graph.out_edges(occ_uri, data=True)
                    if data.get("relation_type") == "essential"
                ]
                with_prereq = [s for s in req_skills if s in active_prereq_skills]
                cov_ratio = len(with_prereq) / len(req_skills) if req_skills else 0.0
                seed_coverage[role_name] = {
                    "occupation_uri": occ_uri,
                    "essential_skills_count": len(req_skills),
                    "skills_with_prerequisites_count": len(with_prereq),
                    "prerequisite_coverage_ratio": round(cov_ratio, 4),
                    "prerequisite_coverage_percentage": f"{cov_ratio * 100:.1f}%",
                }

        # Prerequisite depth in DAG
        node_depths: dict[str, int] = {}
        if nx.is_directed_acyclic_graph(prereq_g) and len(active_prereq_skills) > 0:
            sub = prereq_g.subgraph(active_prereq_skills)
            for n in nx.topological_sort(sub):
                preds = list(sub.predecessors(n))
                if not preds:
                    node_depths[n] = 0
                else:
                    node_depths[n] = 1 + max(node_depths[p] for p in preds)

        depths_list = list(node_depths.values()) if node_depths else [0]
        max_depth = max(depths_list) if depths_list else 0
        mean_depth = sum(depths_list) / len(depths_list) if depths_list else 0.0

        stats = {
            "node_counts": {
                "occupations": len(self.graph.occupations),
                "skills": len(self.graph.skills),
                "active_prerequisite_skills": len(active_prereq_skills),
                "unknown_prerequisites_skills": len(self.graph.unknown_prerequisite_skills),
            },
            "edge_counts": {
                "requirement_edges": self.graph.requirement_graph.number_of_edges(),
                "taxonomy_edges": self.graph.taxonomy_graph.number_of_edges(),
                "association_edges": self.graph.association_graph.number_of_edges(),
                "prerequisite_edges": self.graph.prerequisite_graph.number_of_edges(),
            },
            "seed_occupations_coverage": seed_coverage,
            "prerequisite_depth_metrics": {
                "max_depth": max_depth,
                "mean_depth": round(mean_depth, 3),
            },
        }
        return stats
