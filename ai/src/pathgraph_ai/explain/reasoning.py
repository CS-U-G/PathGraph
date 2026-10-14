"""Fact-based templated reasoning and explanation fidelity verification."""

from __future__ import annotations

from typing import Any

from pathgraph_ai.api.schemas import ReasonCode
from pathgraph_ai.domain.competency_graph import CompetencyGraph


def generate_skill_explanation(
    skill_uri: str,
    skill_label: str,
    is_directly_required: bool,
    relation_type: str | None,
    gap: float,
    dependent_label: str | None,
    all_prereqs_met: bool,
    reuse_level: str | None,
    relevance: float,
    versatility: float,
    high_relevance_threshold: float = 0.70,
    high_demand_threshold: float = 0.50,
) -> tuple[str, list[ReasonCode]]:
    """Builds a deterministic fact-based explanation string and assigns contract reason codes."""
    codes: list[ReasonCode] = []

    # Reason codes assignment
    if is_directly_required:
        codes.append(ReasonCode.REQUIRED_BY_TARGET)
    else:
        codes.append(ReasonCode.STEPPING_STONE)

    if gap > 1e-4:
        codes.append(ReasonCode.MASTERY_GAP)

    if all_prereqs_met:
        codes.append(ReasonCode.PREREQ_SATISFIED)

    if reuse_level in ("transversal", "cross-sector"):
        codes.append(ReasonCode.TRANSVERSAL_FOUNDATION)

    if relevance >= high_relevance_threshold:
        codes.append(ReasonCode.HIGH_RELEVANCE)

    if versatility >= high_demand_threshold:
        codes.append(ReasonCode.HIGH_DEMAND)

    # Ensure at least one code is present
    if not codes:
        codes.append(ReasonCode.REQUIRED_BY_TARGET if is_directly_required else ReasonCode.STEPPING_STONE)

    # Fact-based templated explanation
    if is_directly_required:
        rel_str = relation_type or "essential"
        if gap > 1e-4:
            reason = f"Directly required by target occupation as {rel_str} competency with proficiency gap of {gap:.2f}."
        else:
            reason = f"Directly required by target occupation as {rel_str} competency; current mastery meets target."
    else:
        dep_str = dependent_label or "target skills"
        if gap > 1e-4:
            reason = f"Foundational prerequisite required to unlock '{dep_str}' with proficiency gap of {gap:.2f}."
        else:
            reason = f"Foundational prerequisite required to unlock '{dep_str}'; current mastery meets target."

    return reason, codes


def verify_explanation_fidelity(
    item_payload: dict[str, Any],
    graph: CompetencyGraph,
    occupation_uri: str,
) -> tuple[bool, str]:
    """Verifies that every claim in the explanation and reason codes matches stored graph facts and scores (R10)."""
    s_uri = item_payload.get("skill_id")
    if not s_uri or s_uri not in graph.skills:
        return False, f"Skill {s_uri} does not exist in graph"

    skill_info = graph.skills[s_uri]
    codes = set(item_payload.get("reason_codes", []))
    reason = item_payload.get("reason", "")
    gap = float(item_payload.get("gap", 0.0))
    relevance = float(item_payload.get("relevance", 0.0))
    demand = float(item_payload.get("demand", 0.0))

    # 1. Directly required vs Stepping stone
    is_directly_required = graph.requirement_graph.has_edge(occupation_uri, s_uri)
    if is_directly_required:
        if ReasonCode.REQUIRED_BY_TARGET.value not in codes:
            return False, "Directly required skill missing REQUIRED_BY_TARGET reason code"
        if "Directly required by target" not in reason:
            return False, "Directly required skill has mismatched reason sentence template"
    else:
        if ReasonCode.STEPPING_STONE.value not in codes:
            return False, "Prerequisite skill missing STEPPING_STONE reason code"
        if "Foundational prerequisite" not in reason:
            return False, "Stepping stone skill has mismatched reason sentence template"

    # 2. Mastery gap fidelity
    if gap > 1e-4 and ReasonCode.MASTERY_GAP.value not in codes:
        return False, f"Gap of {gap} present but MASTERY_GAP reason code missing"

    # 3. Transversal foundation fidelity
    reuse = skill_info.get("reuseLevel", "")
    if reuse in ("transversal", "cross-sector") and ReasonCode.TRANSVERSAL_FOUNDATION.value not in codes:
        return False, f"Skill reuseLevel is '{reuse}' but TRANSVERSAL_FOUNDATION missing"

    # 4. Demand fidelity
    stored_vers = float(skill_info.get("skill_versatility", 0.0))
    if abs(demand - stored_vers) > 1e-4:
        return False, f"Demand {demand} does not match stored skill_versatility {stored_vers}"

    return True, "Fidelity verified"
