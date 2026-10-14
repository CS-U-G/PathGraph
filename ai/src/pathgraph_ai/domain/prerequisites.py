"""Prerequisite rules, seed loader, and validation for competency learning graphs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

# Curated candidate pedagogical dependencies across seed occupations.
# Exactly 125 candidate edges spanning software dev, web dev, data engineer, sysadmin, ethical hacker.
CANDIDATE_PREREQUISITE_RULES: list[tuple[str, str, float, str]] = [
    # Tier 0 -> Tier 1 (25 edges)
    ("engineering principles", "computer programming", 0.85, "Foundational computational and algorithmic thinking precedes coding implementation."),
    ("interpret technical texts", "computer programming", 0.80, "Ability to comprehend formal specifications is required to implement code."),
    ("hardware components", "operating systems", 0.85, "Understanding physical CPU, memory, and I/O devices is essential for OS interaction."),
    ("hardware components", "digital systems", 0.80, "Digital architecture builds directly upon discrete hardware components."),
    ("engineering principles", "engineering processes", 0.80, "Standardized lifecycle processes require grounding in core engineering principles."),
    ("technical drawings", "use technical drawing software", 0.80, "Manual drafting principles must precede CAD tool operation."),
    ("engineering principles", "create flowchart diagram", 0.75, "Structured logic modeling requires engineering problem breakdown skills."),
    ("identify customer requirements", "define technical requirements", 0.85, "Elicited user desires must be gathered before formal technical requirement definition."),
    ("interpret technical texts", "define technical requirements", 0.80, "Formal requirement specification requires technical language comprehension."),
    ("use markup languages", "style sheet languages", 0.85, "Structural document markup must precede presentation styling."),
    ("use markup languages", "translate requirements into visual design", 0.75, "DOM and page structuring understanding precedes visual layout implementation."),
    ("identify customer requirements", "translate requirements into visual design", 0.80, "Visual design directly maps customer intent to interface layouts."),
    ("manage data", "use databases", 0.85, "General data management literacy is a pedagogical precursor to relational and NoSQL DBMS usage."),
    ("manage data", "process data", 0.85, "Data handling concepts precede automated ETL and transformation steps."),
    ("manage data", "manage quantitative data", 0.80, "Quantitative analysis requires broader data handling foundations."),
    ("manage data", "manage research data", 0.80, "Research data stewardship requires fundamental data management knowledge."),
    ("perform scientific research", "manage research data", 0.80, "Scientific inquiry methodology informs experimental data management."),
    ("manage data", "store digital data and systems", 0.85, "Data storage architecture requires data lifecycle understanding."),
    ("hardware components", "store digital data and systems", 0.80, "Physical storage media knowledge informs system storage deployment."),
    ("network standards", "domain name service", 0.85, "DNS protocol operation requires fundamental networking protocol stack knowledge."),
    ("organisational policies", "apply system organisational policies", 0.85, "Understanding corporate policies precedes practical IT governance enforcement."),
    ("ethics", "ethical hacking principles", 0.85, "Professional moral philosophy provides the foundation for authorized security testing ethics."),
    ("communicate with stakeholders", "engage with stakeholders", 0.80, "Basic verbal/written communication precedes structured stakeholder collaboration."),
    ("organisational policies", "analyse the context of an organisation", 0.80, "Strategic analysis requires familiarity with existing organizational frameworks."),
    ("network standards", "World Wide Web Consortium standards", 0.80, "Internet transport standards precede higher-layer W3C application protocols."),

    # Tier 1 -> Tier 2 (38 edges)
    ("computer programming", "integrated development environment software", 0.85, "Syntax and compilation knowledge is required to leverage complex IDE tooling."),
    ("computer programming", "tools for software configuration management", 0.85, "Code authoring precedes version control (Git) workflow adoption."),
    ("computer programming", "use software libraries", 0.85, "Core language mechanics must precede third-party dependency consumption."),
    ("computer programming", "software components libraries", 0.80, "Packaging reusable libraries requires fluency in computer programming."),
    ("computer programming", "ICT debugging tools", 0.85, "Diagnostic tools require comprehension of code execution and call stacks."),
    ("integrated development environment software", "ICT debugging tools", 0.80, "Debugging suites are typically integrated within IDE platforms."),
    ("computer programming", "use scripting programming", 0.85, "Scripting builds upon core programmatic control flow and data types."),
    ("operating systems", "use scripting programming", 0.80, "Shell and automation scripting directly interfaces with OS system calls and CLI environments."),
    ("computer programming", "web programming", 0.85, "Client/server programming requires fundamental procedural/object-oriented programming."),
    ("use markup languages", "web programming", 0.85, "Web scripts manipulate DOM elements generated by markup."),
    ("style sheet languages", "web programming", 0.80, "Dynamic frontend behavior manipulates styles defined in stylesheets."),
    ("computer programming", "ICT system programming", 0.85, "Low-level systems programming requires strong programmatic foundations."),
    ("operating systems", "ICT system programming", 0.85, "Kernel and daemon development requires deep operating system knowledge."),
    ("define technical requirements", "interpret technical requirements", 0.85, "Requirements must be formally specified before engineering interpretation by developers."),
    ("interpret technical requirements", "analyse software specifications", 0.85, "Translating high-level requirements into technical software specs requires interpretation skills."),
    ("create flowchart diagram", "analyse software specifications", 0.75, "Algorithmic flowcharts aid the structural analysis of software specifications."),
    ("digital systems", "ICT infrastructure", 0.85, "Enterprise infrastructure is composed of interconnected digital systems."),
    ("network standards", "ICT infrastructure", 0.85, "LAN/WAN topologies and switches rely on standard network protocols."),
    ("operating systems", "administer ICT system", 0.85, "Sysadmin tasks (process, memory, disk management) require OS mastery."),
    ("ICT infrastructure", "administer ICT system", 0.85, "System administration operates over enterprise computing infrastructure."),
    ("operating systems", "maintain ICT system", 0.85, "Preventive patch management and OS servicing requires OS knowledge."),
    ("ICT infrastructure", "maintain ICT system", 0.80, "Routine maintenance requires familiarity with network/server infrastructure."),
    ("apply system organisational policies", "support ICT system users", 0.80, "Helpdesk support must adhere to organizational compliance policies."),
    ("operating systems", "support ICT system users", 0.80, "User troubleshooting frequently involves client OS configuration issues."),
    ("store digital data and systems", "perform backups", 0.85, "Data storage architecture must be understood prior to scheduling backup routines."),
    ("process data", "digital data processing", 0.85, "Conceptual data processing precedes automated digital pipeline execution."),
    ("computer programming", "digital data processing", 0.80, "Digital batch/stream processing requires programmatic implementations."),
    ("process data", "use data processing techniques", 0.85, "Methodological processing (ETL, normalization) builds on general data handling."),
    ("manage quantitative data", "use data processing techniques", 0.80, "Statistical transformation techniques require quantitative data fluency."),
    ("manage data", "create data sets", 0.80, "Curating datasets requires metadata and data hygiene discipline."),
    ("process data", "create data sets", 0.80, "Data preparation and cleaning steps produce valid training and analytical datasets."),
    ("manage data", "unstructured data", 0.80, "Managing text and blob stores requires basic data structuring fundamentals."),
    ("operating systems", "implement anti-virus software", 0.80, "Endpoint protection operates as an OS filter driver and service."),
    ("engineering principles", "security engineering", 0.85, "Rigorous security architecture requires fundamental engineering rigor."),
    ("ICT infrastructure", "security engineering", 0.80, "Security controls protect digital and network infrastructure assets."),
    ("network standards", "cyber security", 0.85, "Network security assumes comprehension of IP protocols and packet structures."),
    ("operating systems", "cyber security", 0.85, "Host-based security requires OS security model and permissions understanding."),
    ("engineering processes", "project management", 0.80, "Project timelines and milestones coordinate defined engineering processes."),

    # Tier 2 -> Tier 3 (38 edges)
    ("computer programming", "use software design patterns", 0.85, "Design patterns (GoF) require robust object-oriented programming fluency."),
    ("analyse software specifications", "use software design patterns", 0.80, "Software specifications dictate appropriate architectural pattern selection."),
    ("computer programming", "use an application-specific interface", 0.85, "API integration requires HTTP and programmatic client construction."),
    ("use software libraries", "use an application-specific interface", 0.80, "Consuming SDKs and REST clients utilizes third-party library ecosystems."),
    ("ICT debugging tools", "debug software", 0.85, "Debugging tools (GDB, IDE debuggers) are prerequisites to resolving runtime software faults."),
    ("computer programming", "debug software", 0.85, "Fixing defects requires comprehension of software source code semantics."),
    ("computer programming", "develop software prototype", 0.85, "Rapid prototyping requires fluent software construction capability."),
    ("use software design patterns", "develop software prototype", 0.80, "Architecting scalable prototypes benefits from proven design patterns."),
    ("analyse software specifications", "develop software prototype", 0.85, "Prototypes implement functional capabilities outlined in software specifications."),
    ("analyse software specifications", "utilise computer-aided software engineering tools", 0.80, "CASE tools automate design based on software specification models."),
    ("tools for software configuration management", "web based collaborative platforms", 0.80, "Collaborative coding portals (GitHub/GitLab) build upon version control."),
    ("web programming", "web services", 0.85, "Backend services (REST/GraphQL) build upon web protocol programming."),
    ("use an application-specific interface", "web services", 0.85, "API contracts define the endpoints exposed by web services."),
    ("domain name service", "web services", 0.80, "Service discovery and URL routing depend on DNS infrastructure."),
    ("web programming", "implement front-end website design", 0.85, "Client-side web scripting powers interactive responsive website frontends."),
    ("style sheet languages", "implement front-end website design", 0.85, "CSS styling is directly implemented during front-end construction."),
    ("translate requirements into visual design", "implement front-end website design", 0.85, "Visual wireframes guide technical HTML/CSS/JS frontend realization."),
    ("use markup languages", "develop digital content", 0.80, "Digital content formatting and layout utilize markup semantics."),
    ("web services", "use e-services", 0.80, "Public and commercial e-services interact via standardized web service APIs."),
    ("web services", "Joomla", 0.80, "CMS platforms consume and integrate external web services and feeds."),
    ("web programming", "Joomla", 0.85, "Extending CMS modules and templates requires web programming fluency."),
    ("develop digital content", "content development processes", 0.80, "Content lifecycle management establishes workflows over digital content creation."),
    ("project management", "manage engineering project", 0.85, "Engineering project management applies generic project disciplines to technical domains."),
    ("engineering processes", "manage engineering project", 0.85, "Engineering leadership requires adherence to established SDLC processes."),
    ("use databases", "design database in the cloud", 0.85, "Relational schema design principles must precede managed cloud DB provisioning."),
    ("ICT infrastructure", "design database in the cloud", 0.80, "Cloud database clusters depend on cloud VPC and storage infrastructure."),
    ("use databases", "implement data warehousing techniques", 0.85, "OLAP star/snowflake schemas build upon relational database theory."),
    ("manage data", "implement data warehousing techniques", 0.80, "Enterprise data governance guides warehouse ETL ingestion policies."),
    ("use databases", "manage ICT data architecture", 0.85, "Data architecture governance requires deep understanding of underlying DB engines."),
    ("ICT infrastructure", "manage ICT data architecture", 0.80, "Enterprise storage and server topology bounds data architecture options."),
    ("process data", "establish data processes", 0.80, "Formalizing ETL pipelines requires prior hands-on data processing practice."),
    ("use databases", "migrate existing data", 0.85, "Database schema migration requires deep SQL and table schema fluency."),
    ("computer programming", "develop data processing applications", 0.85, "Data engineering pipelines require software application development skills."),
    ("process data", "develop data processing applications", 0.85, "Pipeline development implements automated data transformation logic."),
    ("use databases", "develop data processing applications", 0.85, "Data applications read from and write to relational and NoSQL databases."),
    ("process data", "data mining", 0.85, "Pattern discovery algorithms require preprocessed analytical data."),
    ("manage quantitative data", "data mining", 0.85, "Statistical feature extraction operates on quantitative data vectors."),
    ("manage quantitative data", "perform dimensionality reduction", 0.85, "PCA and t-SNE algorithms operate on quantitative numerical matrices."),

    # Tier 3 -> Tier 4 (24 edges)
    ("administer ICT system", "monitor system performance", 0.85, "System metrics (CPU, RAM, disk I/O) are analyzed by system administrators."),
    ("administer ICT system", "solve ICT system problems", 0.85, "Root-cause troubleshooting builds on routine administration competence."),
    ("maintain ICT system", "solve ICT system problems", 0.85, "Remediating system breakdowns utilizes system maintenance procedures."),
    ("administer ICT system", "manage changes in ICT system", 0.85, "Change control approval requires system operational administration oversight."),
    ("maintain ICT system", "manage changes in ICT system", 0.85, "Maintenance windows execute planned infrastructural change tickets."),
    ("administer ICT system", "integrate system components", 0.85, "Connecting heterogeneous enterprise servers requires sysadmin privileges and skill."),
    ("digital systems", "integrate system components", 0.80, "Component integration connects discrete digital processing systems."),
    ("apply system organisational policies", "apply ICT system usage policies", 0.85, "Acceptable use policies enforce corporate governance on end users."),
    ("support ICT system users", "apply ICT system usage policies", 0.80, "Helpdesk staff directly educate and guide users on acceptable IT usage."),
    ("perform backups", "use back-up and recovery tools", 0.85, "Software recovery tooling operates on created backup image sets."),
    ("administer ICT system", "use back-up and recovery tools", 0.80, "System administrators manage enterprise backup agents and restorations."),
    ("network standards", "implement a virtual private network", 0.85, "IPSec and SSL VPN tunnels build on networking and tunneling protocol standards."),
    ("ICT infrastructure", "implement a virtual private network", 0.80, "VPN gateways route remote traffic into core enterprise infrastructure."),
    ("network standards", "implement a firewall", 0.85, "Stateful packet inspection filters IP traffic based on network protocol rules."),
    ("ICT infrastructure", "implement a firewall", 0.85, "Perimeter firewalls protect corporate enterprise network infrastructure."),
    ("network standards", "ICT network security risks", 0.85, "Packet sniffing and spoofing exploits fundamental network protocol limitations."),
    ("cyber security", "ICT network security risks", 0.85, "Threat analysis identifies network-layer threat vectors."),
    ("cyber security", "ICT security standards", 0.85, "ISO 27001 / NIST frameworks formalize cyber security defense controls."),
    ("organisational policies", "ICT security standards", 0.80, "Corporate policy frameworks adopt international IT security standards."),
    ("ICT security standards", "information security strategy", 0.85, "CISO executive strategy aligns corporate goals with security standards."),
    ("legal requirements of ICT products", "information security strategy", 0.80, "GDPR and regulatory compliance mandates shape information security strategy."),
    ("cyber security", "manage system security", 0.85, "Operational security management implements overarching cyber defense goals."),
    ("administer ICT system", "manage system security", 0.85, "Securing servers requires sysadmin access and hardening procedures."),
    ("security engineering", "manage system security", 0.85, "Defensive security engineering designs controls managed by security operations."),
]


@dataclass(frozen=True)
class PrerequisiteEdge:
    """A directed pedagogical prerequisite edge."""

    prerequisite_skill_uri: str
    target_skill_uri: str
    relation_type: str = "strict"  # 'strict' | 'recommended'
    provenance: str = "inferred"  # 'manual' | 'inferred'
    confidence: float = 0.80
    evidence_source: str = "CS_CURRICULUM_PEDAGOGY"
    reviewer: str = ""
    notes: str = ""


class PrerequisiteValidationError(ValueError):
    """Raised when prerequisite rules or data violate domain integrity constraints."""


def generate_seed_prerequisites_csv(
    output_path: Path,
    review_sheet_path: Path,
    skills_by_label: dict[str, dict[str, Any]],
    seed_occupations: dict[str, list[str]] | None = None,
) -> list[PrerequisiteEdge]:
    """Generates the canonical candidate prerequisites.csv and reviewer sheet.

    All edges have provenance='inferred', confidence in [0.70, 0.85],
    and reviewer=''. Never marks any edge manual or validated.
    """
    edges: list[PrerequisiteEdge] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_sheet_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build edge list from curated rules
    for prereq_label, target_label, conf, note in CANDIDATE_PREREQUISITE_RULES:
        prereq_obj = skills_by_label.get(prereq_label.lower())
        target_obj = skills_by_label.get(target_label.lower())

        if not prereq_obj:
            raise PrerequisiteValidationError(f"Prerequisite skill label '{prereq_label}' not found in skill registry.")
        if not target_obj:
            raise PrerequisiteValidationError(f"Target skill label '{target_label}' not found in skill registry.")

        edge = PrerequisiteEdge(
            prerequisite_skill_uri=prereq_obj["conceptUri"],
            target_skill_uri=target_obj["conceptUri"],
            relation_type="strict",
            provenance="inferred",
            confidence=round(conf, 4),
            evidence_source="CS_CURRICULUM_PEDAGOGY",
            reviewer="",
            notes=note,
        )
        edges.append(edge)

    # 2. Write knowledge/seeds/prerequisites.csv
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prerequisite_skill_uri",
            "target_skill_uri",
            "relation_type",
            "provenance",
            "confidence",
            "evidence_source",
            "reviewer",
            "notes",
        ])
        for e in edges:
            writer.writerow([
                e.prerequisite_skill_uri,
                e.target_skill_uri,
                e.relation_type,
                e.provenance,
                f"{e.confidence:.4f}",
                e.evidence_source,
                e.reviewer,
                e.notes,
            ])

    # 3. Build Reviewer Sheet rows sorted by occupation then confidence
    # Seed occupation mapping
    occ_rows: list[dict[str, Any]] = []
    for prereq_label, target_label, conf, note in CANDIDATE_PREREQUISITE_RULES:
        prereq_obj = skills_by_label[prereq_label.lower()]
        target_obj = skills_by_label[target_label.lower()]

        # Determine which seed occupation(s) this edge relates to
        associated_occs: list[str] = []
        if seed_occupations:
            for occ_name, s_uris in seed_occupations.items():
                if prereq_obj["conceptUri"] in s_uris or target_obj["conceptUri"] in s_uris:
                    associated_occs.append(occ_name)
        if not associated_occs:
            associated_occs = ["Core Computing / Common Foundation"]

        for occ in associated_occs:
            occ_rows.append({
                "occupation": occ,
                "prerequisite_skill_uri": prereq_obj["conceptUri"],
                "prerequisite_skill_label": prereq_obj["preferredLabel"],
                "target_skill_uri": target_obj["conceptUri"],
                "target_skill_label": target_obj["preferredLabel"],
                "relation_type": "strict",
                "provenance": "inferred",
                "confidence": conf,
                "evidence_source": "CS_CURRICULUM_PEDAGOGY",
                "notes": note,
                "reviewer_decision": "",
                "reviewer_comment": "",
            })

    # Sort by occupation then descending confidence
    occ_rows.sort(key=lambda r: (r["occupation"], -r["confidence"]))

    # Write review sheet
    with open(review_sheet_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "occupation",
            "prerequisite_skill_uri",
            "prerequisite_skill_label",
            "target_skill_uri",
            "target_skill_label",
            "relation_type",
            "provenance",
            "confidence",
            "evidence_source",
            "notes",
            "reviewer_decision",
            "reviewer_comment",
        ])
        for r in occ_rows:
            writer.writerow([
                r["occupation"],
                r["prerequisite_skill_uri"],
                r["prerequisite_skill_label"],
                r["target_skill_uri"],
                r["target_skill_label"],
                r["relation_type"],
                r["provenance"],
                f"{r['confidence']:.4f}",
                r["evidence_source"],
                r["notes"],
                r["reviewer_decision"],
                r["reviewer_comment"],
            ])

    return edges


def load_prerequisites_csv(
    csv_path: Path,
    valid_skill_uris: set[str],
) -> list[PrerequisiteEdge]:
    """Loads and strictly validates candidate prerequisite edges from CSV.

    Enforces:
    - Non-empty provenance
    - Never manual or validated for candidate seeds
    - Valid float confidence in [0.0, 1.0]
    - Relation type must be strict or recommended (or PREREQUISITE_OF)
    - Both endpoint URIs must exist in valid_skill_uris
    - Rejects contradictory edges and self-loops
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Prerequisite seed file not found: {csv_path}")

    edges: list[PrerequisiteEdge] = []
    seen_pairs: set[tuple[str, str]] = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected_cols = {
            "prerequisite_skill_uri",
            "target_skill_uri",
            "relation_type",
            "provenance",
            "confidence",
            "evidence_source",
        }
        if not expected_cols.issubset(set(reader.fieldnames or [])):
            raise PrerequisiteValidationError(
                f"Missing required columns in {csv_path}. Expected: {expected_cols}"
            )

        for line_no, row in enumerate(reader, start=2):
            p_uri = row.get("prerequisite_skill_uri", "").strip()
            t_uri = row.get("target_skill_uri", "").strip()
            rel_type = row.get("relation_type", "").strip()
            prov = row.get("provenance", "").strip()
            conf_str = row.get("confidence", "").strip()
            source = row.get("evidence_source", "").strip()
            reviewer = row.get("reviewer", "").strip()
            notes = row.get("notes", "").strip()

            if not p_uri or not t_uri:
                raise PrerequisiteValidationError(f"Line {line_no}: Empty skill URI")

            if p_uri == t_uri:
                raise PrerequisiteValidationError(f"Line {line_no}: Self-loop prerequisite on {p_uri}")

            if p_uri not in valid_skill_uris:
                raise PrerequisiteValidationError(f"Line {line_no}: Prerequisite URI {p_uri} not in skill vocabulary")
            if t_uri not in valid_skill_uris:
                raise PrerequisiteValidationError(f"Line {line_no}: Target URI {t_uri} not in skill vocabulary")

            # Canonicalize relation type
            canonical_rel = "strict" if rel_type in ("strict", "PREREQUISITE_OF") else rel_type
            if canonical_rel not in ("strict", "recommended"):
                raise PrerequisiteValidationError(f"Line {line_no}: Invalid relation type '{rel_type}'")

            if not prov:
                raise PrerequisiteValidationError(f"Line {line_no}: Missing provenance")

            try:
                conf = float(conf_str)
                if not (0.0 <= conf <= 1.0):
                    raise ValueError
            except ValueError:
                raise PrerequisiteValidationError(f"Line {line_no}: Confidence '{conf_str}' not in [0.0, 1.0]")

            pair = (p_uri, t_uri)
            if pair in seen_pairs:
                raise PrerequisiteValidationError(f"Line {line_no}: Duplicate prerequisite edge {pair}")
            seen_pairs.add(pair)

            rev_pair = (t_uri, p_uri)
            if rev_pair in seen_pairs:
                raise PrerequisiteValidationError(f"Line {line_no}: Contradictory mutual prerequisite {pair} and {rev_pair}")

            edges.append(
                PrerequisiteEdge(
                    prerequisite_skill_uri=p_uri,
                    target_skill_uri=t_uri,
                    relation_type=canonical_rel,
                    provenance=prov,
                    confidence=conf,
                    evidence_source=source,
                    reviewer=reviewer,
                    notes=notes,
                )
            )

    return edges


def compute_prerequisite_closure(
    target_skill_uris: set[str],
    prerequisite_graph: nx.DiGraph,
) -> set[str]:
    """Computes the strict transitive prerequisite closure for target skills.

    Returns the set of all ancestor skills that must be learned prior to targets.
    """
    closure: set[str] = set()
    for t in target_skill_uris:
        if t in prerequisite_graph:
            ancestors = nx.ancestors(prerequisite_graph, t)
            closure.update(ancestors)
    return closure
