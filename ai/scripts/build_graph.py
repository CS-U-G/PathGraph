"""Build and validate the PathGraph Competency Graph (Phase 3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
AI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_DIR / "src"))

from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.domain.prerequisites import generate_seed_prerequisites_csv
from pathgraph_ai.domain.validation import GraphValidator
from pathgraph_ai.settings import settings

SEED_OCCUPATIONS: dict[str, str] = {
    "software developer": "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1",
    "web developer": "http://data.europa.eu/esco/occupation/c40a2919-48a9-40ea-b506-1f34f693496d",
    "data engineer": "http://data.europa.eu/esco/occupation/2079755f-d809-49e6-8037-4de6180e54c0",
    "ICT system administrator": "http://data.europa.eu/esco/occupation/9e2e6e1e-363b-4e1b-a673-7bc0f7343300",
    "ethical hacker": "http://data.europa.eu/esco/occupation/76ef0a87-6afe-4560-b5d0-9a086abe45c5",
}


def main() -> None:
    print("=" * 75)
    print("PathGraph AI — Phase 3: Competency Graph & Prerequisite Layer Build")
    print("=" * 75)

    processed_dir = settings.processed_data_dir
    interim_dir = settings.interim_data_dir
    seeds_dir = settings.seeds_dir

    skills_file = processed_dir / "skills.json"
    req_file = processed_dir / "occupation_skill_edges.json"
    if not skills_file.exists() or not req_file.exists():
        print(f"Error: Processed dataset files missing. Run build_dataset.py first.")
        sys.exit(1)

    with open(skills_file, "r", encoding="utf-8") as f:
        skills_by_label = {s["preferredLabel"].lower(): s for s in json.load(f)}

    with open(req_file, "r", encoding="utf-8") as f:
        req_edges = json.load(f)

    # Collect essential skill URIs per seed occupation
    seed_occ_skill_sets: dict[str, list[str]] = {}
    for name, uri in SEED_OCCUPATIONS.items():
        skills_for_occ = [
            e["skillUri"]
            for e in req_edges
            if e["occupationUri"] == uri and e.get("relationType") == "essential"
        ]
        seed_occ_skill_sets[name] = skills_for_occ

    prerequisites_csv = seeds_dir / "prerequisites.csv"
    review_sheet_seeds = seeds_dir / "prerequisites_review_sheet.csv"
    review_sheet_interim = interim_dir / "prerequisites_review_sheet.csv"

    print(f"\n[1/4] Generating candidate prerequisite edges for seed occupations...")
    edges = generate_seed_prerequisites_csv(
        output_path=prerequisites_csv,
        review_sheet_path=review_sheet_seeds,
        skills_by_label=skills_by_label,
        seed_occupations=seed_occ_skill_sets,
    )
    # Also mirror review sheet to interim
    import shutil
    shutil.copyfile(review_sheet_seeds, review_sheet_interim)

    print(f"      - Created {len(edges)} candidate edges in: {prerequisites_csv}")
    print(f"      - Created reviewer sheet with empty decision/comment columns:")
    print(f"        -> {review_sheet_seeds}")
    print(f"        -> {review_sheet_interim}")

    print(f"\n[2/4] Constructing 4-layer Competency Graph...")
    graph = CompetencyGraph.from_processed_data(
        processed_dir=processed_dir,
        prerequisites_csv_path=prerequisites_csv,
    )

    print(f"      - Occupations registered (Tier 1 core): {len(graph.occupations)}")
    print(f"      - Unique skills in vocabulary: {len(graph.skills)}")
    print(f"      - Layer 1: Requirement edges (G_req): {graph.requirement_graph.number_of_edges()}")
    print(f"      - Layer 2: Taxonomy edges (G_tax): {graph.taxonomy_graph.number_of_edges()}")
    print(f"      - Layer 3: Associative edges (G_assoc): {graph.association_graph.number_of_edges()}")
    print(f"      - Layer 4: Prerequisite edges (G_prereq): {graph.prerequisite_graph.number_of_edges()}")
    print(f"      - Skills flagged prerequisites_unknown: {len(graph.unknown_prerequisite_skills)}")

    print(f"\n[3/4] Running Graph Validation Suite (V1–V8)...")
    validator = GraphValidator(graph)
    report = validator.run_all(
        graph_version=settings.graph_version,
        esco_version=settings.esco_version,
    )

    report_path = processed_dir / "graph_validation_report.json"
    report.write_json(report_path)
    print(f"      - Validation report written to: {report_path}")

    # Compute & write graph_stats.json
    stats = validator.compute_graph_stats(SEED_OCCUPATIONS)
    stats_path = processed_dir / "graph_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"      - Graph statistics written to: {stats_path}")

    print(f"\n[4/4] Validation Summary:")
    print("-" * 75)
    print(f"{'Check':<8} | {'Status':<6} | {'Violations':<11} | {'Description'}")
    print("-" * 75)
    for check_id, res in report.checks.items():
        status = "PASS" if res.passed else "FAIL"
        print(f"{check_id:<8} | {status:<6} | {res.violations_count:<11} | {res.name}")
    print("-" * 75)
    print(f"Overall Result: {'PASSED (8/8)' if report.overall_pass else 'FAILED'}")

    print("\nPrerequisite Coverage Across Seed Occupations:")
    for role, cdata in stats["seed_occupations_coverage"].items():
        print(f"  • {role:<25}: {cdata['skills_with_prerequisites_count']}/{cdata['essential_skills_count']} essential skills ({cdata['prerequisite_coverage_percentage']})")
    print(f"Prerequisite DAG Max Depth: {stats['prerequisite_depth_metrics']['max_depth']}, Mean Depth: {stats['prerequisite_depth_metrics']['mean_depth']}")
    print("=" * 75)

    if not report.overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
