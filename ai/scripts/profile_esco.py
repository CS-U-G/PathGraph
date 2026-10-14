"""CLI script to profile raw ESCO datasets and report true metrics."""

import sys
from pathgraph_ai.pipelines.esco_ingestion import ESCOIngestionPipeline

def main():
    print("=" * 70)
    print("ESCO v1.2.1 Raw Dataset Profiling")
    print("=" * 70)
    pipeline = ESCOIngestionPipeline()
    report = pipeline.run_profiling()

    print(f"Total Occupations: {report.total_unfiltered_occupations:,}")
    print(f"Total Skills:      {report.total_unfiltered_skills:,}")
    print(f"Total Alt Labels:  {report.total_alt_labels:,}")
    print(f"Unique Alt Labels: {report.unique_alt_labels:,}")
    print("\nReferential Integrity Violations:")
    for k, v in report.referential_integrity_violations.items():
        print(f"  {k}: {v}")

    print("\nRelation Types in occupationSkillRelations:")
    for k, v in report.relation_type_distribution.items():
        print(f"  {k}: {v:,}")

    print("\nSkill Types in skills:")
    for k, v in report.skill_type_distribution.items():
        print(f"  {k}: {v:,}")

    print("\nReuse Levels in skills:")
    for k, v in report.reuse_level_distribution.items():
        print(f"  {k}: {v:,}")

    core_whitelist = [
        "1330", "2511", "2512", "2513", "2514", "2519",
        "2521", "2522", "2523", "2529", "3511", "3512", "3513", "3514"
    ]
    core_count = sum(report.isco_occupation_counts.get(c, 0) for c in core_whitelist)
    print("\nCore Computing ISCO Whitelist Counts:")
    for c in core_whitelist:
        print(f"  {c}: {report.isco_occupation_counts.get(c, 0)}")
    print(f"Total Core Whitelist Occupations: {core_count}")
    print("=" * 70)

if __name__ == "__main__":
    main()
