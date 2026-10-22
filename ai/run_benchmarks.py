"""CLI executable to run PathGraph Recommendation Quality Benchmarks."""

import sys
import logging
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pathgraph_ai.benchmarks.runner import BenchmarkRunner


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 76)
    print("  PATHGRAPH AI - RECOMMENDATION QUALITY BENCHMARKS")
    print("=" * 76)
    runner = BenchmarkRunner()
    report = runner.run_all()

    print("\nBenchmark Evaluation Completed Successfully!")
    print(f" * Personas Evaluated: {len(report.personas_evaluated)}")
    print(f" * Mean Precision@5:   {report.mean_precision_at_5:.4f}")
    print(f" * Mean Recall@5:      {report.mean_recall_at_5:.4f}")
    print(f" * Mean NDCG@5:        {report.mean_ndcg_at_5:.4f}")
    print(f" * Mean MAP:           {report.mean_map:.4f}")
    print(f" * Mean Prereq Violations: {report.mean_prerequisite_violation_rate*100:.1f}%")
    print(f" * Mean Bloat Ratio:   {report.mean_curriculum_bloat_ratio:.2f}x")
    print(f" * Negative Leakage:   {report.mean_leakage_rate*100:.1f}%")
    print(f"\nReports written to:")
    print(f" - JSON:     ai/reports/benchmarks/recommendation_quality_report.json")
    print(f" - Markdown: ai/reports/benchmarks/recommendation_quality_report.md")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
