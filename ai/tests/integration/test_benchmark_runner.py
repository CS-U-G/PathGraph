"""Integration tests for BenchmarkRunner against the production recommendation engine."""

import tempfile
from pathlib import Path

from pathgraph_ai.benchmarks.personas import PERSONA_FRONTEND_TO_FULLSTACK
from pathgraph_ai.benchmarks.runner import BenchmarkRunner


def test_benchmark_runner_evaluates_persona_and_writes_reports():
    """Verifies that BenchmarkRunner evaluates a persona and exports valid JSON and Markdown."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = BenchmarkRunner(
            personas=[PERSONA_FRONTEND_TO_FULLSTACK],
            output_dir=Path(tmp_dir),
        )

        report = runner.run_all()

        assert len(report.personas_evaluated) == 1
        res = report.personas_evaluated[0]
        assert res.persona_id == "persona_frontend_to_fullstack"
        assert res.generated_path_length > 0
        assert res.prerequisite_violation_rate == 0.0
        assert res.leakage_rate == 0.0

        json_file = Path(tmp_dir) / "recommendation_quality_report.json"
        md_file = Path(tmp_dir) / "recommendation_quality_report.md"

        assert json_file.exists()
        assert md_file.exists()
        assert json_file.stat().st_size > 0
        assert md_file.stat().st_size > 0

        # Verify Markdown content
        md_text = md_file.read_text(encoding="utf-8")
        assert "PathGraph Recommendation Quality Benchmark Report" in md_text
        assert "persona_frontend_to_fullstack" in md_text
