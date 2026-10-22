"""Deterministic Benchmark Runner for PathGraph Recommendation Quality.

Executes the production PathEngine against the 5 standardized benchmark personas,
computes precision, recall, NDCG, MAP, prerequisite violation rate, curriculum bloat ratio,
and recommendation leakage, and exports machine-readable JSON and Markdown reports.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pathgraph_ai.api.schemas import (
    LearnerProfile,
    LearnerSkill,
    RecommendationRequest,
    RecommendationTarget,
)
from pathgraph_ai.benchmarks.metrics import (
    average_precision,
    curriculum_bloat_ratio,
    essential_curriculum_coverage,
    ndcg_at_k,
    precision_at_k,
    prerequisite_violation_rate,
    recall_at_k,
    recommendation_leakage_rate,
)
from pathgraph_ai.benchmarks.personas import ALL_BENCHMARK_PERSONAS, BenchmarkPersona
from pathgraph_ai.benchmarks.schemas import (
    AggregateBenchmarkReport,
    PersonaMetricResult,
)
from pathgraph_ai.domain.competency_graph import CompetencyGraph
from pathgraph_ai.recommenders.path_engine import PathEngine
from pathgraph_ai.settings import AI_DIR, settings

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Executes recommendation quality benchmarks across curated personas."""

    def __init__(
        self,
        graph: CompetencyGraph | None = None,
        personas: list[BenchmarkPersona] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else AI_DIR / "reports" / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if graph is None:
            logger.info("Initializing CompetencyGraph for benchmark evaluation...")
            prereqs_path = settings.seeds_dir / "prerequisites.csv"
            self.graph = CompetencyGraph.from_processed_data(
                processed_dir=settings.processed_data_dir,
                prerequisites_csv_path=prereqs_path,
            )
        else:
            self.graph = graph

        self.path_engine = PathEngine(graph=self.graph)
        self.personas = personas or ALL_BENCHMARK_PERSONAS

    def evaluate_persona(self, persona: BenchmarkPersona) -> PersonaMetricResult:
        """Executes recommendation pipeline for one persona and computes quality metrics."""
        learner_skills = [
            LearnerSkill(
                skill_id=s.skill_uri,
                label=s.label,
                proficiency=s.proficiency,
            )
            for s in persona.current_skills
        ]

        req = RecommendationRequest(
            learner=LearnerProfile(
                skills=learner_skills,
                target=RecommendationTarget(occupation_id=persona.target_occupation_uri),
            )
        )

        response = self.path_engine.recommend(request=req)

        # Extract ordered sequence of recommended skill IDs
        recommended_skill_uris = [item.skill_id for item in response.learning_path]

        relevant_gt = set(persona.expert_relevant_target_skills)
        essential_gt = set(persona.expected_missing_essential)
        irrelevant_gt = set(persona.known_irrelevant_skills)

        # Ranking Metrics against relevant target skills
        p5 = precision_at_k(recommended_skill_uris, relevant_gt, 5)
        p10 = precision_at_k(recommended_skill_uris, relevant_gt, 10)
        r5 = recall_at_k(recommended_skill_uris, essential_gt, 5)
        r10 = recall_at_k(recommended_skill_uris, essential_gt, 10)
        ndcg5 = ndcg_at_k(recommended_skill_uris, relevant_gt, 5)
        ndcg10 = ndcg_at_k(recommended_skill_uris, relevant_gt, 10)
        ap = average_precision(recommended_skill_uris, relevant_gt)

        # Curriculum Completeness & Prerequisite Sequencing
        coverage = essential_curriculum_coverage(recommended_skill_uris, essential_gt)
        prereq_res = prerequisite_violation_rate(
            recommended_skill_uris, persona.explicit_prerequisites
        )

        # Curriculum Bloat & Irrelevant Leakage
        bloat_res = curriculum_bloat_ratio(recommended_skill_uris, essential_gt)
        leakage_res = recommendation_leakage_rate(recommended_skill_uris, irrelevant_gt)

        return PersonaMetricResult(
            persona_id=persona.persona_id,
            target_occupation_label=persona.target_occupation_label,
            precision_at_5=p5,
            precision_at_10=p10,
            recall_at_5=r5,
            recall_at_10=r10,
            ndcg_at_5=ndcg5,
            ndcg_at_10=ndcg10,
            map_score=ap,
            essential_curriculum_coverage=coverage,
            prerequisite_violation_rate=prereq_res["violation_rate"],
            prerequisite_violations_count=prereq_res["violations_count"],
            prerequisites_checked_count=prereq_res["total_checked"],
            curriculum_bloat_ratio=bloat_res["bloat_ratio"],
            bloat_numerator=bloat_res["numerator"],
            bloat_denominator=bloat_res["denominator"],
            leakage_rate=leakage_res["leakage_rate"],
            leaked_skills=leakage_res["leaked_items"],
            generated_path_length=len(recommended_skill_uris),
            verified_missing_essential_count=len(essential_gt),
        )

    def run_all(self) -> AggregateBenchmarkReport:
        """Executes benchmarks across all configured personas and compiles report."""
        t_now = datetime.now(timezone.utc).isoformat()
        results: list[PersonaMetricResult] = []

        logger.info("Executing benchmark evaluations for %d personas...", len(self.personas))
        for p in self.personas:
            logger.info("Evaluating persona: %s (%s)", p.persona_id, p.target_occupation_label)
            res = self.evaluate_persona(p)
            results.append(res)

        n = max(1, len(results))
        mean_p5 = round(sum(r.precision_at_5 for r in results) / n, 4)
        mean_p10 = round(sum(r.precision_at_10 for r in results) / n, 4)
        mean_r5 = round(sum(r.recall_at_5 for r in results) / n, 4)
        mean_r10 = round(sum(r.recall_at_10 for r in results) / n, 4)
        mean_ndcg5 = round(sum(r.ndcg_at_5 for r in results) / n, 4)
        mean_ndcg10 = round(sum(r.ndcg_at_10 for r in results) / n, 4)
        mean_map = round(sum(r.map_score for r in results) / n, 4)
        mean_cov = round(sum(r.essential_curriculum_coverage for r in results) / n, 4)
        mean_violation = round(sum(r.prerequisite_violation_rate for r in results) / n, 4)
        mean_bloat = round(sum(r.curriculum_bloat_ratio for r in results) / n, 4)
        mean_leak = round(sum(r.leakage_rate for r in results) / n, 4)

        # Synthesize observed weaknesses based on measurements
        observed_weaknesses: list[str] = []
        recommendations: list[str] = []

        if mean_bloat > 2.0:
            observed_weaknesses.append(
                f"Curriculum bloat ratio is elevated (mean {mean_bloat:.2f}x). Generated learning paths "
                "schedule an average of several times more skills than the core essential gap, "
                "diluting learner focus."
            )
            recommendations.append(
                "Introduce curriculum budgeting / max_path_length constraints in PathEngine to bound "
                "optional skill expansion."
            )

        if mean_cov < 0.90:
            observed_weaknesses.append(
                f"Essential curriculum coverage (mean {mean_cov*100:.1f}%) leaves some core skills omitted "
                "due to reachability or prerequisite filtering."
            )
            recommendations.append(
                "Ensure un-anchored essential target skills are guaranteed inclusion in the final DAG tier."
            )

        if mean_violation > 0.0:
            observed_weaknesses.append(
                f"Prerequisite violation rate is {mean_violation*100:.1f}%, indicating non-topological sequencing."
            )
            recommendations.append(
                "Enforce strict topological sorting across all prerequisite DAG partitions."
            )
        else:
            observed_weaknesses.append(
                "Prerequisite violation rate is 0.0% across all evaluated instructional pairs, "
                "confirming DAG ordering integrity where prerequisite data exists."
            )

        if mean_leak == 0.0:
            observed_weaknesses.append(
                "Zero cross-domain negative control leakage observed (0.0% leakage rate across all personas)."
            )

        report = AggregateBenchmarkReport(
            benchmark_version="1.0.0",
            execution_timestamp=t_now,
            environment_info={
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "occupations_count": len(self.graph.occupations),
                "skills_count": len(self.graph.skills),
                "prerequisite_edges_count": self.graph.prerequisite_graph.number_of_edges(),
            },
            configuration={
                "k_values": [5, 10],
                "pareto_ranking_enabled": True,
                "unblocking_centrality_enabled": True,
            },
            personas_evaluated=results,
            mean_precision_at_5=mean_p5,
            mean_precision_at_10=mean_p10,
            mean_recall_at_5=mean_r5,
            mean_recall_at_10=mean_r10,
            mean_ndcg_at_5=mean_ndcg5,
            mean_ndcg_at_10=mean_ndcg10,
            mean_map=mean_map,
            mean_essential_coverage=mean_cov,
            mean_prerequisite_violation_rate=mean_violation,
            mean_curriculum_bloat_ratio=mean_bloat,
            mean_leakage_rate=mean_leak,
            observed_weaknesses=observed_weaknesses,
            recommendations_for_improvement=recommendations,
        )

        self.save_json_report(report)
        self.save_markdown_report(report)
        return report

    def save_json_report(self, report: AggregateBenchmarkReport) -> Path:
        """Serializes benchmark report to machine-readable JSON."""
        target = self.output_dir / "recommendation_quality_report.json"
        with open(target, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        logger.info("Saved benchmark JSON report to %s", target)
        return target

    def save_markdown_report(self, report: AggregateBenchmarkReport) -> Path:
        """Generates a comprehensive Markdown evaluation report."""
        target = self.output_dir / "recommendation_quality_report.md"

        lines: list[str] = [
            "# PathGraph Recommendation Quality Benchmark Report",
            "",
            f"**Benchmark Version:** {report.benchmark_version}  ",
            f"**Execution Date:** {report.execution_timestamp}  ",
            f"**Environment:** Python {report.environment_info.get('python_version')} on {report.environment_info.get('platform')}  ",
            f"**Graph Scale:** {report.environment_info.get('occupations_count'):,} occupations, "
            f"{report.environment_info.get('skills_count'):,} skills, "
            f"{report.environment_info.get('prerequisite_edges_count'):,} curated prerequisite edges  ",
            "",
            "---",
            "",
            "## 1. Executive Summary & Aggregate Performance",
            "",
            "| Evaluation Metric | Aggregate Mean | Target / Ideal | Status |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Precision@5** | `{report.mean_precision_at_5:.4f}` | > 0.7000 | {'PASS' if report.mean_precision_at_5 >= 0.7 else 'INSPECT'} |",
            f"| **Precision@10** | `{report.mean_precision_at_10:.4f}` | > 0.6000 | {'PASS' if report.mean_precision_at_10 >= 0.6 else 'INSPECT'} |",
            f"| **Recall@5** | `{report.mean_recall_at_5:.4f}` | > 0.5000 | {'PASS' if report.mean_recall_at_5 >= 0.5 else 'INSPECT'} |",
            f"| **Recall@10** | `{report.mean_recall_at_10:.4f}` | > 0.7000 | {'PASS' if report.mean_recall_at_10 >= 0.7 else 'INSPECT'} |",
            f"| **NDCG@5** | `{report.mean_ndcg_at_5:.4f}` | > 0.7500 | {'PASS' if report.mean_ndcg_at_5 >= 0.75 else 'INSPECT'} |",
            f"| **NDCG@10** | `{report.mean_ndcg_at_10:.4f}` | > 0.7000 | {'PASS' if report.mean_ndcg_at_10 >= 0.70 else 'INSPECT'} |",
            f"| **Mean Average Precision (MAP)** | `{report.mean_map:.4f}` | > 0.6000 | {'PASS' if report.mean_map >= 0.6 else 'INSPECT'} |",
            f"| **Essential Curriculum Coverage** | `{report.mean_essential_coverage*100:.1f}%` | 100.0% | {'PASS' if report.mean_essential_coverage >= 0.9 else 'WARN'} |",
            f"| **Prerequisite Violation Rate** | `{report.mean_prerequisite_violation_rate*100:.1f}%` | 0.0% | {'PASS' if report.mean_prerequisite_violation_rate == 0.0 else 'FAIL'} |",
            f"| **Curriculum Bloat Ratio** | `{report.mean_curriculum_bloat_ratio:.2f}x` | 1.0x - 2.5x | {'PASS' if report.mean_curriculum_bloat_ratio <= 3.0 else 'BLOAT'} |",
            f"| **Negative Control Leakage Rate** | `{report.mean_leakage_rate*100:.1f}%` | 0.0% | {'PASS' if report.mean_leakage_rate == 0.0 else 'FAIL'} |",
            "",
            "---",
            "",
            "## 2. Per-Persona Evaluation Breakdown",
            "",
            "| Persona ID | Target Occupation | P@5 | R@5 | NDCG@5 | MAP | Coverage | Prereq Violations | Bloat Ratio | Leakage |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for p in report.personas_evaluated:
            lines.append(
                f"| `{p.persona_id}` | {p.target_occupation_label} | "
                f"`{p.precision_at_5:.2f}` | `{p.recall_at_5:.2f}` | `{p.ndcg_at_5:.2f}` | "
                f"`{p.map_score:.2f}` | `{p.essential_curriculum_coverage*100:.0f}%` | "
                f"`{p.prerequisite_violations_count}/{p.prerequisites_checked_count}` (`{p.prerequisite_violation_rate*100:.1f}%`) | "
                f"`{p.curriculum_bloat_ratio:.2f}x` | `{p.leakage_rate*100:.1f}%` |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Metric Definitions & Formulas",
                "",
                "1. **Precision@k**: Proximity of top-k recommended skills to expert-verified target competencies:",
                "   $$\\text{Precision@k} = \\frac{|\\text{Unique Recommended Top-k} \\cap \\text{Ground Truth Relevant}|}{k}$$",
                "",
                "2. **Recall@k**: Proportion of missing essential gap requirements retrieved in top-k:",
                "   $$\\text{Recall@k} = \\frac{|\\text{Unique Recommended Top-k} \\cap \\text{Ground Truth Missing Essential}|}{|\\text{Ground Truth Missing Essential}|}$$",
                "",
                "3. **NDCG@k (Normalized Discounted Cumulative Gain)**: Rank-weighted position discount of relevant skills:",
                "   $$\\text{DCG@k} = \\sum_{i=1}^{k} \\frac{\\text{rel}_i}{\\log_2(i + 1)}, \\quad \\text{NDCG@k} = \\frac{\\text{DCG@k}}{\\text{IDCG@k}}$$",
                "",
                "4. **Prerequisite Violation Rate**: Percentage of active instructional pairs where dependent precedes prerequisite:",
                "   $$\\text{Violation Rate} = \\frac{|\\{(u, v) \\in E_{\\text{prereq}} : \\text{pos}(u) > \\text{pos}(v)\\}|}{|\\{(u, v) \\in E_{\\text{prereq}} : u, v \\in S\\}|}$$",
                "",
                "5. **Curriculum Bloat Ratio**: Ratio of total scheduled learning path steps to core verified essential requirements:",
                "   $$\\text{Bloat Ratio} = \\frac{\\text{Scheduled Path Length}}{\\max(1, |\\text{Verified Missing Essential}|)}$$",
                "",
                "6. **Negative Control Leakage Rate**: Frequency of cross-domain irrelevant control skills surfaced:",
                "   $$\\text{Leakage Rate} = \\frac{|S_{\\text{recommended}} \\cap S_{\\text{irrelevant negative controls}}|}{|S_{\\text{recommended}}|}$$",
                "",
                "---",
                "",
                "## 4. Observed Weaknesses & Architectural Next Actions",
                "",
                "### Measured Weaknesses:",
            ]
        )

        for w in report.observed_weaknesses:
            lines.append(f"- {w}")

        lines.append("")
        lines.append("### Recommended Action Items:")
        for a in report.recommendations_for_improvement:
            lines.append(f"1. {a}")

        lines.append("")
        lines.append("---")
        lines.append("*Report generated automatically by PathGraph AI Benchmark Runner.*")

        content = "\n".join(lines)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Saved benchmark Markdown report to %s", target)
        return target
