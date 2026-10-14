"""Filtering and validation sampling pipeline for computing domain occupations."""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pathgraph_ai.pipelines.esco_ingestion import ESCOIngestionPipeline
from pathgraph_ai.settings import settings

logger = logging.getLogger(__name__)

# Tier 1 structural ISCO whitelist (Core Computing & ICT)
TIER1_ISCO_WHITELIST = {
    "1330",  # ICT service managers
    "2511",  # Systems analysts
    "2512",  # Software developers
    "2513",  # Web and multimedia developers
    "2514",  # Applications programmers
    "2519",  # Software and applications developers NEC
    "2521",  # Database designers and administrators
    "2522",  # Systems administrators
    "2523",  # Computer network professionals
    "2529",  # Database and network professionals NEC / Cybersecurity
    "3511",  # ICT operations technicians
    "3512",  # ICT user support technicians
    "3513",  # Computer network and systems technicians
    "3514",  # Web technicians
}

# Outright excluded ISCO groups
OUTRIGHT_EXCLUDED_ISCO = {
    "3521",  # Broadcasting and audiovisual technicians (boom operator, camera operator, etc.)
}

# Tier 2 candidate groups (Adjacent & Interdisciplinary)
TIER2_CANDIDATE_ISCO = {
    "3522",  # Telecommunications engineering technicians
    "2153",  # Telecommunications engineers
    "2166",  # Graphic and multimedia designers
    "2131",  # Biologists, botanists, zoologists (includes bioinformatics)
    "1342",  # Health services managers (includes clinical informatics)
}


@dataclass
class OccupationAuditRecord:
    """Audit entry documenting classification reasoning for an occupation."""

    concept_uri: str
    preferred_label: str
    isco_group: str
    signals: dict[str, Any]
    rule_triggered: str
    status: str
    decision: str | None


class ComputingOccupationFilter:
    """Applies two-tier filtering and generates audit trails and validation samples."""

    def __init__(self, ingestion_pipeline: ESCOIngestionPipeline) -> None:
        self.pipeline = ingestion_pipeline
        self.audit_records: list[OccupationAuditRecord] = []
        self.computing_occupations: dict[str, dict[str, Any]] = {}
        self.tier2_unvalidated: dict[str, dict[str, Any]] = {}
        self.interdisciplinary_review: dict[str, dict[str, Any]] = {}
        self.excluded_occupations: dict[str, dict[str, Any]] = {}

        # ISCED-F 06 digital skill URIs
        self.isced_f_06_uris: set[str] = set()
        self._resolve_isced_f_06_skills()

    def _resolve_isced_f_06_skills(self) -> None:
        """Finds all skills associated with ISCED-F 06 ('ICTs') in skill broader relations."""
        for r in self.pipeline.broader_skills:
            broader_uri = r.get("broaderUri", "")
            concept_uri = r.get("conceptUri", "")
            if "isced-f/06" in broader_uri:
                self.isced_f_06_uris.add(concept_uri)

    def calculate_digital_skill_ratio(self, occ_uri: str) -> tuple[float, int, int]:
        """Calculates ratio of essential skills belonging to digital collections or ISCED-F 06."""
        essential_skills: list[str] = [
            r["skillUri"]
            for r in self.pipeline.occupation_skill_relations
            if r["occupationUri"] == occ_uri and r.get("relationType") == "essential"
        ]

        if not essential_skills:
            return 0.0, 0, 0

        digital_target_set = self.pipeline.digital_skills_uris | self.isced_f_06_uris
        digital_count = sum(1 for s in essential_skills if s in digital_target_set)
        ratio = round(digital_count / len(essential_skills), 4)
        return ratio, digital_count, len(essential_skills)

    def run_filter(self) -> None:
        """Evaluates all occupations and categorizes them into audit tiers."""
        self.audit_records.clear()
        self.computing_occupations.clear()
        self.tier2_unvalidated.clear()
        self.interdisciplinary_review.clear()
        self.excluded_occupations.clear()

        for uri, occ in self.pipeline.occupations.items():
            label = occ["preferredLabel"]
            isco = occ.get("iscoGroup", "").strip()
            ratio, dig_count, ess_count = self.calculate_digital_skill_ratio(uri)

            signals = {
                "digital_skill_ratio": ratio,
                "digital_essential_skills": dig_count,
                "total_essential_skills": ess_count,
                "code": occ.get("code", ""),
            }

            # Outright Exclusion Check
            if isco in OUTRIGHT_EXCLUDED_ISCO:
                record = OccupationAuditRecord(
                    concept_uri=uri,
                    preferred_label=label,
                    isco_group=isco,
                    signals=signals,
                    rule_triggered="RULE_EXCLUDE_OUTRIGHT_ISCO_3521",
                    status="EXCLUDED",
                    decision="EXCLUDE",
                )
                self.audit_records.append(record)
                self.excluded_occupations[uri] = occ
                continue

            # Tier 1 Whitelist Check
            if isco in TIER1_ISCO_WHITELIST:
                record = OccupationAuditRecord(
                    concept_uri=uri,
                    preferred_label=label,
                    isco_group=isco,
                    signals=signals,
                    rule_triggered="RULE_TIER1_ISCO_WHITELIST",
                    status="ACCEPTED",
                    decision="ACCEPT",
                )
                self.audit_records.append(record)
                self.computing_occupations[uri] = occ
                continue

            # Interdisciplinary Keyword Signals (informatics, bioinformatics, etc.)
            is_informatics = "informatics" in label.lower() or "bioinformatic" in label.lower()

            # Tier 2 Candidate Evaluation
            if isco in TIER2_CANDIDATE_ISCO or is_informatics:
                rule_name = (
                    "RULE_TIER2_INTERDISCIPLINARY" if is_informatics else "RULE_TIER2_ADJACENT_CANDIDATE"
                )
                status_name = "INTERDISCIPLINARY_REVIEW" if is_informatics else "UNVALIDATED_CANDIDATE"

                record = OccupationAuditRecord(
                    concept_uri=uri,
                    preferred_label=label,
                    isco_group=isco,
                    signals=signals,
                    rule_triggered=rule_name,
                    status=status_name,
                    decision=None,  # Requires human validation label
                )
                self.audit_records.append(record)
                if is_informatics:
                    self.interdisciplinary_review[uri] = occ
                else:
                    self.tier2_unvalidated[uri] = occ
                continue

            # Default Exclusion
            record = OccupationAuditRecord(
                concept_uri=uri,
                preferred_label=label,
                isco_group=isco,
                signals=signals,
                rule_triggered="RULE_DEFAULT_EXCLUSION_NON_COMPUTING",
                status="EXCLUDED",
                decision="EXCLUDE",
            )
            self.audit_records.append(record)
            self.excluded_occupations[uri] = occ

    def write_audit_log(self, output_path: Path | None = None) -> Path:
        """Writes computing_occupation_filter_audit.json to data/interim/."""
        path = output_path or (settings.interim_data_dir / "computing_occupation_filter_audit.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [asdict(r) for r in self.audit_records]
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        logger.info(f"Wrote filter audit log with {len(serializable)} entries to {path}")
        return path

    def generate_validation_sample(
        self, output_path: Path | None = None, sample_size: int = 90, seed: int = 42
    ) -> Path:
        """Generates stratified + random validation sample CSV with an empty human_label column."""
        path = output_path or (settings.interim_data_dir / "filter_validation_sample.csv")
        path.parent.mkdir(parents=True, exist_ok=True)

        rng = random.Random(seed)

        # 1. Stratum: Core Tier 1 (30 sampled)
        tier1_uris = sorted(list(self.computing_occupations.keys()))
        sample_tier1 = rng.sample(tier1_uris, min(30, len(tier1_uris)))

        # 2. Stratum: Tier 2 Candidates (all unvalidated + all interdisciplinary)
        tier2_uris = sorted(list(self.tier2_unvalidated.keys()) + list(self.interdisciplinary_review.keys()))
        sample_tier2 = rng.sample(tier2_uris, min(40, len(tier2_uris)))

        # 3. Stratum: Controls / Non-computing (20 sampled from excluded)
        excluded_uris = sorted(list(self.excluded_occupations.keys()))
        sample_controls = rng.sample(excluded_uris, min(20, len(excluded_uris)))

        selected_uris = sample_tier1 + sample_tier2 + sample_controls
        # Shuffle deterministically
        rng.shuffle(selected_uris)

        audit_map = {r.concept_uri: r for r in self.audit_records}

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "concept_uri",
                "preferred_label",
                "isco_group",
                "tier_status",
                "digital_skill_ratio",
                "digital_essential_skills",
                "total_essential_skills",
                "rule_triggered",
                "human_label",  # Empty for manual domain expert annotation: KEEP / DROP / INTERDISCIPLINARY
            ])

            for uri in selected_uris:
                rec = audit_map[uri]
                writer.writerow([
                    rec.concept_uri,
                    rec.preferred_label,
                    rec.isco_group,
                    rec.status,
                    rec.signals["digital_skill_ratio"],
                    rec.signals["digital_essential_skills"],
                    rec.signals["total_essential_skills"],
                    rec.rule_triggered,
                    "",  # empty human_label column
                ])

        logger.info(f"Generated validation sample with {len(selected_uris)} occupations at {path}")
        return path
