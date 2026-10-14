"""ESCO dataset ingestion and profiling pipeline using RFC 4180 CSV parsing."""

from __future__ import annotations

import csv
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathgraph_ai.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FileProfile:
    """Summary metrics of a parsed ESCO CSV file."""

    filename: str
    record_count: int
    columns: list[str]
    null_counts: dict[str, int]
    duplicate_pks: int
    duplicate_labels: int
    unique_pks: int


@dataclass
class ESCOProfileReport:
    """Full profiling report across all canonical ESCO dataset files."""

    file_profiles: dict[str, FileProfile] = field(default_factory=dict)
    referential_integrity_violations: dict[str, int] = field(default_factory=dict)
    relation_type_distribution: Counter = field(default_factory=Counter)
    relation_type_by_isco: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    skill_type_distribution: Counter = field(default_factory=Counter)
    reuse_level_distribution: Counter = field(default_factory=Counter)
    total_alt_labels: int = 0
    unique_alt_labels: int = 0
    skills_with_alt_labels: int = 0
    occupations_with_alt_labels: int = 0
    isco_occupation_counts: dict[str, int] = field(default_factory=dict)
    raw_occupation_rows: int = 0
    unique_occupations: int = 0
    duplicate_occupation_rows: int = 0
    raw_skill_rows: int = 0
    unique_skills: int = 0
    duplicate_skill_rows: int = 0


class ESCOIngestionPipeline:
    """Parses, profiles, and validates the canonical ESCO v1.2.1 dataset."""

    def __init__(self, raw_data_dir: Path | None = None) -> None:
        self.raw_data_dir = raw_data_dir or settings.raw_data_dir
        self.occupations: dict[str, dict[str, Any]] = {}
        self.skills: dict[str, dict[str, Any]] = {}
        self.occupation_skill_relations: list[dict[str, Any]] = []
        self.skill_skill_relations: list[dict[str, Any]] = []
        self.broader_skills: list[dict[str, Any]] = []
        self.broader_occupations: list[dict[str, Any]] = []
        self.skill_groups: dict[str, dict[str, Any]] = {}
        self.digital_skills_uris: set[str] = set()
        self.transversal_skills_uris: set[str] = set()
        self.skill_versatility: dict[str, float] = {}

    def _read_csv(self, filename: str) -> tuple[list[str], list[dict[str, str]]]:
        """Reads a CSV file using standard RFC 4180 compliance, properly parsing multiline text."""
        filepath = self.raw_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"ESCO data file not found: {filepath}")

        rows: list[dict[str, str]] = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return [], []
            # Strip BOM if present
            header[0] = header[0].lstrip("\ufeff")
            for row in reader:
                if not row:
                    continue
                # Map row cells to header columns
                row_dict = {header[i]: row[i] if i < len(row) else "" for i in range(len(header))}
                rows.append(row_dict)
        return header, rows

    def run_profiling(self) -> ESCOProfileReport:
        """Profiles the full ESCO raw dataset files and checks cross-file referential integrity."""
        report = ESCOProfileReport()

        files_to_profile = [
            ("occupations_en.csv", "conceptUri", "preferredLabel"),
            ("skills_en.csv", "conceptUri", "preferredLabel"),
            ("occupationSkillRelations_en.csv", None, None),
            ("skillSkillRelations_en.csv", None, None),
            ("broaderRelationsOccPillar_en.csv", None, None),
            ("broaderRelationsSkillPillar_en.csv", None, None),
            ("ISCOGroups_en.csv", "conceptUri", "preferredLabel"),
            ("skillGroups_en.csv", "conceptUri", "preferredLabel"),
            ("digitalSkillsCollection_en.csv", "conceptUri", "preferredLabel"),
            ("transversalSkillsCollection_en.csv", "conceptUri", "preferredLabel"),
        ]

        raw_records: dict[str, list[dict[str, str]]] = {}

        for filename, pk_col, label_col in files_to_profile:
            cols, rows = self._read_csv(filename)
            raw_records[filename] = rows

            null_counts = {c: 0 for c in cols}
            pks: set[str] = set()
            labels: set[str] = set()
            dup_pks = 0
            dup_labels = 0

            for r in rows:
                for c in cols:
                    if not r.get(c) or r[c].strip() == "":
                        null_counts[c] += 1

                if pk_col and r.get(pk_col):
                    pk_val = r[pk_col].strip()
                    if pk_val in pks:
                        dup_pks += 1
                    else:
                        pks.add(pk_val)

                if label_col and r.get(label_col):
                    lbl_val = r[label_col].strip().lower()
                    if lbl_val in labels:
                        dup_labels += 1
                    else:
                        labels.add(lbl_val)

            report.file_profiles[filename] = FileProfile(
                filename=filename,
                record_count=len(rows),
                columns=cols,
                null_counts=null_counts,
                duplicate_pks=dup_pks,
                duplicate_labels=dup_labels,
                unique_pks=len(pks),
            )

        # 1. Profile Occupations
        occ_rows = raw_records["occupations_en.csv"]
        report.raw_occupation_rows = len(occ_rows)
        for r in occ_rows:
            uri = r["conceptUri"]
            isco = r.get("iscoGroup", "").strip()
            self.occupations[uri] = r  # Deduplicates identical duplicate rows by URI
            report.isco_occupation_counts[isco] = report.isco_occupation_counts.get(isco, 0) + 1

            alt_raw = r.get("altLabels", "")
            if alt_raw.strip():
                report.occupations_with_alt_labels += 1
                for alt in alt_raw.split("\n"):
                    cleaned = alt.strip()
                    if cleaned:
                        report.total_alt_labels += 1

        report.unique_occupations = len(self.occupations)
        report.duplicate_occupation_rows = report.raw_occupation_rows - report.unique_occupations

        # 2. Profile Skills
        skill_rows = raw_records["skills_en.csv"]
        report.raw_skill_rows = len(skill_rows)
        all_alt_set: set[str] = set()

        for r in skill_rows:
            uri = r["conceptUri"]
            self.skills[uri] = r  # Deduplicates identical duplicate rows by URI
            stype = r.get("skillType", "").strip()
            reuse = r.get("reuseLevel", "").strip()
            report.skill_type_distribution[stype] += 1
            report.reuse_level_distribution[reuse] += 1

            alt_raw = r.get("altLabels", "")
            if alt_raw.strip():
                report.skills_with_alt_labels += 1
                for alt in alt_raw.split("\n"):
                    cleaned = alt.strip()
                    if cleaned:
                        report.total_alt_labels += 1
                        all_alt_set.add(cleaned.lower())

        report.unique_skills = len(self.skills)
        report.duplicate_skill_rows = report.raw_skill_rows - report.unique_skills
        report.unique_alt_labels = len(all_alt_set)

        # 3. Profile Occupation-Skill Relations & Versatility over UNIQUE Occupations
        occ_skill_rows = raw_records["occupationSkillRelations_en.csv"]
        self.occupation_skill_relations = occ_skill_rows

        # Map each skill to the set of UNIQUE occupation URIs listing it
        skill_unique_occs: dict[str, set[str]] = defaultdict(set)
        broken_occ_refs = 0
        broken_skill_refs = 0

        for r in occ_skill_rows:
            occ_uri = r["occupationUri"]
            skill_uri = r["skillUri"]
            rel_type = r.get("relationType", "").strip()

            if occ_uri not in self.occupations:
                broken_occ_refs += 1
            else:
                occ_isco = self.occupations[occ_uri].get("iscoGroup", "").strip()
                report.relation_type_by_isco[occ_isco][rel_type] += 1

            if skill_uri not in self.skills:
                broken_skill_refs += 1
            else:
                if occ_uri in self.occupations:
                    skill_unique_occs[skill_uri].add(occ_uri)

            report.relation_type_distribution[rel_type] += 1

        report.referential_integrity_violations["occupationSkill_broken_occ"] = broken_occ_refs
        report.referential_integrity_violations["occupationSkill_broken_skill"] = broken_skill_refs

        # Compute skill_versatility across ALL 3,039 UNIQUE occupations
        max_unique_occs = max(len(occs) for occs in skill_unique_occs.values()) if skill_unique_occs else 1
        for s_uri in self.skills:
            unique_occ_count = len(skill_unique_occs.get(s_uri, set()))
            self.skill_versatility[s_uri] = round(unique_occ_count / max_unique_occs, 6)

        # 4. Profile Skill-Skill Relations
        skill_skill_rows = raw_records["skillSkillRelations_en.csv"]
        self.skill_skill_relations = skill_skill_rows
        broken_ss_orig = 0
        broken_ss_rel = 0
        for r in skill_skill_rows:
            orig = r["originalSkillUri"]
            related = r["relatedSkillUri"]
            if orig not in self.skills:
                broken_ss_orig += 1
            if related not in self.skills:
                broken_ss_rel += 1
        report.referential_integrity_violations["skillSkill_broken_original"] = broken_ss_orig
        report.referential_integrity_violations["skillSkill_broken_related"] = broken_ss_rel

        # 5. Profile Collections
        for r in raw_records.get("digitalSkillsCollection_en.csv", []):
            uri = r.get("conceptUri", "").strip()
            if uri in self.skills:
                self.digital_skills_uris.add(uri)

        for r in raw_records.get("transversalSkillsCollection_en.csv", []):
            uri = r.get("conceptUri", "").strip()
            if uri in self.skills:
                self.transversal_skills_uris.add(uri)

        for r in raw_records.get("skillGroups_en.csv", []):
            uri = r.get("conceptUri", "").strip()
            self.skill_groups[uri] = r

        self.broader_skills = raw_records.get("broaderRelationsSkillPillar_en.csv", [])
        self.broader_occupations = raw_records.get("broaderRelationsOccPillar_en.csv", [])

        return report
