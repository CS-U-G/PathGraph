"""ESCO Dataset Loader and Validator for PathGraph.

Provides utilities to locate, inspect, validate, and load ESCO
(European Skills, Competences, Qualifications and Occupations) CSV files.
Implements deterministic deduplication for concept versioning and strict
relational/composite-key validation with cross-file referential integrity.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pathgraph_ai.settings import settings

logger = logging.getLogger(__name__)

# Core expected schema definitions based on actual ESCO release files
EXPECTED_COLUMNS: dict[str, list[str]] = {
    "occupations_en.csv": [
        "conceptType",
        "conceptUri",
        "iscoGroup",
        "preferredLabel",
        "altLabels",
        "hiddenLabels",
        "status",
        "modifiedDate",
        "regulatedProfessionNote",
        "scopeNote",
        "definition",
        "inScheme",
        "description",
        "code",
        "naceCode",
    ],
    "skills_en.csv": [
        "conceptType",
        "conceptUri",
        "skillType",
        "reuseLevel",
        "preferredLabel",
        "altLabels",
        "hiddenLabels",
        "status",
        "modifiedDate",
        "scopeNote",
        "definition",
        "inScheme",
        "description",
    ],
    "occupationSkillRelations_en.csv": [
        "occupationUri",
        "occupationLabel",
        "relationType",
        "skillType",
        "skillUri",
        "skillLabel",
    ],
    "skillsHierarchy_en.csv": [
        "Level 0 URI",
        "Level 0 preferred term",
        "Level 1 URI",
        "Level 1 preferred term",
        "Level 2 URI",
        "Level 2 preferred term",
        "Level 3 URI",
        "Level 3 preferred term",
        "Description",
        "Scope note",
        "Level 0 code",
        "Level 1 code",
        "Level 2 code",
        "Level 3 code",
    ],
    "skillSkillRelations_en.csv": [
        "originalSkillUri",
        "originalSkillType",
        "relationType",
        "relatedSkillType",
        "relatedSkillUri",
    ],
}

# Identifier definitions: single primary key for entity files, composite keys for relations/hierarchy
IDENTIFIER_COLUMNS: dict[str, tuple[str, ...]] = {
    "occupations_en.csv": ("conceptUri",),
    "skills_en.csv": ("conceptUri",),
    "occupationSkillRelations_en.csv": ("occupationUri", "skillUri"),
    "skillsHierarchy_en.csv": (
        "Level 0 URI",
        "Level 1 URI",
        "Level 2 URI",
        "Level 3 URI",
    ),
    "skillSkillRelations_en.csv": ("originalSkillUri", "relatedSkillUri"),
}

PRIMARY_IDENTIFIERS: dict[str, str] = {
    fname: keys[0] for fname, keys in IDENTIFIER_COLUMNS.items()
}

VALID_RELATION_TYPES: set[str] = {"essential", "optional"}


def parse_esco_date(date_str: str | None) -> datetime:
    """Parses an ESCO ISO-8601 modifiedDate string into a timezone-naive UTC datetime."""
    if not date_str:
        return datetime.min
    clean_str = date_str.strip().rstrip("Z")
    try:
        return datetime.fromisoformat(clean_str)
    except (ValueError, TypeError):
        return datetime.min


def resolve_duplicate_records(
    records: list[dict[str, str]], key_field: str = "conceptUri"
) -> tuple[list[dict[str, str]], int, list[str], list[str]]:
    """Deterministically deduplicates entity records by primary key in memory.

    Resolution Policy:
    1. Records are grouped by `key_field` (e.g. conceptUri).
    2. Groups with a single record are preserved as-is.
    3. Groups with multiple records:
       a. Attribute consistency: checks preferredLabel.
       b. Status priority: records with status == 'released' prioritized.
       c. Timestamp priority: record with latest 'modifiedDate' chosen.
       d. Tie-breaking: stable tie-breaking on sequence index.
    """
    by_key: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for idx, row in enumerate(records):
        key = row.get(key_field, "").strip()
        by_key.setdefault(key, []).append((idx, row))

    deduped: list[dict[str, str]] = []
    duplicate_count = 0
    errors: list[str] = []
    warnings: list[str] = []

    for key, group in by_key.items():
        if len(group) == 1:
            deduped.append(group[0][1])
            continue

        duplicate_count += len(group) - 1

        labels = {r.get("preferredLabel", "").strip() for _, r in group}
        if len(labels) > 1:
            errors.append(
                f"Conflicting duplicate records for {key_field}='{key}': conflicting labels {labels}"
            )

        sorted_group = sorted(
            group,
            key=lambda item: (
                item[1].get("status", "").strip().lower() == "released",
                parse_esco_date(item[1].get("modifiedDate")),
                item[0],
            ),
            reverse=True,
        )

        chosen_idx, chosen_row = sorted_group[0]
        chosen_date = chosen_row.get("modifiedDate", "unknown")
        warnings.append(
            f"Deduplicated {key_field}='{key}': retained latest revision (modifiedDate={chosen_date}) from {len(group)} records."
        )
        deduped.append(chosen_row)

    return deduped, duplicate_count, errors, warnings


@dataclass
class ValidationReport:
    """Stores dataset file validation statistics and diagnostic results."""

    filename: str
    exists: bool = False
    row_count: int = 0
    header: list[str] = field(default_factory=list)
    missing_expected_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    missing_identifier_count: int = 0
    duplicate_identifier_count: int = 0
    invalid_relation_type_count: int = 0
    orphaned_reference_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def status_label(self) -> str:
        """Returns PASS, WARN, or FAIL status label."""
        if not self.is_valid:
            return "FAIL"
        if self.warnings or self.duplicate_identifier_count > 0 or self.unexpected_columns:
            return "WARN"
        return "PASS"


class ESCODataLoader:
    """Loads and validates ESCO datasets from the raw dataset repository."""

    def __init__(self, raw_data_dir: Path | str | None = None) -> None:
        if raw_data_dir is None:
            self.raw_data_dir = settings.raw_data_dir
        else:
            self.raw_data_dir = Path(raw_data_dir)

    def get_file_path(self, filename: str) -> Path:
        """Returns the absolute path to a raw dataset file."""
        return self.raw_data_dir / filename

    def validate_file(
        self,
        filename: str,
        strict: bool = False,
        known_occupations: set[str] | None = None,
        known_skills: set[str] | None = None,
    ) -> ValidationReport:
        """Validates CSV headers, row counts, identifiers, and relationship consistency."""
        file_path = self.get_file_path(filename)
        report = ValidationReport(filename=filename)

        if not file_path.exists():
            report.errors.append(f"File not found at: {file_path}")
            report.is_valid = False
            return report

        report.exists = True
        expected_cols = EXPECTED_COLUMNS.get(filename, [])
        id_keys = IDENTIFIER_COLUMNS.get(filename, ())

        seen_keys: dict[tuple[str, ...], list[int]] = {}
        missing_ids = 0
        total_rows = 0

        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                report.errors.append("File is empty.")
                report.is_valid = False
                return report

            # Strip BOM if present
            if header:
                header[0] = header[0].lstrip("\ufeff")

            report.header = header
            col_indices = {col_name: i for i, col_name in enumerate(header)}

            if expected_cols:
                report.missing_expected_columns = [
                    col for col in expected_cols if col not in col_indices
                ]
                report.unexpected_columns = [
                    col for col in header if col not in expected_cols
                ]
                if report.missing_expected_columns:
                    report.errors.append(
                        f"Missing expected columns: {report.missing_expected_columns}"
                    )
                if report.unexpected_columns:
                    report.warnings.append(
                        f"Found unexpected columns: {report.unexpected_columns}"
                    )

            key_indices = [col_indices.get(k) for k in id_keys]
            relation_type_idx = col_indices.get("relationType")

            for row_idx, row in enumerate(reader, start=1):
                total_rows += 1

                # 1. Identifier extraction & missing ID check
                key_values: list[str] = []
                has_missing_id = False
                for k_name, k_idx in zip(id_keys, key_indices):
                    if k_idx is None or k_idx >= len(row):
                        has_missing_id = True
                        break
                    val = row[k_idx].strip()
                    if not val and (filename != "skillsHierarchy_en.csv" or k_name == "Level 0 URI"):
                        has_missing_id = True
                    key_values.append(val)

                if has_missing_id:
                    missing_ids += 1
                else:
                    composite_tuple = tuple(key_values)
                    seen_keys.setdefault(composite_tuple, []).append(row_idx)

                # 2. Relation type validation
                if relation_type_idx is not None and relation_type_idx < len(row):
                    rel_type = row[relation_type_idx].strip()
                    if rel_type not in VALID_RELATION_TYPES:
                        report.invalid_relation_type_count += 1
                        if report.invalid_relation_type_count <= 5:
                            report.errors.append(
                                f"Row {row_idx}: Invalid relationType '{rel_type}'. Expected one of {VALID_RELATION_TYPES}"
                            )

                # 3. Referential integrity checks
                if filename == "occupationSkillRelations_en.csv":
                    occ_idx = col_indices.get("occupationUri")
                    skill_idx = col_indices.get("skillUri")
                    if occ_idx is not None and occ_idx < len(row) and known_occupations is not None:
                        occ_val = row[occ_idx].strip()
                        if occ_val and occ_val not in known_occupations:
                            report.orphaned_reference_count += 1
                            if report.orphaned_reference_count <= 5:
                                report.errors.append(
                                    f"Row {row_idx}: Orphaned occupationUri '{occ_val}' not found in occupations."
                                )
                    if skill_idx is not None and skill_idx < len(row) and known_skills is not None:
                        skill_val = row[skill_idx].strip()
                        if skill_val and skill_val not in known_skills:
                            report.orphaned_reference_count += 1
                            if report.orphaned_reference_count <= 5:
                                report.errors.append(
                                    f"Row {row_idx}: Orphaned skillUri '{skill_val}' not found in skills."
                                )

                elif filename == "skillSkillRelations_en.csv" and known_skills is not None:
                    orig_idx = col_indices.get("originalSkillUri")
                    rel_idx = col_indices.get("relatedSkillUri")
                    if orig_idx is not None and orig_idx < len(row):
                        orig_val = row[orig_idx].strip()
                        if orig_val and orig_val not in known_skills:
                            report.orphaned_reference_count += 1
                            if report.orphaned_reference_count <= 5:
                                report.errors.append(
                                    f"Row {row_idx}: Orphaned originalSkillUri '{orig_val}' not found in skills."
                                )
                    if rel_idx is not None and rel_idx < len(row):
                        rel_val = row[rel_idx].strip()
                        if rel_val and rel_val not in known_skills:
                            report.orphaned_reference_count += 1
                            if report.orphaned_reference_count <= 5:
                                report.errors.append(
                                    f"Row {row_idx}: Orphaned relatedSkillUri '{rel_val}' not found in skills."
                                )

        report.row_count = total_rows
        report.missing_identifier_count = missing_ids
        if missing_ids > 0:
            report.errors.append(f"Found {missing_ids} rows with missing or empty primary/composite identifier.")

        duplicate_tuples = {k: v for k, v in seen_keys.items() if len(v) > 1}
        report.duplicate_identifier_count = sum(len(v) - 1 for v in duplicate_tuples.values())

        if report.duplicate_identifier_count > 0:
            if filename in ("occupations_en.csv", "skills_en.csv"):
                msg = (
                    f"Found {report.duplicate_identifier_count} duplicate {id_keys[0]} records "
                    f"({len(duplicate_tuples)} unique concepts with multiple entries)."
                )
                if strict:
                    report.errors.append(f"{msg} In strict mode, duplicates are treated as errors.")
                else:
                    report.warnings.append(
                        f"{msg} Resolvable in-memory via deterministic version selection policy."
                    )
            else:
                report.errors.append(
                    f"Found {report.duplicate_identifier_count} duplicate composite keys in relationship/hierarchy table: {list(duplicate_tuples.keys())[:3]}"
                )

        report.is_valid = (
            report.exists
            and len(report.missing_expected_columns) == 0
            and report.missing_identifier_count == 0
            and report.invalid_relation_type_count == 0
            and report.orphaned_reference_count == 0
            and len(report.errors) == 0
        )

        return report

    def validate_all_core_files(
        self, strict: bool = False, check_referential_integrity: bool = True
    ) -> dict[str, ValidationReport]:
        """Validates all 5 core ESCO dataset files with optional cross-file referential integrity."""
        known_occupations: set[str] | None = None
        known_skills: set[str] | None = None

        if check_referential_integrity:
            try:
                occ_rows = self.load_occupations(deduplicate=True)
                known_occupations = {r["conceptUri"].strip() for r in occ_rows if r.get("conceptUri")}
            except Exception as e:
                logger.warning("Could not pre-load occupations for referential integrity check: %s", e)

            try:
                skill_rows = self.load_skills(deduplicate=True)
                known_skills = {r["conceptUri"].strip() for r in skill_rows if r.get("conceptUri")}
            except Exception as e:
                logger.warning("Could not pre-load skills for referential integrity check: %s", e)

        reports: dict[str, ValidationReport] = {}
        for filename in EXPECTED_COLUMNS.keys():
            reports[filename] = self.validate_file(
                filename=filename,
                strict=strict,
                known_occupations=known_occupations,
                known_skills=known_skills,
            )

        return reports

    def load_csv_rows(
        self, filename: str, limit: int | None = None, deduplicate: bool = False
    ) -> list[dict[str, str]]:
        """Loads CSV rows into a list of dictionaries with optional limiting and deduplication."""
        file_path = self.get_file_path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {file_path}")

        records: list[dict[str, str]] = []
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit is not None and not deduplicate and i >= limit:
                    break
                records.append(row)

        if deduplicate and filename in ("occupations_en.csv", "skills_en.csv"):
            deduped, _, errors, _ = resolve_duplicate_records(records, key_field="conceptUri")
            if errors:
                logger.warning("Deduplication warnings/errors for %s: %s", filename, errors)
            if limit is not None:
                return deduped[:limit]
            return deduped

        if limit is not None:
            return records[:limit]
        return records

    def load_occupations(
        self, limit: int | None = None, deduplicate: bool = True
    ) -> list[dict[str, str]]:
        """Loads occupations records with deterministic deduplication by default."""
        return self.load_csv_rows("occupations_en.csv", limit=limit, deduplicate=deduplicate)

    def load_skills(
        self, limit: int | None = None, deduplicate: bool = True
    ) -> list[dict[str, str]]:
        """Loads skills records with deterministic deduplication by default."""
        return self.load_csv_rows("skills_en.csv", limit=limit, deduplicate=deduplicate)

    def load_occupation_skill_relations(
        self, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Loads occupation-skill relationship records."""
        return self.load_csv_rows("occupationSkillRelations_en.csv", limit=limit, deduplicate=False)

    def load_skill_skill_relations(
        self, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Loads skill-skill relationship records."""
        return self.load_csv_rows("skillSkillRelations_en.csv", limit=limit, deduplicate=False)

    def load_skills_hierarchy(
        self, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Loads skills hierarchy records."""
        return self.load_csv_rows("skillsHierarchy_en.csv", limit=limit, deduplicate=False)
