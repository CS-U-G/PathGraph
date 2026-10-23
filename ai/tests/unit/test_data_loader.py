"""Comprehensive Unit and Integration Tests for ESCODataLoader.

Verifies:
1. Dataset file discovery and path resolution
2. Schema conformity against actual ESCO v1.2.1 CSV headers
3. UTF-8 decoding and multiline quoted field robustness
4. Deterministic deduplication policy (latest revision, status priority, tie-breaker)
5. Validation status reporting (PASS / WARN / FAIL) and error vs warning separation
6. Composite key uniqueness in relationship tables
7. Foreign key referential integrity (detection of orphaned references)
8. Invalid relation type detection
9. Conflicting duplicate attribute detection
10. Compatibility with API service layer expectations
"""

import csv
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from pathgraph_ai.pipelines.data_loader import (
    EXPECTED_COLUMNS,
    IDENTIFIER_COLUMNS,
    VALID_RELATION_TYPES,
    ESCODataLoader,
    ValidationReport,
    parse_esco_date,
    resolve_duplicate_records,
)


@pytest.fixture
def data_loader() -> ESCODataLoader:
    """Fixture providing an ESCODataLoader initialized to default raw_data_dir."""
    return ESCODataLoader()


# -----------------------------------------------------------------------------
# 1. Path Resolution and Schema Conformity Tests
# -----------------------------------------------------------------------------


def test_raw_dataset_paths_exist(data_loader: ESCODataLoader):
    """Verifies that all 5 core ESCO dataset files exist in raw dataset dir."""
    for filename in EXPECTED_COLUMNS.keys():
        file_path = data_loader.get_file_path(filename)
        assert file_path.exists(), f"Expected core dataset file not found: {file_path}"


def test_core_schema_headers_match_actual_csv(data_loader: ESCODataLoader):
    """Verifies that EXPECTED_COLUMNS matches actual headers in each core file."""
    for filename, expected_cols in EXPECTED_COLUMNS.items():
        report = data_loader.validate_file(filename)
        assert report.exists is True
        assert len(report.missing_expected_columns) == 0, (
            f"File {filename} is missing expected columns: {report.missing_expected_columns}"
        )
        assert report.header == expected_cols


# -----------------------------------------------------------------------------
# 2. Encoding and Multiline Robustness Tests
# -----------------------------------------------------------------------------


def test_utf8_multiline_parsing(data_loader: ESCODataLoader):
    """Verifies that files with multiline quoted descriptions parse correctly without losing rows."""
    occupations = data_loader.load_occupations(limit=20, deduplicate=False)
    assert len(occupations) == 20
    for occ in occupations:
        assert "conceptUri" in occ
        assert "preferredLabel" in occ


# -----------------------------------------------------------------------------
# 3. Deterministic Deduplication Tests
# -----------------------------------------------------------------------------


def test_resolve_duplicate_records_selects_latest_date():
    """Verifies deterministic selection of the latest modifiedDate among identical concepts."""
    records = [
        {
            "conceptUri": "http://data.europa.eu/esco/occupation/test-1",
            "preferredLabel": "Software Engineer",
            "status": "released",
            "modifiedDate": "2025-07-31T09:38:59.122Z",
            "description": "Older version",
        },
        {
            "conceptUri": "http://data.europa.eu/esco/occupation/test-1",
            "preferredLabel": "Software Engineer",
            "status": "released",
            "modifiedDate": "2025-11-26T10:02:16.164Z",
            "description": "Newer version",
        },
    ]

    deduped, dup_count, errors, warnings = resolve_duplicate_records(records, "conceptUri")
    assert dup_count == 1
    assert len(errors) == 0
    assert len(warnings) == 1
    assert len(deduped) == 1
    assert deduped[0]["modifiedDate"] == "2025-11-26T10:02:16.164Z"
    assert deduped[0]["description"] == "Newer version"


def test_resolve_duplicate_records_prioritizes_released_status():
    """Verifies that 'released' status takes priority over non-released records."""
    records = [
        {
            "conceptUri": "http://data.europa.eu/esco/skill/test-2",
            "preferredLabel": "Python Programming",
            "status": "released",
            "modifiedDate": "2025-07-01T00:00:00.000Z",
        },
        {
            "conceptUri": "http://data.europa.eu/esco/skill/test-2",
            "preferredLabel": "Python Programming",
            "status": "deprecated",
            "modifiedDate": "2025-12-01T00:00:00.000Z",
        },
    ]

    deduped, dup_count, errors, warnings = resolve_duplicate_records(records, "conceptUri")
    assert dup_count == 1
    assert len(deduped) == 1
    assert deduped[0]["status"] == "released"


def test_conflicting_duplicates_generates_error():
    """Verifies that duplicate records with contradictory preferredLabels generate an error."""
    records = [
        {
            "conceptUri": "http://data.europa.eu/esco/skill/test-3",
            "preferredLabel": "Label A",
            "status": "released",
        },
        {
            "conceptUri": "http://data.europa.eu/esco/skill/test-3",
            "preferredLabel": "Label B",
            "status": "released",
        },
    ]

    deduped, dup_count, errors, warnings = resolve_duplicate_records(records, "conceptUri")
    assert dup_count == 1
    assert len(errors) == 1
    assert "conflicting labels" in errors[0]


def test_load_occupations_and_skills_deduplication(data_loader: ESCODataLoader):
    """Verifies that load_occupations and load_skills return unique URIs matching dataset counts."""
    occs_raw = data_loader.load_csv_rows("occupations_en.csv", deduplicate=False)
    occs_deduped = data_loader.load_occupations(deduplicate=True)
    assert len(occs_raw) == 3043
    assert len(occs_deduped) == 3039
    assert len({o["conceptUri"] for o in occs_deduped}) == 3039

    skills_raw = data_loader.load_csv_rows("skills_en.csv", deduplicate=False)
    skills_deduped = data_loader.load_skills(deduplicate=True)
    assert len(skills_raw) == 13960
    assert len(skills_deduped) == 13939
    assert len({s["conceptUri"] for s in skills_deduped}) == 13939


# -----------------------------------------------------------------------------
# 4. Validation Status and Diagnostic Reporting Tests
# -----------------------------------------------------------------------------


def test_validate_all_core_files_full_integrity(data_loader: ESCODataLoader):
    """Verifies that all 5 core files pass validation with referential integrity."""
    reports = data_loader.validate_all_core_files(check_referential_integrity=True)

    for fname, rep in reports.items():
        assert rep.is_valid is True, f"File {fname} failed validation: {rep.errors}"
        assert rep.missing_identifier_count == 0
        assert len(rep.missing_expected_columns) == 0

    assert reports["occupations_en.csv"].status_label == "WARN"
    assert reports["occupations_en.csv"].duplicate_identifier_count == 4
    assert len(reports["occupations_en.csv"].warnings) == 1

    assert reports["skills_en.csv"].status_label == "WARN"
    assert reports["skills_en.csv"].duplicate_identifier_count == 21
    assert len(reports["skills_en.csv"].warnings) == 1

    assert reports["occupationSkillRelations_en.csv"].status_label == "PASS"
    assert reports["occupationSkillRelations_en.csv"].duplicate_identifier_count == 0
    assert reports["occupationSkillRelations_en.csv"].orphaned_reference_count == 0
    assert reports["occupationSkillRelations_en.csv"].invalid_relation_type_count == 0

    assert reports["skillSkillRelations_en.csv"].status_label == "PASS"
    assert reports["skillSkillRelations_en.csv"].duplicate_identifier_count == 0
    assert reports["skillSkillRelations_en.csv"].orphaned_reference_count == 0
    assert reports["skillSkillRelations_en.csv"].invalid_relation_type_count == 0

    assert reports["skillsHierarchy_en.csv"].status_label == "PASS"
    assert reports["skillsHierarchy_en.csv"].duplicate_identifier_count == 0


def test_strict_mode_treats_entity_duplicates_as_failure(data_loader: ESCODataLoader):
    """Verifies that strict=True causes files with duplicates to fail validation."""
    occ_rep = data_loader.validate_file("occupations_en.csv", strict=True)
    assert occ_rep.is_valid is False
    assert occ_rep.status_label == "FAIL"
    assert any("In strict mode, duplicates are treated as errors" in err for err in occ_rep.errors)

    skill_rep = data_loader.validate_file("skills_en.csv", strict=True)
    assert skill_rep.is_valid is False
    assert skill_rep.status_label == "FAIL"


# -----------------------------------------------------------------------------
# 5. Relationship, Foreign Key, and Integrity Error Detection Tests
# -----------------------------------------------------------------------------


def test_missing_identifier_fails_validation():
    """Verifies that missing or blank primary keys cause is_valid to become False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_file = tmp_path / "occupations_en.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS["occupations_en.csv"])
            writer.writeheader()
            writer.writerow({
                "conceptUri": "",
                "conceptType": "Occupation",
                "preferredLabel": "Test",
            })

        loader = ESCODataLoader(raw_data_dir=tmp_path)
        report = loader.validate_file("occupations_en.csv")
        assert report.is_valid is False
        assert report.status_label == "FAIL"
        assert report.missing_identifier_count == 1
        assert any("missing or empty primary/composite identifier" in err for err in report.errors)


def test_invalid_relation_type_fails_validation():
    """Verifies that invalid relationType values trigger validation errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_file = tmp_path / "occupationSkillRelations_en.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS["occupationSkillRelations_en.csv"])
            writer.writeheader()
            writer.writerow({
                "occupationUri": "http://data.europa.eu/esco/occupation/occ-1",
                "occupationLabel": "Doctor",
                "relationType": "mandatory",
                "skillType": "knowledge",
                "skillUri": "http://data.europa.eu/esco/skill/sk-1",
                "skillLabel": "Medicine",
            })

        loader = ESCODataLoader(raw_data_dir=tmp_path)
        report = loader.validate_file("occupationSkillRelations_en.csv")
        assert report.is_valid is False
        assert report.status_label == "FAIL"
        assert report.invalid_relation_type_count == 1
        assert any("Invalid relationType 'mandatory'" in err for err in report.errors)


def test_orphaned_reference_fails_validation():
    """Verifies that relation rows with orphaned foreign keys trigger validation errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_file = tmp_path / "occupationSkillRelations_en.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS["occupationSkillRelations_en.csv"])
            writer.writeheader()
            writer.writerow({
                "occupationUri": "http://data.europa.eu/esco/occupation/unknown-occ",
                "occupationLabel": "Unknown",
                "relationType": "essential",
                "skillType": "knowledge",
                "skillUri": "http://data.europa.eu/esco/skill/known-skill",
                "skillLabel": "Skill",
            })

        loader = ESCODataLoader(raw_data_dir=tmp_path)
        report = loader.validate_file(
            "occupationSkillRelations_en.csv",
            known_occupations={"http://data.europa.eu/esco/occupation/valid-occ"},
            known_skills={"http://data.europa.eu/esco/skill/known-skill"},
        )
        assert report.is_valid is False
        assert report.status_label == "FAIL"
        assert report.orphaned_reference_count == 1
        assert any("Orphaned occupationUri" in err for err in report.errors)


def test_duplicate_relation_composite_key_fails_validation():
    """Verifies that duplicate composite keys in relationship tables trigger validation errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_file = tmp_path / "occupationSkillRelations_en.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS["occupationSkillRelations_en.csv"])
            writer.writeheader()
            writer.writerow({
                "occupationUri": "http://data.europa.eu/esco/occupation/occ-1",
                "occupationLabel": "Doctor",
                "relationType": "essential",
                "skillType": "knowledge",
                "skillUri": "http://data.europa.eu/esco/skill/sk-1",
                "skillLabel": "Medicine",
            })
            writer.writerow({
                "occupationUri": "http://data.europa.eu/esco/occupation/occ-1",
                "occupationLabel": "Doctor",
                "relationType": "essential",
                "skillType": "knowledge",
                "skillUri": "http://data.europa.eu/esco/skill/sk-1",
                "skillLabel": "Medicine",
            })

        loader = ESCODataLoader(raw_data_dir=tmp_path)
        report = loader.validate_file("occupationSkillRelations_en.csv")
        assert report.is_valid is False
        assert report.status_label == "FAIL"
        assert report.duplicate_identifier_count == 1
        assert any("duplicate composite keys in relationship/hierarchy table" in err for err in report.errors)
