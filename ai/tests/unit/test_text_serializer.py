"""Comprehensive Automated Test Suite for ESCO Text Serialization Module.

Verifies:
1. clean_text functionality (whitespace normalization, null/nan sentinel detection).
2. parse_alt_labels functionality (newline splitting, deduplication, excluding preferred label).
3. ESCOTextSerializer.serialize_fields (with/without altLabels, with/without description).
4. Prevention of literal 'nan', 'none', and duplicate text.
5. Validation of mandatory preferredLabel and conceptUri.
6. serialize_skill, serialize_occupation, and serialize_record behavior.
7. serialize_dataset alignment, ordering preservation, and duplicate URI detection.
"""

from typing import Dict, List

import pytest

from pathgraph_ai.embeddings.text_serializer import (
    INVALID_TEXT_MARKERS,
    ESCOTextSerializer,
    clean_text,
    parse_alt_labels,
)


# -----------------------------------------------------------------------------
# Clean Text Tests
# -----------------------------------------------------------------------------


class TestCleanText:
    """Verifies text normalization and sentinel null/nan value filtering."""

    def test_clean_valid_string(self):
        assert clean_text("Python Developer") == "Python Developer"

    def test_collapses_internal_whitespace(self):
        assert clean_text("  software   engineer \n with \t python  ") == "software engineer with python"

    def test_none_returns_none(self):
        assert clean_text(None) is None

    def test_empty_string_returns_none(self):
        assert clean_text("") is None

    def test_whitespace_string_returns_none(self):
        assert clean_text("   \t  \n  ") is None

    @pytest.mark.parametrize(
        "marker",
        ["nan", "NaN", "NAN", "None", "NONE", "null", "NULL", "n/a", "N/A", "na", "undefined"],
    )
    def test_sentinel_markers_return_none(self, marker: str):
        assert clean_text(marker) is None

    @pytest.mark.parametrize(
        "marker",
        ["nan", "None", "null", "N/A"],
    )
    def test_sentinel_markers_with_whitespace_return_none(self, marker: str):
        assert clean_text(f"  {marker}  ") is None

    def test_converts_numeric_to_str(self):
        assert clean_text(123) == "123"


# -----------------------------------------------------------------------------
# Parse Alt Labels Tests
# -----------------------------------------------------------------------------


class TestParseAltLabels:
    """Verifies splitting, deduplication, and exclusion of preferred labels."""

    def test_parses_newline_separated_string(self):
        raw = "music supervisor\nhead of music\nmusical director"
        res = parse_alt_labels(raw)
        assert res == ["music supervisor", "head of music", "musical director"]

    def test_parses_crlf_separated_string(self):
        raw = "music supervisor\r\nhead of music\r\nmusical director"
        res = parse_alt_labels(raw)
        assert res == ["music supervisor", "head of music", "musical director"]

    def test_parses_sequence_of_strings(self):
        raw = ["music supervisor", "head of music"]
        res = parse_alt_labels(raw)
        assert res == ["music supervisor", "head of music"]

    def test_deduplicates_case_insensitively_preserving_first_occurrence(self):
        raw = "Python Developer\npython developer\nPYTHON DEVELOPER\nCoder"
        res = parse_alt_labels(raw)
        assert res == ["Python Developer", "Coder"]

    def test_excludes_preferred_label_case_insensitively(self):
        pref = "Software Engineer"
        raw = "software engineer\nSenior Developer\nSoftware Engineer\nProgrammer"
        res = parse_alt_labels(raw, exclude_label=pref)
        assert res == ["Senior Developer", "Programmer"]

    def test_filters_empty_lines_and_nan_markers(self):
        raw = "Dev\n\n  \nnan\nNone\nnull\nArch"
        res = parse_alt_labels(raw)
        assert res == ["Dev", "Arch"]

    def test_none_and_empty_inputs_return_empty_list(self):
        assert parse_alt_labels(None) == []
        assert parse_alt_labels("") == []
        assert parse_alt_labels("   ") == []
        assert parse_alt_labels([]) == []


# -----------------------------------------------------------------------------
# ESCOTextSerializer Field and Record Tests
# -----------------------------------------------------------------------------


class TestESCOTextSerializer:
    """Verifies structured text serialization for skills and occupations."""

    def setup_method(self):
        self.serializer = ESCOTextSerializer()

    def test_serialize_all_fields_present(self):
        pref = "machine learning engineer"
        alts = "ML engineer\nAI practitioner"
        desc = "Design and build scalable machine learning architectures."

        text = self.serializer.serialize_fields(
            preferred_label=pref,
            alt_labels=alts,
            description=desc,
        )

        expected = (
            "machine learning engineer\n"
            "Alternative labels: ML engineer, AI practitioner\n"
            "Design and build scalable machine learning architectures."
        )
        assert text == expected

    def test_serialize_missing_alt_labels(self):
        pref = "cloud architect"
        desc = "Design cloud infrastructure."

        text = self.serializer.serialize_fields(
            preferred_label=pref,
            alt_labels=None,
            description=desc,
        )
        assert text == "cloud architect\nDesign cloud infrastructure."

    def test_serialize_empty_alt_labels(self):
        pref = "cloud architect"
        text = self.serializer.serialize_fields(
            preferred_label=pref,
            alt_labels="   \nnan\n",
            description="Design cloud infrastructure.",
        )
        assert text == "cloud architect\nDesign cloud infrastructure."

    def test_serialize_missing_description(self):
        pref = "database administrator"
        alts = "DBA\ndatabase manager"

        text = self.serializer.serialize_fields(
            preferred_label=pref,
            alt_labels=alts,
            description=None,
        )
        assert text == "database administrator\nAlternative labels: DBA, database manager"

    def test_serialize_nan_description(self):
        pref = "database administrator"
        text = self.serializer.serialize_fields(
            preferred_label=pref,
            alt_labels=None,
            description="nan",
        )
        assert text == "database administrator"
        assert "nan" not in text

    def test_serialize_only_preferred_label(self):
        text = self.serializer.serialize_fields(
            preferred_label="data analyst",
            alt_labels=None,
            description=None,
        )
        assert text == "data analyst"

    def test_omits_description_if_identical_to_preferred_label(self):
        text = self.serializer.serialize_fields(
            preferred_label="Software Tester",
            alt_labels=None,
            description="software tester",
        )
        assert text == "Software Tester"

    def test_serialize_custom_separator(self):
        custom_serializer = ESCOTextSerializer(separator=". ")
        text = custom_serializer.serialize_fields(
            preferred_label="data analyst",
            alt_labels="business analyst",
            description="Analyze tabular datasets.",
        )
        assert text == "data analyst. Alternative labels: business analyst. Analyze tabular datasets."

    def test_serialize_without_alt_prefix(self):
        custom_serializer = ESCOTextSerializer(include_alt_prefix=False)
        text = custom_serializer.serialize_fields(
            preferred_label="data analyst",
            alt_labels="business analyst, reporting analyst",
        )
        assert text == "data analyst\nbusiness analyst, reporting analyst"

    def test_missing_preferred_label_raises_value_error(self):
        with pytest.raises(ValueError, match="Entity must have a valid non-empty preferredLabel"):
            self.serializer.serialize_fields(preferred_label="")

        with pytest.raises(ValueError, match="Entity must have a valid non-empty preferredLabel"):
            self.serializer.serialize_fields(preferred_label="   ")

        with pytest.raises(ValueError, match="Entity must have a valid non-empty preferredLabel"):
            self.serializer.serialize_fields(preferred_label="nan")

        with pytest.raises(ValueError, match="Entity must have a valid non-empty preferredLabel"):
            self.serializer.serialize_fields(preferred_label=None)

    def test_serialize_skill_record(self):
        skill_record = {
            "conceptUri": "http://data.europa.eu/esco/skill/12345",
            "preferredLabel": "manage musical staff",
            "altLabels": "manage music staff\ndirect musical staff",
            "description": "Coordinate musical staff across productions.",
        }
        res = self.serializer.serialize_skill(skill_record)
        assert "manage musical staff" in res
        assert "Alternative labels: manage music staff, direct musical staff" in res
        assert "Coordinate musical staff" in res

    def test_serialize_skill_record_invalid_type(self):
        with pytest.raises(TypeError, match="Skill record must be a dict"):
            self.serializer.serialize_skill("not a dict")

    def test_serialize_occupation_record(self):
        occ_record = {
            "conceptUri": "http://data.europa.eu/esco/occupation/67890",
            "preferredLabel": "technical director",
            "altLabels": "director of technical arts\nhead of technical",
            "description": "Realise artistic visions within technical constraints.",
        }
        res = self.serializer.serialize_occupation(occ_record)
        assert "technical director" in res
        assert "Alternative labels: director of technical arts, head of technical" in res
        assert "Realise artistic visions" in res

    def test_serialize_occupation_record_invalid_type(self):
        with pytest.raises(TypeError, match="Occupation record must be a dict"):
            self.serializer.serialize_occupation(None)

    def test_serialize_record_dispatch(self):
        skill_record = {
            "conceptType": "Skill",
            "preferredLabel": "python programming",
        }
        assert self.serializer.serialize_record(skill_record) == "python programming"

        occ_record = {
            "conceptType": "Occupation",
            "preferredLabel": "software architect",
        }
        assert self.serializer.serialize_record(occ_record) == "software architect"


# -----------------------------------------------------------------------------
# Dataset Serialization Tests
# -----------------------------------------------------------------------------


class TestSerializeDataset:
    """Verifies dataset-level batch serialization, alignment, and uniqueness."""

    def setup_method(self):
        self.serializer = ESCOTextSerializer()

    def test_serialize_dataset_success(self):
        records = [
            {
                "conceptUri": "http://esco/skill/1",
                "preferredLabel": "Python",
                "altLabels": "Python 3\nPython language",
                "description": "Programming language.",
            },
            {
                "conceptUri": "http://esco/skill/2",
                "preferredLabel": "SQL",
                "altLabels": None,
                "description": "Database query language.",
            },
        ]
        pairs = self.serializer.serialize_dataset(records, entity_type="skill")
        assert len(pairs) == 2
        assert pairs[0][0] == "http://esco/skill/1"
        assert "Python" in pairs[0][1]
        assert pairs[1][0] == "http://esco/skill/2"
        assert "SQL" in pairs[1][1]

    def test_empty_dataset_raises_value_error(self):
        with pytest.raises(ValueError, match="Records collection cannot be empty"):
            self.serializer.serialize_dataset([])

    def test_non_list_records_raises_type_error(self):
        with pytest.raises(TypeError, match="Records must be a list or tuple"):
            self.serializer.serialize_dataset("invalid")

    def test_record_not_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="Record at index 0 must be a dict"):
            self.serializer.serialize_dataset(["invalid"])

    def test_missing_uri_raises_value_error(self):
        records = [{"preferredLabel": "Python"}]
        with pytest.raises(ValueError, match="Record at index 0 missing valid conceptUri"):
            self.serializer.serialize_dataset(records)

    def test_nan_uri_raises_value_error(self):
        records = [{"conceptUri": "nan", "preferredLabel": "Python"}]
        with pytest.raises(ValueError, match="Record at index 0 missing valid conceptUri"):
            self.serializer.serialize_dataset(records)

    def test_strict_mode_rejects_duplicate_uris(self):
        records = [
            {"conceptUri": "http://esco/skill/1", "preferredLabel": "Python"},
            {"conceptUri": "http://esco/skill/1", "preferredLabel": "Python duplicate"},
        ]
        with pytest.raises(ValueError, match="Duplicate conceptUri encountered at index 1"):
            self.serializer.serialize_dataset(records, strict_unique_keys=True)
