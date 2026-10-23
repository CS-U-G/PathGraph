"""ESCO Text Serialization Module for PathGraph.

Prepares text representations of ESCO skills and occupations for dense vector
embedding precomputation. Preserves stable ESCO URIs, extracts and normalizes
preferred labels, alternative labels, and descriptions, and safely handles missing,
dirty, or duplicate values without injecting literal 'nan' strings.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# String representations of invalid / missing / NaN markers across datasets
INVALID_TEXT_MARKERS: frozenset[str] = frozenset(
    {"nan", "none", "null", "n/a", "na", "undefined", ""}
)


def clean_text(text: str | None) -> str | None:
    """Cleans and normalizes a text string.

    Strips leading and trailing whitespace, collapses multiple internal whitespace
    characters into a single space, and checks against sentinel values for missing or
    NaN markers.

    Args:
        text: Input string or None.

    Returns:
        Cleaned string, or None if the input is None, empty, whitespace-only,
        or a sentinel value such as 'nan', 'none', or 'null'.
    """
    if text is None:
        return None

    if not isinstance(text, str):
        text = str(text)

    stripped = text.strip()
    if not stripped or stripped.lower() in INVALID_TEXT_MARKERS:
        return None

    # Collapse internal whitespace (tabs, newlines, multiple spaces)
    normalized = " ".join(stripped.split())
    if not normalized or normalized.lower() in INVALID_TEXT_MARKERS:
        return None

    return normalized


def parse_alt_labels(
    alt_labels_raw: str | Sequence[str] | None,
    exclude_label: str | None = None,
) -> list[str]:
    """Parses, deduplicates, and filters alternative labels.

    Handles newline-delimited strings (standard ESCO CSV format) or sequences of
    strings. Removes empty entries, sentinel missing values ('nan', 'none'),
    internal duplicates, and any candidate that duplicates the preferred label
    (case-insensitive).

    Args:
        alt_labels_raw: Raw alternative labels as a newline-separated string
                        or a sequence of strings.
        exclude_label: Optional label (e.g. preferredLabel) to exclude from altLabels
                       to prevent textual duplication.

    Returns:
        List of cleaned, unique alternative label strings preserving original order.
    """
    if alt_labels_raw is None:
        return []

    # Determine candidates from newline-separated string or sequence
    if isinstance(alt_labels_raw, str):
        raw_candidates = alt_labels_raw.replace("\r\n", "\n").split("\n")
    elif isinstance(alt_labels_raw, (list, tuple, set)):
        raw_candidates = list(alt_labels_raw)
    else:
        raw_candidates = [str(alt_labels_raw)]

    exclude_norm = exclude_label.strip().lower() if exclude_label else None

    cleaned_labels: list[str] = []
    seen: set[str] = set()

    for item in raw_candidates:
        cleaned = clean_text(item)
        if cleaned is None:
            continue

        item_norm = cleaned.lower()

        # Exclude candidate if it duplicates the preferred label
        if exclude_norm and item_norm == exclude_norm:
            continue

        # Exclude candidate if already encountered (case-insensitive deduplication)
        if item_norm in seen:
            continue

        seen.add(item_norm)
        cleaned_labels.append(cleaned)

    return cleaned_labels


class ESCOTextSerializer:
    """Dedicated serializer for preparing ESCO entity records for semantic embedding."""

    def __init__(
        self,
        separator: str = "\n",
        include_alt_prefix: bool = True,
        alt_prefix: str = "Alternative labels: ",
    ) -> None:
        """Initializes the serializer with formatting options.

        Args:
            separator: Character or string used to separate sections (default '\n').
            include_alt_prefix: Whether to prepend a prefix to alternative labels.
            alt_prefix: Prefix text for alternative labels (default 'Alternative labels: ').
        """
        self.separator = separator
        self.include_alt_prefix = include_alt_prefix
        self.alt_prefix = alt_prefix

    def serialize_fields(
        self,
        preferred_label: str,
        alt_labels: str | Sequence[str] | None = None,
        description: str | None = None,
    ) -> str:
        """Serializes entity fields into a structured text string.

        Args:
            preferred_label: The preferred label (mandatory).
            alt_labels: Optional raw alternative labels (newline-separated or list).
            description: Optional functional description.

        Returns:
            Clean serialized text representation suitable for dense embeddings.

        Raises:
            ValueError: If preferred_label is None, empty, whitespace-only, or a NaN marker.
        """
        clean_pref = clean_text(preferred_label)
        if clean_pref is None:
            raise ValueError(
                f"Entity must have a valid non-empty preferredLabel, got {preferred_label!r}."
            )

        # Parse altLabels excluding the preferred label to avoid duplication
        clean_alts = parse_alt_labels(alt_labels, exclude_label=clean_pref)

        # Clean description
        clean_desc = clean_text(description)

        # Avoid redundant description if it's identical to preferred label
        if clean_desc and clean_desc.lower() == clean_pref.lower():
            clean_desc = None

        parts: list[str] = [clean_pref]

        if clean_alts:
            alt_text = ", ".join(clean_alts)
            if self.include_alt_prefix:
                alt_text = f"{self.alt_prefix}{alt_text}"
            parts.append(alt_text)

        if clean_desc:
            parts.append(clean_desc)

        return self.separator.join(parts)

    def serialize_skill(self, record: dict[str, Any]) -> str:
        """Serializes an ESCO skill record dictionary into embedding text.

        Args:
            record: Dictionary containing skill attributes ('conceptUri',
                    'preferredLabel', 'altLabels', 'description').

        Returns:
            Serialized text representation.

        Raises:
            ValueError: If record is not a dictionary or lacks a valid preferredLabel.
        """
        if not isinstance(record, dict):
            raise TypeError(f"Skill record must be a dict, got {type(record).__name__}.")

        return self.serialize_fields(
            preferred_label=record.get("preferredLabel", ""),
            alt_labels=record.get("altLabels"),
            description=record.get("description"),
        )

    def serialize_occupation(self, record: dict[str, Any]) -> str:
        """Serializes an ESCO occupation record dictionary into embedding text.

        Args:
            record: Dictionary containing occupation attributes ('conceptUri',
                    'preferredLabel', 'altLabels', 'description').

        Returns:
            Serialized text representation.

        Raises:
            ValueError: If record is not a dictionary or lacks a valid preferredLabel.
        """
        if not isinstance(record, dict):
            raise TypeError(f"Occupation record must be a dict, got {type(record).__name__}.")

        return self.serialize_fields(
            preferred_label=record.get("preferredLabel", ""),
            alt_labels=record.get("altLabels"),
            description=record.get("description"),
        )

    def serialize_record(
        self, record: dict[str, Any], entity_type: str | None = None
    ) -> str:
        """Serializes any ESCO record dictionary (skill or occupation).

        Args:
            record: Record dictionary.
            entity_type: Optional explicit type ('skill' or 'occupation').
                         If omitted, detects from 'conceptType' or defaults to generic field serialization.

        Returns:
            Serialized text representation.
        """
        if not isinstance(record, dict):
            raise TypeError(f"Record must be a dict, got {type(record).__name__}.")

        etype = (entity_type or record.get("conceptType") or "").lower()
        if "skill" in etype or "competence" in etype:
            return self.serialize_skill(record)
        if "occ" in etype:
            return self.serialize_occupation(record)

        return self.serialize_fields(
            preferred_label=record.get("preferredLabel", ""),
            alt_labels=record.get("altLabels"),
            description=record.get("description"),
        )

    def serialize_dataset(
        self,
        records: Sequence[dict[str, Any]],
        entity_type: str | None = None,
        key_field: str = "conceptUri",
        strict_unique_keys: bool = True,
    ) -> list[tuple[str, str]]:
        """Serializes a collection of records into aligned (key, serialized_text) pairs.

        Args:
            records: Sequence of record dictionaries.
            entity_type: Optional explicit type ('skill' or 'occupation').
            key_field: Primary key field name (default 'conceptUri').
            strict_unique_keys: If True, raises ValueError upon encountering duplicate keys.

        Returns:
            List of (concept_uri, serialized_text) tuples preserving original sequence order.

        Raises:
            ValueError: If records is empty, any record has a missing/empty key,
                        or duplicate keys are found under strict mode.
        """
        if not isinstance(records, (list, tuple)):
            raise TypeError(f"Records must be a list or tuple, got {type(records).__name__}.")

        if len(records) == 0:
            raise ValueError("Records collection cannot be empty.")

        serialized_pairs: list[tuple[str, str]] = []
        seen_keys: set[str] = set()

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise TypeError(
                    f"Record at index {idx} must be a dict, got {type(rec).__name__}."
                )

            raw_key = rec.get(key_field)
            clean_key = clean_text(raw_key)
            if clean_key is None:
                raise ValueError(
                    f"Record at index {idx} missing valid {key_field}: {raw_key!r}."
                )

            if clean_key in seen_keys:
                msg = f"Duplicate {key_field} encountered at index {idx}: '{clean_key}'."
                if strict_unique_keys:
                    raise ValueError(msg)
                logger.warning(msg)

            seen_keys.add(clean_key)
            text = self.serialize_record(rec, entity_type=entity_type)
            serialized_pairs.append((clean_key, text))

        return serialized_pairs
