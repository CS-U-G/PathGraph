"""Multi-stage lexical skill resolver grounded in ESCO v1.2.1."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz, process


def normalize_text(text: str) -> str:
    """Normalizes text by NFKD decomposition, lowercasing, punctuation removal, and whitespace collapsing."""
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class ResolutionResult:
    """Result of resolving a skill label or identifier against ESCO."""

    query: str
    status: str  # EXACT_ID | EXACT_LABEL | NORMALIZED_LABEL | NORMALIZED_ALIAS | FUZZY_MATCH | AMBIGUOUS | UNRESOLVED
    skill_uri: str | None
    preferred_label: str | None
    confidence: float
    matched_term: str | None = None
    candidates: list[tuple[str, str, float]] | None = None  # (uri, label, score)


class LexicalResolver:
    """Deterministic lexical resolver cascade with exact, normalized, and fuzzy matching."""

    def __init__(
        self,
        skills: dict[str, dict[str, Any]],
        fuzzy_threshold: float = 88.0,
        ambiguity_margin: float = 3.0,
    ) -> None:
        self.skills = skills
        self.fuzzy_threshold = fuzzy_threshold
        self.ambiguity_margin = ambiguity_margin

        # Indexing for lookup
        self.uri_index = skills
        self.exact_pref_index: dict[str, str] = {}
        self.exact_alt_index: dict[str, set[str]] = {}
        self.norm_pref_index: dict[str, str] = {}
        self.norm_alt_index: dict[str, set[str]] = {}

        # Distinct terms for fuzzy search: term -> list of (uri, is_pref)
        self.fuzzy_vocab: dict[str, list[tuple[str, bool]]] = {}

        for uri, skill in skills.items():
            pref = skill.get("preferredLabel", "").strip()
            if pref:
                pref_lower = pref.lower()
                self.exact_pref_index[pref_lower] = uri
                norm_p = normalize_text(pref)
                if norm_p:
                    self.norm_pref_index[norm_p] = uri
                    if norm_p not in self.fuzzy_vocab:
                        self.fuzzy_vocab[norm_p] = []
                    self.fuzzy_vocab[norm_p].append((uri, True))

            # altLabels
            alt_raw = skill.get("altLabels", "")
            if isinstance(alt_raw, list):
                alt_list = alt_raw
            elif isinstance(alt_raw, str):
                alt_list = [a.strip() for a in alt_raw.split("\n") if a.strip()]
            else:
                alt_list = []

            for alt in alt_list:
                alt_lower = alt.lower()
                if alt_lower not in self.exact_alt_index:
                    self.exact_alt_index[alt_lower] = set()
                self.exact_alt_index[alt_lower].add(uri)

                norm_a = normalize_text(alt)
                if norm_a:
                    if norm_a not in self.norm_alt_index:
                        self.norm_alt_index[norm_a] = set()
                    self.norm_alt_index[norm_a].add(uri)
                    if norm_a not in self.fuzzy_vocab:
                        self.fuzzy_vocab[norm_a] = []
                    self.fuzzy_vocab[norm_a].append((uri, False))

        self.fuzzy_keys = list(self.fuzzy_vocab.keys())

    def resolve(self, query: str, skill_id: str | None = None) -> ResolutionResult:
        """Resolves a skill input through the deterministic cascade."""
        clean_q = query.strip() if query else ""

        # Stage 1: Exact URI match
        if skill_id:
            skill_id_clean = skill_id.strip()
            if skill_id_clean in self.uri_index:
                pref = self.uri_index[skill_id_clean].get("preferredLabel", "")
                return ResolutionResult(
                    query=clean_q or skill_id_clean,
                    status="EXACT_ID",
                    skill_uri=skill_id_clean,
                    preferred_label=pref,
                    confidence=1.0,
                    matched_term=skill_id_clean,
                )

        if not clean_q:
            return ResolutionResult(
                query="",
                status="UNRESOLVED",
                skill_uri=None,
                preferred_label=None,
                confidence=0.0,
            )

        # Stage 2: Normalized exact match on preferredLabel
        norm_q = normalize_text(clean_q)
        if norm_q in self.norm_pref_index:
            target_uri = self.norm_pref_index[norm_q]
            pref = self.uri_index[target_uri].get("preferredLabel", "")
            return ResolutionResult(
                query=clean_q,
                status="NORMALIZED_LABEL",
                skill_uri=target_uri,
                preferred_label=pref,
                confidence=0.98,
                matched_term=norm_q,
            )

        # Stage 3: Normalized exact match on altLabels
        if norm_q in self.norm_alt_index:
            uris = list(self.norm_alt_index[norm_q])
            if len(uris) == 1:
                target_uri = uris[0]
                pref = self.uri_index[target_uri].get("preferredLabel", "")
                return ResolutionResult(
                    query=clean_q,
                    status="NORMALIZED_ALIAS",
                    skill_uri=target_uri,
                    preferred_label=pref,
                    confidence=0.92,
                    matched_term=norm_q,
                )
            else:
                # Ambiguous alias shared by multiple ESCO concepts
                cands = [(u, self.uri_index[u].get("preferredLabel", ""), 0.90) for u in uris]
                return ResolutionResult(
                    query=clean_q,
                    status="AMBIGUOUS",
                    skill_uri=None,
                    preferred_label=None,
                    confidence=0.0,
                    candidates=cands,
                )

        # Stage 4: RapidFuzz token matching
        top_matches = process.extract(
            norm_q,
            self.fuzzy_keys,
            scorer=fuzz.token_sort_ratio,
            limit=5,
        )

        if top_matches:
            best_term, best_score, _ = top_matches[0]
            if best_score >= self.fuzzy_threshold:
                # Check for ambiguity with second best match
                if len(top_matches) > 1:
                    second_term, second_score, _ = top_matches[1]
                    # If within ambiguity margin and pointing to a different skill concept
                    if (best_score - second_score) <= self.ambiguity_margin:
                        best_uris = {u for u, _ in self.fuzzy_vocab[best_term]}
                        second_uris = {u for u, _ in self.fuzzy_vocab[second_term]}
                        if best_uris != second_uris:
                            cands = [
                                (u, self.uri_index[u].get("preferredLabel", ""), float(sc))
                                for term, sc, _ in top_matches[:3]
                                for u, _ in self.fuzzy_vocab[term]
                            ]
                            return ResolutionResult(
                                query=clean_q,
                                status="AMBIGUOUS",
                                skill_uri=None,
                                preferred_label=None,
                                confidence=0.0,
                                candidates=cands[:5],
                            )

                # Confident unique fuzzy match
                target_uri, _ = self.fuzzy_vocab[best_term][0]
                pref = self.uri_index[target_uri].get("preferredLabel", "")
                return ResolutionResult(
                    query=clean_q,
                    status="FUZZY_MATCH",
                    skill_uri=target_uri,
                    preferred_label=pref,
                    confidence=round(best_score / 100.0, 4),
                    matched_term=best_term,
                )

        # Stage 5: Unresolved
        return ResolutionResult(
            query=clean_q,
            status="UNRESOLVED",
            skill_uri=None,
            preferred_label=None,
            confidence=0.0,
        )
