"""Skill Gap Analysis Module for PathGraph.

Compares a user's extracted skills against the required (essential and optional)
skills of a target occupation identified from ESCO relations.

Supports:
1. Exact URI matching (analyze_gap_exact): deterministic, strict set membership.
2. Embedding-aware semantic matching (analyze_gap_semantic / analyze_gap):
   evaluates cosine similarities using precomputed offline embeddings without
   loading heavy transformer models.
3. Fast matrix-vector similarity search (find_similar_skills, find_similar_by_vector):
   NumPy-accelerated dot product over unit-normalized vectors.

CRITICAL DESIGN PRINCIPLES:
- Exact URI matches are reported separately and retain their definitive status.
- Semantic similarity surfaces candidate suggestions for review; it does NOT
  certify user proficiency, does not imply prerequisite order, and does NOT
  silently remove required skills from the verified gap.
- Thresholds are uncalibrated heuristic filters, not proven decision boundaries.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pathgraph_ai.embeddings.artifact import (
    EmbeddingArtifact,
    load_embeddings_artifact,
)
from pathgraph_ai.settings import settings

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class MatchedSkill:
    """Represents a verified satisfied skill requirement."""

    skill_uri: str
    skill_label: str
    relation_type: str  # 'essential' | 'optional'
    match_source: str  # 'exact_uri' | 'label_match' | 'semantic'
    similarity_score: float = 1.0


@dataclass
class MissingSkill:
    """Represents an unfulfilled skill requirement."""

    skill_uri: str
    skill_label: str
    relation_type: str  # 'essential' | 'optional'
    skill_type: str | None = None  # 'knowledge' | 'skill/competence'


@dataclass
class SemanticCandidate:
    """Represents a potential semantic skill match or related skill suggestion.

    IMPORTANT:
    Semantic similarity indicates semantic proximity or topical relatedness.
    It does NOT certify user proficiency, does not imply prerequisite order,
    and must NOT automatically remove a required skill from the verified gap.
    """

    required_skill_uri: str
    required_skill_label: str
    candidate_skill_uri: str
    candidate_skill_label: str
    similarity_score: float
    relation_type: str  # 'essential' | 'optional'
    matching_method: str = "precomputed_cosine"


@dataclass
class SkillGapReport:
    """Detailed summary of the gap analysis between user skills and target occupation."""

    occupation_uri: str
    occupation_label: str
    total_required_skills: int
    matched_essential: list[MatchedSkill] = field(default_factory=list)
    missing_essential: list[MissingSkill] = field(default_factory=list)
    matched_optional: list[MatchedSkill] = field(default_factory=list)
    missing_optional: list[MissingSkill] = field(default_factory=list)
    semantic_candidates: list[SemanticCandidate] = field(default_factory=list)

    @property
    def essential_match_percentage(self) -> float:
        """Percentage of essential skills satisfied via verified exact matches."""
        total = len(self.matched_essential) + len(self.missing_essential)
        return (len(self.matched_essential) / total * 100.0) if total > 0 else 0.0

    @property
    def overall_match_percentage(self) -> float:
        """Percentage of all required skills satisfied via verified exact matches."""
        matched = len(self.matched_essential) + len(self.matched_optional)
        return (matched / self.total_required_skills * 100.0) if self.total_required_skills > 0 else 0.0

    @property
    def candidates_by_required_skill(self) -> dict[str, list[SemanticCandidate]]:
        """Groups semantic candidates by required_skill_uri for review."""
        result: dict[str, list[SemanticCandidate]] = defaultdict(list)
        for cand in self.semantic_candidates:
            result[cand.required_skill_uri].append(cand)
        return dict(result)


class SkillGapAnalyzer:
    """Analyzes gaps between acquired user skills and target occupation skill profiles.

    Supports:
    1. Exact URI matching (analyze_gap_exact): deterministic, strict set membership.
    2. Embedding-aware semantic matching (analyze_gap_semantic / analyze_gap):
       computes cosine similarities using precomputed offline embeddings without
       loading heavy transformer models. Semantic matches are surfaced as candidate
       suggestions for review and do not certify proficiency or silently alter
       the verified gap.
    3. Fast similarity search (find_similar_skills): matrix-vector dot-product
       lookup over precomputed unit-normalized embeddings.
    """

    UNRECALIBRATED_PROVISIONAL_THRESHOLD: float = 0.85

    def __init__(
        self,
        knowledge_graph: Any | None = None,
        skills_artifact: EmbeddingArtifact | None = None,
        artifact_path: str | Path | None = None,
    ) -> None:
        """Initializes the SkillGapAnalyzer.

        Args:
            knowledge_graph: Optional ESCOKnowledgeGraph instance for graph queries.
            skills_artifact: Optional pre-loaded EmbeddingArtifact instance.
            artifact_path: Optional path to skills_embeddings.npz. Defaults to settings.skills_embeddings_path.
        """
        self.kg = knowledge_graph
        self._skills_artifact: EmbeddingArtifact | None = skills_artifact
        self._artifact_path: Path | None = Path(artifact_path) if artifact_path else None

        if self._skills_artifact is not None:
            self._validate_artifact(self._skills_artifact)

    @property
    def skills_artifact(self) -> EmbeddingArtifact:
        """Lazily loads and returns the skills embedding artifact."""
        if self._skills_artifact is None:
            self._load_default_or_path_artifact()
        assert self._skills_artifact is not None
        return self._skills_artifact

    def _load_default_or_path_artifact(self) -> None:
        if np is None:
            raise ImportError("NumPy is required to load and utilize embedding artifacts.")

        target_path = self._artifact_path or settings.skills_embeddings_path
        target_path = Path(target_path).resolve()
        if not target_path.exists():
            raise FileNotFoundError(
                f"Skills embedding artifact not found at '{target_path}'. "
                "Ensure precomputed embeddings have been placed or inject an EmbeddingArtifact."
            )

        artifact = load_embeddings_artifact(target_path)
        self._validate_artifact(artifact)
        self._skills_artifact = artifact

    @staticmethod
    def _validate_artifact(artifact: Any) -> None:
        """Validates that an EmbeddingArtifact is structural, finite, and consistent."""
        if artifact is None:
            raise ValueError("Embedding artifact cannot be None.")
        if not hasattr(artifact, "embeddings") or not hasattr(artifact, "uris"):
            raise ValueError("Artifact must have 'embeddings' and 'uris' attributes.")

        emb = artifact.embeddings
        if not isinstance(emb, np.ndarray):
            emb = np.asarray(emb)

        if emb.ndim != 2:
            raise ValueError(f"Artifact embeddings must be a 2D matrix (N, D), got ndim={emb.ndim}.")

        num_rows, dim = emb.shape
        if num_rows == 0 or dim == 0:
            raise ValueError(f"Artifact embeddings cannot be empty, got shape {emb.shape}.")

        if not np.all(np.isfinite(emb)):
            raise ValueError("Artifact embeddings contain non-finite values (NaN or Inf).")

        if len(artifact.uris) != num_rows:
            raise ValueError(
                f"Artifact row count mismatch: {num_rows} embedding rows but {len(artifact.uris)} URIs."
            )

    def analyze_gap_exact(
        self,
        occupation_uri: str,
        occupation_label: str,
        required_skills: list[dict[str, str]],
        user_skill_uris: set[str],
    ) -> SkillGapReport:
        """Performs exact identifier comparison between user skills and target occupation skills."""
        report = SkillGapReport(
            occupation_uri=occupation_uri,
            occupation_label=occupation_label,
            total_required_skills=len(required_skills),
        )

        for skill in required_skills:
            s_uri = skill.get("skillUri", "")
            s_label = skill.get("skillLabel", "")
            rel_type = skill.get("relationType", "optional").lower()
            s_type = skill.get("skillType")

            if s_uri in user_skill_uris:
                matched = MatchedSkill(
                    skill_uri=s_uri,
                    skill_label=s_label,
                    relation_type=rel_type,
                    match_source="exact_uri",
                    similarity_score=1.0,
                )
                if rel_type == "essential":
                    report.matched_essential.append(matched)
                else:
                    report.matched_optional.append(matched)
            else:
                missing = MissingSkill(
                    skill_uri=s_uri,
                    skill_label=s_label,
                    relation_type=rel_type,
                    skill_type=s_type,
                )
                if rel_type == "essential":
                    report.missing_essential.append(missing)
                else:
                    report.missing_optional.append(missing)

        return report

    def find_similar_by_vector(
        self,
        query_vector: Any,
        top_k: int = 5,
        min_similarity: float | None = None,
        candidate_uris: Sequence[str] | None = None,
        exclude_uri: str | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieves top-k similar skills from precomputed matrix for a given query vector.

        Uses vector-matrix dot product over unit-normalized embeddings with NumPy.
        Tie-breaking is deterministic (descending similarity, then ascending URI).
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        artifact = self.skills_artifact
        q_vec = np.asarray(query_vector, dtype=np.float32)

        if q_vec.ndim > 1:
            q_vec = np.squeeze(q_vec)
        if q_vec.ndim != 1:
            raise ValueError(f"query_vector must be 1D, got shape {q_vec.shape}")

        if q_vec.shape[0] != artifact.dimension:
            raise ValueError(
                f"Dimension mismatch: query_vector dimension {q_vec.shape[0]} "
                f"does not match artifact dimension {artifact.dimension}."
            )

        if not np.all(np.isfinite(q_vec)):
            raise ValueError("query_vector contains non-finite values (NaN or Inf).")

        norm = float(np.linalg.norm(q_vec))
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError("query_vector has zero or invalid norm.")

        if not np.isclose(norm, 1.0, atol=1e-5):
            q_vec = q_vec / norm

        if candidate_uris is not None:
            seen: set[str] = set()
            valid_cands: list[str] = []
            cand_indices: list[int] = []
            for u in candidate_uris:
                if u in artifact.uri_to_index and u not in seen:
                    seen.add(u)
                    if exclude_uri is None or u != exclude_uri:
                        valid_cands.append(u)
                        cand_indices.append(artifact.uri_to_index[u])

            if not valid_cands:
                return []

            sub_matrix = artifact.embeddings[cand_indices]
            scores = sub_matrix @ q_vec

            scored_items: list[tuple[str, float]] = []
            for uri, score in zip(valid_cands, scores):
                s_float = float(score)
                if min_similarity is not None and s_float < min_similarity:
                    continue
                scored_items.append((uri, s_float))

            scored_items.sort(key=lambda item: (-item[1], item[0]))
            return scored_items[:top_k]

        else:
            scores = artifact.embeddings @ q_vec
            exclude_idx = artifact.uri_to_index.get(exclude_uri) if exclude_uri else None
            if exclude_idx is not None:
                scores[exclude_idx] = -np.inf

            n_items = len(artifact.uris)
            if min_similarity is not None:
                qualifying_mask = scores >= min_similarity
                if exclude_idx is not None:
                    qualifying_mask[exclude_idx] = False
                qualifying_indices = np.where(qualifying_mask)[0]
                if len(qualifying_indices) == 0:
                    return []
                scored_items = [
                    (artifact.uris[idx], float(scores[idx]))
                    for idx in qualifying_indices
                ]
                scored_items.sort(key=lambda item: (-item[1], item[0]))
                return scored_items[:top_k]
            else:
                k_bound = min(top_k, n_items - (1 if exclude_idx is not None else 0))
                if k_bound <= 0:
                    return []

                part_indices = np.argpartition(-scores, k_bound - 1)[:k_bound]
                cutoff_score = scores[part_indices[-1]]
                candidate_indices = np.where(scores >= cutoff_score)[0]
                scored_items = [
                    (artifact.uris[idx], float(scores[idx]))
                    for idx in candidate_indices
                    if not np.isneginf(scores[idx])
                ]
                scored_items.sort(key=lambda item: (-item[1], item[0]))
                return scored_items[:top_k]

    def find_similar_skills(
        self,
        skill_uri: str,
        top_k: int = 5,
        min_similarity: float | None = None,
        candidate_uris: Sequence[str] | None = None,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """Retrieves top-k similar ESCO skills to a given skill URI using precomputed embeddings."""
        artifact = self.skills_artifact
        if skill_uri not in artifact.uri_to_index:
            raise KeyError(f"Skill URI '{skill_uri}' not found in precomputed embedding artifact.")

        query_vec = artifact.get_embedding(skill_uri)
        if query_vec is None:
            raise KeyError(f"Skill URI '{skill_uri}' has no embedding vector in artifact.")

        exclude = skill_uri if exclude_self else None
        return self.find_similar_by_vector(
            query_vector=query_vec,
            top_k=top_k,
            min_similarity=min_similarity,
            candidate_uris=candidate_uris,
            exclude_uri=exclude,
        )

    def analyze_gap_semantic(
        self,
        occupation_uri: str,
        user_skill_uris: set[str],
        occupation_label: str | None = None,
        required_skills: list[dict[str, Any]] | None = None,
        top_k_candidates: int = 3,
        similarity_threshold: float | None = None,
        candidate_pool: str = "user_skills",
    ) -> SkillGapReport:
        """Performs embedding-aware skill gap analysis.

        Exact matches are reported separately in `matched_essential` and `matched_optional`.
        Semantic similarity is evaluated to identify potential candidate matches for
        review in `semantic_candidates`.
        """
        # Resolve required skills from KG if not explicitly passed
        if required_skills is None:
            if not self.kg:
                raise ValueError("Either required_skills or a knowledge_graph must be provided.")
            detailed_rels = self.kg.get_occupation_skills_detailed(occupation_uri)
            if not detailed_rels and hasattr(self.kg, "occupations") and occupation_uri not in self.kg.occupations:
                raise KeyError(f"Occupation URI '{occupation_uri}' not found in knowledge graph.")

            req_list: list[dict[str, Any]] = []
            for r in detailed_rels:
                s_node = self.kg.get_node(r.skill_uri)
                label = r.skill_label or (s_node.label if s_node and hasattr(s_node, "label") else r.skill_uri)
                req_list.append(
                    {
                        "skillUri": r.skill_uri,
                        "skillLabel": label,
                        "relationType": r.relation_type,
                        "skillType": r.skill_type or (s_node.skill_type if s_node and hasattr(s_node, "skill_type") else None),
                    }
                )
            required_skills = req_list

        if occupation_label is None:
            if self.kg:
                node = self.kg.get_node(occupation_uri)
                if node and hasattr(node, "label") and node.label:
                    occupation_label = node.label
                elif hasattr(self.kg, "occupations") and occupation_uri in self.kg.occupations:
                    occupation_label = self.kg.occupations[occupation_uri].label
                else:
                    occupation_label = occupation_uri
            else:
                occupation_label = occupation_uri

        report = self.analyze_gap_exact(
            occupation_uri=occupation_uri,
            occupation_label=occupation_label,
            required_skills=required_skills,
            user_skill_uris=user_skill_uris,
        )

        missing_list: list[MissingSkill] = report.missing_essential + report.missing_optional
        if not missing_list:
            return report

        artifact = self.skills_artifact

        if candidate_pool == "user_skills":
            cand_pool_uris = [u for u in user_skill_uris if u in artifact.uri_to_index]
            if not cand_pool_uris:
                return report
        elif candidate_pool == "catalog":
            cand_pool_uris = None
        else:
            raise ValueError(
                f"Invalid candidate_pool '{candidate_pool}'. Expected 'user_skills' or 'catalog'."
            )

        threshold = similarity_threshold

        for missing in missing_list:
            req_uri = missing.skill_uri
            if req_uri not in artifact.uri_to_index:
                continue

            top_matches = self.find_similar_skills(
                skill_uri=req_uri,
                top_k=top_k_candidates,
                min_similarity=threshold,
                candidate_uris=cand_pool_uris,
                exclude_self=True,
            )

            for cand_uri, sim_score in top_matches:
                cand_label = cand_uri
                if self.kg:
                    node = self.kg.get_node(cand_uri)
                    if node and hasattr(node, "label"):
                        cand_label = node.label

                cand = SemanticCandidate(
                    required_skill_uri=req_uri,
                    required_skill_label=missing.skill_label,
                    candidate_skill_uri=cand_uri,
                    candidate_skill_label=cand_label,
                    similarity_score=round(sim_score, 4),
                    relation_type=missing.relation_type,
                    matching_method="precomputed_cosine",
                )
                report.semantic_candidates.append(cand)

        return report

    def analyze_gap(
        self,
        occupation_uri: str,
        user_skill_uris: set[str],
        occupation_label: str | None = None,
        required_skills: list[dict[str, Any]] | None = None,
        top_k_candidates: int = 3,
        similarity_threshold: float | None = None,
        candidate_pool: str = "user_skills",
        enable_semantic_matching: bool = True,
    ) -> SkillGapReport:
        """Unified entry point for skill gap analysis."""
        if not enable_semantic_matching:
            if required_skills is None:
                if not self.kg:
                    raise ValueError("Either required_skills or a knowledge_graph must be provided.")
                detailed_rels = self.kg.get_occupation_skills_detailed(occupation_uri)
                if not detailed_rels and hasattr(self.kg, "occupations") and occupation_uri not in self.kg.occupations:
                    raise KeyError(f"Occupation URI '{occupation_uri}' not found in knowledge graph.")
                req_list: list[dict[str, Any]] = []
                for r in detailed_rels:
                    s_node = self.kg.get_node(r.skill_uri)
                    label = r.skill_label or (s_node.label if s_node and hasattr(s_node, "label") else r.skill_uri)
                    req_list.append(
                        {
                            "skillUri": r.skill_uri,
                            "skillLabel": label,
                            "relationType": r.relation_type,
                            "skillType": r.skill_type or (s_node.skill_type if s_node and hasattr(s_node, "skill_type") else None),
                        }
                    )
                required_skills = req_list

            if occupation_label is None:
                if self.kg:
                    node = self.kg.get_node(occupation_uri)
                    if node and hasattr(node, "label") and node.label:
                        occupation_label = node.label
                    elif hasattr(self.kg, "occupations") and occupation_uri in self.kg.occupations:
                        occupation_label = self.kg.occupations[occupation_uri].label
                    else:
                        occupation_label = occupation_uri
                else:
                    occupation_label = occupation_uri

            return self.analyze_gap_exact(
                occupation_uri=occupation_uri,
                occupation_label=occupation_label,
                required_skills=required_skills,
                user_skill_uris=user_skill_uris,
            )

        return self.analyze_gap_semantic(
            occupation_uri=occupation_uri,
            user_skill_uris=user_skill_uris,
            occupation_label=occupation_label,
            required_skills=required_skills,
            top_k_candidates=top_k_candidates,
            similarity_threshold=similarity_threshold,
            candidate_pool=candidate_pool,
        )
