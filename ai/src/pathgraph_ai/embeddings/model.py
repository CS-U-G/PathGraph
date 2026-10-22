"""Semantic Representation and Similarity Module for PathGraph.

Provides interfaces for encoding occupations and skills into dense vector spaces
using Sentence Transformers and calculating cosine similarity between embeddings.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# Lazy optional imports for heavy ML packages
try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def pure_cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Computes cosine similarity between two 1D float vectors in pure Python.

    Args:
        vec_a: First 1D sequence of float values.
        vec_b: Second 1D sequence of float values.

    Returns:
        Cosine similarity clamped to [-1.0, 1.0], or 0.0 if either vector has zero norm.

    Raises:
        ValueError: If either vector is empty, dimensions mismatch, or contains NaN/Inf.
        TypeError: If inputs are not sequences or contain non-numeric values.
    """
    if vec_a is None or vec_b is None:
        raise ValueError("Vectors must not be None.")

    try:
        len_a = len(vec_a)
        len_b = len(vec_b)
    except TypeError as e:
        raise TypeError("Vectors must be sequences with a defined length.") from e

    if len_a == 0 or len_b == 0:
        raise ValueError("Vectors must not be empty.")

    if len_a != len_b:
        raise ValueError(
            f"Vectors must have the same dimension: {len_a} != {len_b}."
        )

    dot_product = 0.0
    norm_sq_a = 0.0
    norm_sq_b = 0.0

    for idx, (a, b) in enumerate(zip(vec_a, vec_b)):
        if isinstance(a, bool) or not isinstance(a, (int, float)):
            raise TypeError(
                f"Vector A element at index {idx} must be numeric, got {type(a).__name__}."
            )
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise TypeError(
                f"Vector B element at index {idx} must be numeric, got {type(b).__name__}."
            )

        fa = float(a)
        fb = float(b)

        if math.isnan(fa) or math.isinf(fa):
            raise ValueError(
                f"Vector A contains non-finite value ({fa}) at index {idx}."
            )
        if math.isnan(fb) or math.isinf(fb):
            raise ValueError(
                f"Vector B contains non-finite value ({fb}) at index {idx}."
            )

        dot_product += fa * fb
        norm_sq_a += fa * fa
        norm_sq_b += fb * fb

    norm_a = math.sqrt(norm_sq_a)
    norm_b = math.sqrt(norm_sq_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = dot_product / (norm_a * norm_b)
    # Clamp to [-1.0, 1.0] to guard against floating-point inaccuracies
    return max(-1.0, min(1.0, sim))


def _validate_vector(vec: Any, name: str = "Embedding") -> Any:
    """Validates and normalizes an input vector into a 1D sequence or ndarray.

    Args:
        vec: Vector representation (ndarray, list, or tuple).
        name: Name of the vector parameter for informative error messages.

    Returns:
        1D numpy array of floats (if numpy available) or 1D list of floats.

    Raises:
        ValueError: If vector is None, empty, multidimensional, or contains non-finite values.
        TypeError: If vector is not a sequence/array or contains non-numeric values.
    """
    if vec is None:
        raise ValueError(f"{name} cannot be None.")

    if np is not None and isinstance(vec, np.ndarray):
        if vec.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if vec.ndim == 1:
            arr = vec
        elif vec.ndim == 2 and (vec.shape[0] == 1 or vec.shape[1] == 1):
            arr = vec.reshape(-1)
        else:
            raise ValueError(
                f"{name} must be a 1D vector, got array with shape {vec.shape}."
            )

        if not np.issubdtype(arr.dtype, np.number) or arr.dtype == bool:
            raise TypeError(
                f"{name} elements must be numeric, got dtype {arr.dtype}."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{name} contains non-finite values (NaN or Inf)."
            )

        return arr.astype(float, copy=False)

    if isinstance(vec, (list, tuple)):
        if len(vec) == 0:
            raise ValueError(f"{name} must not be empty.")

        # Check for nested structures and elements
        for idx, item in enumerate(vec):
            if isinstance(item, (list, tuple)) or (
                np is not None and isinstance(item, np.ndarray)
            ):
                raise ValueError(
                    f"{name} must be a 1D vector, got nested sequence at index {idx}."
                )
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(
                    f"{name} element at index {idx} must be numeric, got {type(item).__name__}."
                )
            f_val = float(item)
            if math.isnan(f_val) or math.isinf(f_val):
                raise ValueError(
                    f"{name} contains non-finite value ({f_val}) at index {idx}."
                )

        if np is not None:
            return np.asarray(vec, dtype=float)
        return [float(x) for x in vec]

    raise TypeError(
        f"{name} must be a sequence or 1D ndarray, got {type(vec).__name__}."
    )


class SkillEmbeddingModel:
    """Wrapper around SentenceTransformer models for generating and querying embeddings."""

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    _validate_vector = staticmethod(_validate_vector)

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self._model = None

    @property
    def is_loaded(self) -> bool:
        """Checks if the sentence transformer model is currently loaded in memory."""
        return self._model is not None

    def load_model(self) -> None:
        """Loads the pretrained Sentence Transformer model."""
        global SentenceTransformer
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer as _ST
                SentenceTransformer = _ST
            except ImportError:
                SentenceTransformer = None

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with 'pip install sentence-transformers'."
            )
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: str | list[str]) -> Any:
        """Encodes text or list of texts into dense embedding vectors.

        Args:
            texts: A single text string or a list of text strings.

        Returns:
            Dense embedding vectors produced by the SentenceTransformer model.

        Raises:
            ValueError: If input is None, empty string, whitespace-only, empty list,
                        or contains invalid/empty string entries.
            TypeError: If input or list items are of incorrect type.
            RuntimeError: If the model has not been loaded via load_model().
        """
        if texts is None:
            raise ValueError("Input text cannot be None.")

        if isinstance(texts, str):
            if not texts.strip():
                raise ValueError("Input text cannot be empty or whitespace-only.")
        elif isinstance(texts, (list, tuple)):
            if len(texts) == 0:
                raise ValueError("Input text collection cannot be empty.")
            for idx, item in enumerate(texts):
                if item is None:
                    raise ValueError(
                        f"Invalid text entry at index {idx}: text cannot be None."
                    )
                if not isinstance(item, str):
                    raise TypeError(
                        f"Invalid text entry at index {idx}: expected str, got {type(item).__name__}."
                    )
                if not item.strip():
                    raise ValueError(
                        f"Invalid text entry at index {idx}: text cannot be empty or whitespace-only."
                    )
        else:
            raise TypeError(
                f"Expected str or List[str], got {type(texts).__name__}."
            )

        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Call load_model() before encoding."
            )

        return self._model.encode(texts)

    def compute_similarity(self, embedding_a: Any, embedding_b: Any) -> float:
        """Computes cosine similarity between two embedding vectors.

        Args:
            embedding_a: First embedding vector (1D ndarray or sequence).
            embedding_b: Second embedding vector (1D ndarray or sequence).

        Returns:
            Cosine similarity clamped to [-1.0, 1.0], or 0.0 for zero-norm vectors.

        Raises:
            ValueError: If vectors are empty, shapes/dimensions mismatch, or values are non-finite.
            TypeError: If vector types or element types are invalid.
        """
        vec_a = self._validate_vector(embedding_a, name="embedding_a")
        vec_b = self._validate_vector(embedding_b, name="embedding_b")

        len_a = len(vec_a)
        len_b = len(vec_b)
        if len_a != len_b:
            raise ValueError(
                f"Embedding dimensions do not match: {len_a} != {len_b}."
            )

        if np is not None and isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
            norm_a = float(np.linalg.norm(vec_a))
            norm_b = float(np.linalg.norm(vec_b))
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
            return max(-1.0, min(1.0, sim))

        return pure_cosine_similarity(vec_a, vec_b)

    def find_top_k(
        self,
        query_embedding: Any,
        candidate_embeddings: dict[str, Any],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Finds top-K most similar candidate IDs given a query embedding.

        Args:
            query_embedding: Query embedding vector (1D ndarray or sequence).
            candidate_embeddings: Dict mapping candidate identifier to embedding vector.
            top_k: Number of top matches to return. If non-positive, returns an empty list.

        Returns:
            List of (candidate_id, similarity) tuples sorted in descending order of similarity.

        Raises:
            TypeError: If candidate_embeddings is not a dict, or top_k is not an int.
            ValueError: If query_embedding or any candidate vector is empty, invalid,
                        or has mismatched dimensions.
        """
        if not isinstance(candidate_embeddings, dict):
            raise TypeError(
                f"candidate_embeddings must be a dict, got {type(candidate_embeddings).__name__}."
            )

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")

        if top_k <= 0 or len(candidate_embeddings) == 0:
            return []

        query_vec = self._validate_vector(query_embedding, name="query_embedding")
        query_dim = len(query_vec)

        keys = list(candidate_embeddings.keys())

        if np is not None and isinstance(query_vec, np.ndarray):
            cand_list = []
            for key in keys:
                cand_val = candidate_embeddings[key]
                cand_vec = self._validate_vector(cand_val, name=f"candidate '{key}'")
                if len(cand_vec) != query_dim:
                    raise ValueError(
                        f"Candidate '{key}' dimension ({len(cand_vec)}) does not match "
                        f"query dimension ({query_dim})."
                    )
                cand_list.append(cand_vec)

            cand_matrix = np.vstack(cand_list)  # Shape: (N, D)
            query_norm = float(np.linalg.norm(query_vec))

            if query_norm == 0.0:
                sims = np.zeros(len(keys), dtype=float)
            else:
                dots = cand_matrix.dot(query_vec)  # Shape: (N,)
                cand_norms = np.linalg.norm(cand_matrix, axis=1)  # Shape: (N,)
                denom = query_norm * cand_norms
                sims = np.zeros_like(dots, dtype=float)
                nonzero_mask = denom > 0.0
                sims[nonzero_mask] = dots[nonzero_mask] / denom[nonzero_mask]
                sims = np.clip(sims, -1.0, 1.0)

            scores = [(keys[i], float(sims[i])) for i in range(len(keys))]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        # Pure Python fallback
        scores = []
        for key in keys:
            cand_val = candidate_embeddings[key]
            sim = self.compute_similarity(query_vec, cand_val)
            scores.append((key, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
