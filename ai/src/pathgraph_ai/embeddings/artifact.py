"""Precomputed Embedding Artifact Container and Serialization Utilities.

Stores and loads dense embedding matrices, aligned ESCO URI mappings, and metadata
using compressed NumPy archives (.npz).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingArtifact:
    """Container for precomputed embeddings, URI mappings, and metadata.

    Attributes:
        embeddings: 2D numpy array of shape (N, D).
        uris: List of N stable ESCO URIs matching the rows of embeddings.
        metadata: Dictionary containing model and generation metadata.
    """

    embeddings: Any  # np.ndarray
    uris: list[str]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self._uri_to_index: dict[str, int] | None = None
        self._uri_to_vector: dict[str, Any] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the shape (N, D) of the embedding matrix."""
        return self.embeddings.shape

    @property
    def count(self) -> int:
        """Returns the number of entity embeddings in the artifact."""
        return len(self.uris)

    @property
    def dimension(self) -> int:
        """Returns the vector dimension of the embeddings."""
        return self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 0

    @property
    def uri_to_index(self) -> dict[str, int]:
        """Returns a cached dictionary mapping each ESCO URI to its row index."""
        if self._uri_to_index is None:
            self._uri_to_index = {uri: idx for idx, uri in enumerate(self.uris)}
        return self._uri_to_index

    @property
    def uri_to_vector(self) -> dict[str, Any]:
        """Returns a cached dictionary mapping each ESCO URI to its 1D vector."""
        if self._uri_to_vector is None:
            self._uri_to_vector = {
                uri: self.embeddings[idx] for idx, uri in enumerate(self.uris)
            }
        return self._uri_to_vector

    def get_embedding(self, uri: str) -> Any | None:
        """Retrieves the embedding vector for a given ESCO URI.

        Args:
            uri: The target ESCO concept URI.

        Returns:
            1D numpy array if the URI is present, or None if not found.
        """
        idx = self.uri_to_index.get(uri)
        if idx is not None:
            return self.embeddings[idx]
        return None


def save_embeddings_artifact(
    output_path: str | Path,
    embeddings: Any,
    uris: Sequence[str],
    metadata: dict[str, Any],
    save_metadata_json: bool = True,
) -> Path:
    """Saves embedding matrix, URI mapping, and metadata to a compressed .npz file.

    Args:
        output_path: Target path for the .npz artifact.
        embeddings: 2D numpy array of shape (N, D).
        uris: Sequence of N unique ESCO concept URIs.
        metadata: Metadata dictionary to serialize alongside the vectors.
        save_metadata_json: If True, also writes a companion .json metadata file.

    Returns:
        Path to the saved .npz file.

    Raises:
        ValueError: If embeddings is invalid, dimensions mismatch, URIs are empty,
                    or row-to-URI count does not align.
        TypeError: If numpy is unavailable or arguments are of incorrect type.
    """
    if np is None:
        raise ImportError("NumPy is required to save embedding artifacts.")

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be a 2D matrix (N, D), got shape {embeddings.shape}."
        )

    num_rows, dim = embeddings.shape
    if num_rows == 0 or dim == 0:
        raise ValueError(
            f"Embeddings matrix cannot be empty, got shape {embeddings.shape}."
        )

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings matrix contains non-finite values (NaN or Inf).")

    if not isinstance(uris, (list, tuple)):
        uris = list(uris)

    if len(uris) != num_rows:
        raise ValueError(
            f"Row count mismatch: {num_rows} embedding rows but {len(uris)} URIs."
        )

    if len(set(uris)) != len(uris):
        raise ValueError("URI list contains duplicate entries; mappings must be unique.")

    out_file = Path(output_path).resolve()
    if out_file.suffix != ".npz":
        out_file = out_file.with_suffix(".npz")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Serialize metadata to JSON string
    meta_json_str = json.dumps(metadata, indent=2)

    # Save compressed .npz with embeddings, uris, and metadata
    np.savez_compressed(
        out_file,
        embeddings=embeddings.astype(np.float32, copy=False),
        uris=np.array(uris, dtype=str),
        metadata=np.array(meta_json_str),
    )
    logger.info("Saved compressed embedding artifact (%dx%d) to %s", num_rows, dim, out_file)

    # Optionally save companion .json file for inspection
    if save_metadata_json:
        companion_json = out_file.with_name(f"{out_file.stem}_metadata.json")
        with open(companion_json, mode="w", encoding="utf-8") as f:
            f.write(meta_json_str)
        logger.info("Saved companion metadata to %s", companion_json)

    return out_file


def load_embeddings_artifact(artifact_path: str | Path) -> EmbeddingArtifact:
    """Loads a precomputed embedding artifact from a .npz file.

    Args:
        artifact_path: Path to the .npz artifact.

    Returns:
        EmbeddingArtifact containing embeddings, URI list, and metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required arrays are missing or data is corrupted.
    """
    if np is None:
        raise ImportError("NumPy is required to load embedding artifacts.")

    target = Path(artifact_path).resolve()
    if target.suffix != ".npz":
        target = target.with_suffix(".npz")

    if not target.exists():
        raise FileNotFoundError(f"Embedding artifact not found at {target}")

    with np.load(target, allow_pickle=False) as data:
        if "embeddings" not in data:
            raise ValueError(f"Corrupted artifact '{target}': missing 'embeddings' array.")
        if "uris" not in data:
            raise ValueError(f"Corrupted artifact '{target}': missing 'uris' array.")

        embeddings = data["embeddings"].astype(np.float32, copy=False)
        uris = [str(u) for u in data["uris"]]

        metadata: dict[str, Any] = {}
        if "metadata" in data:
            meta_raw = str(data["metadata"])
            try:
                metadata = json.loads(meta_raw)
            except json.JSONDecodeError:
                metadata = {"raw": meta_raw}

    if embeddings.ndim != 2:
        raise ValueError(
            f"Corrupted artifact '{target}': embeddings array has invalid ndim {embeddings.ndim}."
        )

    if embeddings.shape[0] != len(uris):
        raise ValueError(
            f"Corrupted artifact '{target}': embeddings rows ({embeddings.shape[0]}) "
            f"!= URIs length ({len(uris)})."
        )

    return EmbeddingArtifact(embeddings=embeddings, uris=uris, metadata=metadata)
