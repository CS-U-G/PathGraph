"""Offline Embedding Precomputation Module for PathGraph.

Precomputes dense vector embeddings for ESCO skills and occupations using
SkillEmbeddingModel and ESCOTextSerializer. Saves embeddings, stable ESCO URI
mappings, and metadata into compressed NumPy (.npz) artifacts with exact row-to-vector
alignment.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pathgraph_ai.embeddings.artifact import (
    EmbeddingArtifact,
    load_embeddings_artifact,
    save_embeddings_artifact,
)
from pathgraph_ai.embeddings.model import SkillEmbeddingModel
from pathgraph_ai.embeddings.text_serializer import ESCOTextSerializer
from pathgraph_ai.pipelines.data_loader import ESCODataLoader
from pathgraph_ai.settings import settings

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class OfflineEmbeddingPrecomputer:
    """Orchestrates batch text serialization and vector precomputation for ESCO."""

    def __init__(
        self,
        model: SkillEmbeddingModel | None = None,
        model_name: str | None = None,
        serializer: ESCOTextSerializer | None = None,
        batch_size: int = 128,
        data_loader: ESCODataLoader | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Initializes the precomputer without loading model weights into memory.

        Args:
            model: Optional pre-configured SkillEmbeddingModel instance.
            model_name: Model name if creating a new SkillEmbeddingModel.
            serializer: Optional ESCOTextSerializer instance.
            batch_size: Number of entity texts per model inference batch.
            data_loader: Optional ESCODataLoader instance.
            output_dir: Directory where precomputed artifacts are stored.
                        Defaults to settings.processed_data_dir.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        self.model = model or SkillEmbeddingModel(model_name=model_name or settings.model_name)
        self.serializer = serializer or ESCOTextSerializer()
        self.batch_size = batch_size
        self.data_loader = data_loader or ESCODataLoader(raw_data_dir=settings.raw_data_dir)

        if output_dir is None:
            self.output_dir = settings.processed_data_dir
        else:
            self.output_dir = Path(output_dir)

    def compute_embeddings(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> Any:
        """Encodes texts in configurable batches using the underlying SkillEmbeddingModel.

        Ensures the model is loaded only upon explicit invocation. Validates batch
        shapes and detects non-finite vector values.

        Args:
            texts: List of non-empty text strings to encode.
            show_progress: Whether to log batch encoding progress.

        Returns:
            2D numpy array of shape (len(texts), D) with float32 values.
        """
        if np is None:
            raise ImportError("NumPy is required to compute and store embedding matrices.")

        if not texts:
            raise ValueError("Input text list for embedding computation cannot be empty.")

        if not self.model.is_loaded:
            logger.info("Loading sentence transformer model '%s'...", self.model.model_name)
            self.model.load_model()

        embeddings_list: list[Any] = []
        total_texts = len(texts)
        total_batches = (total_texts + self.batch_size - 1) // self.batch_size
        expected_dim: int | None = None

        for b_idx in range(total_batches):
            start = b_idx * self.batch_size
            end = min(start + self.batch_size, total_texts)
            batch = texts[start:end]

            if show_progress and total_batches > 1:
                logger.info(
                    "Encoding batch %d/%d (items %d-%d)...",
                    b_idx + 1,
                    total_batches,
                    start,
                    end,
                )

            batch_embs = self.model.encode(batch)
            if not isinstance(batch_embs, np.ndarray):
                batch_embs = np.asarray(batch_embs, dtype=np.float32)

            if batch_embs.ndim != 2:
                raise ValueError(
                    f"Model encode returned array with shape {batch_embs.shape}; expected 2D matrix."
                )

            if batch_embs.shape[0] != len(batch):
                raise ValueError(
                    f"Batch size mismatch: encoded {len(batch)} texts but received {batch_embs.shape[0]} rows."
                )

            if expected_dim is None:
                expected_dim = batch_embs.shape[1]
            elif batch_embs.shape[1] != expected_dim:
                raise ValueError(
                    f"Dimension mismatch in batch {b_idx}: expected {expected_dim}, got {batch_embs.shape[1]}."
                )

            if not np.all(np.isfinite(batch_embs)):
                raise ValueError(
                    f"Batch {b_idx} contains non-finite values (NaN or Inf)."
                )

            embeddings_list.append(batch_embs)

        full_matrix = np.vstack(embeddings_list).astype(np.float32, copy=False)
        return full_matrix

    def precompute(
        self,
        records: Any,
        entity_type: str,
        output_path: str | Path | None = None,
        source_file: str | None = None,
    ) -> EmbeddingArtifact:
        """Precomputes embeddings for an in-memory list of records and writes artifact to disk."""
        if not records:
            raise ValueError(f"Cannot precompute embeddings for empty {entity_type} dataset.")
        clean_etype = entity_type.strip().lower()
        if clean_etype not in ("skill", "occupation"):
            raise ValueError(f"entity_type must be 'skill' or 'occupation', got {entity_type!r}.")

        serialized_pairs = self.serializer.serialize_dataset(
            records=records,
            entity_type=clean_etype,
            strict_unique_keys=True,
        )
        uris = [p[0] for p in serialized_pairs]
        texts = [p[1] for p in serialized_pairs]

        emb_matrix = self.compute_embeddings(texts, show_progress=False)
        if emb_matrix.shape[0] != len(uris):
            raise ValueError(f"Alignment error: {emb_matrix.shape[0]} embeddings produced for {len(uris)} URIs.")

        # Unit-normalize vectors
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized_matrix = (emb_matrix / norms).astype(np.float32)

        metadata = {
            "model_name": self.model.model_name,
            "entity_type": clean_etype,
            "embedding_dimension": int(normalized_matrix.shape[1]),
            "entity_count": int(normalized_matrix.shape[0]),
            "batch_size": self.batch_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0",
            "source_file": source_file or f"{clean_etype}s_en.csv",
        }

        target_path = Path(output_path) if output_path else self.output_dir / f"{clean_etype}s_embeddings.npz"
        saved_path = save_embeddings_artifact(
            output_path=target_path,
            embeddings=normalized_matrix,
            uris=uris,
            metadata=metadata,
            save_metadata_json=True,
        )
        return load_embeddings_artifact(saved_path)

    def precompute_skills(
        self,
        records: Any = None,
        limit: int | None = None,
        output_path: str | Path | None = None,
    ) -> EmbeddingArtifact:
        if records is None:
            return self.precompute_entity_type("skills", limit=limit)
        return self.precompute(records=records, entity_type="skill", output_path=output_path)

    def precompute_occupations(
        self,
        records: Any = None,
        limit: int | None = None,
        output_path: str | Path | None = None,
    ) -> EmbeddingArtifact:
        if records is None:
            return self.precompute_entity_type("occupations", limit=limit)
        return self.precompute(records=records, entity_type="occupation", output_path=output_path)

    def precompute_entity_type(
        self,
        entity_type: str,
        limit: int | None = None,
        save_artifact: bool = True,
        filename_prefix: str | None = None,
    ) -> EmbeddingArtifact:
        """Precomputes and optionally saves embeddings for a single ESCO entity type.

        Args:
            entity_type: 'skills' (or 'skill') or 'occupations' (or 'occupation').
            limit: Optional maximum number of entity records to process.
            save_artifact: Whether to write the resulting matrix to .npz on disk.
            filename_prefix: Optional filename stem (defaults to entity_type).

        Returns:
            EmbeddingArtifact containing normalized embeddings, URIs, and metadata.
        """
        etype = entity_type.strip().lower()
        if "skill" in etype:
            logger.info("Loading skills dataset via ESCODataLoader...")
            records = self.data_loader.load_skills(limit=limit, deduplicate=True)
            default_prefix = "skills_embeddings"
            source_file = "skills_en.csv"
            kind = "skill"
        elif "occ" in etype:
            logger.info("Loading occupations dataset via ESCODataLoader...")
            records = self.data_loader.load_occupations(limit=limit, deduplicate=True)
            default_prefix = "occupations_embeddings"
            source_file = "occupations_en.csv"
            kind = "occupation"
        else:
            raise ValueError(
                f"Unknown entity_type '{entity_type}'. Expected 'skills' or 'occupations'."
            )

        if not records:
            raise ValueError(f"No records found for entity type '{entity_type}'.")

        logger.info("Serializing %d %s records for embedding...", len(records), kind)
        serialized_pairs = self.serializer.serialize_dataset(
            records=records, entity_type=kind, key_field="conceptUri", strict_unique_keys=True
        )

        uris = [pair[0] for pair in serialized_pairs]
        texts = [pair[1] for pair in serialized_pairs]

        logger.info("Computing dense embeddings for %d items (batch_size=%d)...", len(texts), self.batch_size)
        t0 = time.time()
        emb_matrix = self.compute_embeddings(texts, show_progress=True)
        elapsed = time.time() - t0
        logger.info(
            "Computed %dx%d embedding matrix in %.2fs (%.1f items/sec).",
            emb_matrix.shape[0],
            emb_matrix.shape[1],
            elapsed,
            len(texts) / max(elapsed, 0.001),
        )

        # Normalize unit vectors for fast dot product cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized_matrix = emb_matrix / norms

        metadata = {
            "model_name": self.model.model_name,
            "entity_type": kind,
            "embedding_dimension": int(normalized_matrix.shape[1]),
            "entity_count": int(normalized_matrix.shape[0]),
            "batch_size": self.batch_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0",
            "source_file": source_file,
            "limit": limit,
        }

        artifact = EmbeddingArtifact(
            embeddings=normalized_matrix,
            uris=uris,
            metadata=metadata,
        )

        if save_artifact:
            prefix = filename_prefix or default_prefix
            out_path = self.output_dir / f"{prefix}.npz"
            save_embeddings_artifact(
                output_path=out_path,
                embeddings=artifact.embeddings,
                uris=artifact.uris,
                metadata=artifact.metadata,
                save_metadata_json=True,
            )

        return artifact

    def precompute_all(
        self,
        limit: int | None = None,
        save_artifact: bool = True,
    ) -> dict[str, EmbeddingArtifact]:
        """Precomputes embeddings for both skills and occupations."""
        results: dict[str, EmbeddingArtifact] = {}
        logger.info("=== Precomputing Skills Embeddings ===")
        results["skills"] = self.precompute_entity_type(
            entity_type="skills", limit=limit, save_artifact=save_artifact
        )
        logger.info("=== Precomputing Occupations Embeddings ===")
        results["occupations"] = self.precompute_entity_type(
            entity_type="occupations", limit=limit, save_artifact=save_artifact
        )
        return results


def main(args: Any = None) -> int:
    """CLI entrypoint for batch embedding precomputation."""
    parser = argparse.ArgumentParser(
        description="PathGraph ESCO Embedding Precomputation CLI"
    )
    parser.add_argument(
        "--entity-type",
        choices=["skills", "occupations", "all"],
        default="all",
        help="Entity type to encode into vector embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on records for test/verification runs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for .npz artifacts. Defaults to data/processed/.",
    )

    parsed_args = parser.parse_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        precomputer = OfflineEmbeddingPrecomputer(
            model_name=parsed_args.model_name,
            batch_size=parsed_args.batch_size,
            output_dir=parsed_args.output_dir,
        )

        if parsed_args.entity_type == "all":
            precomputer.precompute_all(limit=parsed_args.limit)
        else:
            precomputer.precompute_entity_type(
                entity_type=parsed_args.entity_type, limit=parsed_args.limit
            )

        print("All requested precomputation tasks finished successfully.")
        return 0
    except ImportError as e:
        print(f"[ERROR] Required ML package is unavailable: {e}")
        print("Install required dependencies using: pip install torch sentence-transformers")
        return 1


if __name__ == "__main__":
    sys.exit(main())
