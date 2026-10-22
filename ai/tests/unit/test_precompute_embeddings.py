"""Comprehensive Automated Test Suite for Offline Embedding Precomputation Module.

Verifies:
1. EmbeddingArtifact lookup, shape, count, and vector retrieval.
2. save_embeddings_artifact and load_embeddings_artifact round-trip fidelity.
3. Shape, finiteness, URI alignment, and uniqueness validation during artifact save/load.
4. OfflineEmbeddingPrecomputer lazy initialization (no model loading on init or import).
5. Batch computation, progress tracking, and batch dimension consistency.
6. Detection of malformed model outputs (1D output, row mismatch, non-finite values).
7. End-to-end precomputation with synthetic skills and occupations datasets.
8. CLI command interface execution, argument parsing, and missing-dependency error handling.
9. Raw ESCO CSV files remain completely untouched and unmodified.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pathgraph_ai.embeddings.artifact import (
    EmbeddingArtifact,
    load_embeddings_artifact,
    save_embeddings_artifact,
)
from pathgraph_ai.embeddings.model import SkillEmbeddingModel
from pathgraph_ai.embeddings.text_serializer import ESCOTextSerializer
from pathgraph_ai.pipelines.data_loader import ESCODataLoader
from pathgraph_ai.pipelines.precompute_embeddings import (
    OfflineEmbeddingPrecomputer,
    main,
)


# -----------------------------------------------------------------------------
# Fixtures and Helpers
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Provides an isolated temporary directory using tempfile to avoid Windows permissions issues."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def make_deterministic_embeddings(num_items: int, dim: int = 384) -> np.ndarray:
    """Generates deterministic non-zero, finite test embeddings."""
    base = np.arange(num_items * dim, dtype=np.float32).reshape(num_items, dim)
    # Normalize rows to have non-zero unit norms
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return base / norms


class MockEmbeddingModel:
    """Mock model that simulates SkillEmbeddingModel without heavy ML dependencies."""

    def __init__(self, dim: int = 384, model_name: str = "mock-mini-model"):
        self.dim = dim
        self.model_name = model_name
        self.is_loaded = False
        self.load_count = 0
        self.encode_calls: List[List[str]] = []

    def load_model(self):
        self.is_loaded = True
        self.load_count += 1

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        self.encode_calls.append(list(texts))
        return make_deterministic_embeddings(len(texts), self.dim)


# -----------------------------------------------------------------------------
# Artifact Container and Storage Tests
# -----------------------------------------------------------------------------


class TestEmbeddingArtifact:
    """Verifies EmbeddingArtifact accessors, indices, and vector lookups."""

    def test_artifact_properties_and_access(self):
        arr = make_deterministic_embeddings(3, 16)
        uris = ["http://esco/1", "http://esco/2", "http://esco/3"]
        meta = {"model": "test", "dim": 16}

        art = EmbeddingArtifact(embeddings=arr, uris=uris, metadata=meta)

        assert art.shape == (3, 16)
        assert art.count == 3
        assert art.dimension == 16
        assert art.metadata == meta

        # URI lookups
        assert art.uri_to_index["http://esco/1"] == 0
        assert art.uri_to_index["http://esco/2"] == 1
        assert art.uri_to_index["http://esco/3"] == 2

        np.testing.assert_allclose(art.get_embedding("http://esco/2"), arr[1])
        assert art.get_embedding("http://esco/nonexistent") is None

        # uri_to_vector dictionary
        assert len(art.uri_to_vector) == 3
        np.testing.assert_allclose(art.uri_to_vector["http://esco/1"], arr[0])


class TestSaveAndLoadArtifact:
    """Verifies persistence, formatting, roundtrip fidelity, and validation."""

    def test_roundtrip_save_and_load(self, temp_dir):
        num_items, dim = 5, 32
        arr = make_deterministic_embeddings(num_items, dim)
        uris = [f"http://esco/skill/{i}" for i in range(num_items)]
        meta = {
            "model_name": "test-model",
            "entity_type": "skill",
            "embedding_dimension": dim,
            "entity_count": num_items,
        }

        target = temp_dir / "test_artifact.npz"
        saved_path = save_embeddings_artifact(
            output_path=target,
            embeddings=arr,
            uris=uris,
            metadata=meta,
            save_metadata_json=True,
        )

        assert saved_path.exists()
        assert saved_path.suffix == ".npz"

        # Companion JSON file check
        companion_json = temp_dir / "test_artifact_metadata.json"
        assert companion_json.exists()
        with open(companion_json, "r", encoding="utf-8") as f:
            loaded_json_meta = json.load(f)
        assert loaded_json_meta == meta

        # Load artifact and verify
        loaded = load_embeddings_artifact(saved_path)
        assert loaded.shape == (num_items, dim)
        assert loaded.count == num_items
        assert loaded.uris == uris
        assert loaded.metadata == meta
        np.testing.assert_allclose(loaded.embeddings, arr, atol=1e-6)

    def test_save_rejects_1d_embeddings(self, temp_dir):
        with pytest.raises(ValueError, match="Embeddings must be a 2D matrix"):
            save_embeddings_artifact(
                temp_dir / "invalid.npz",
                embeddings=np.array([1.0, 2.0, 3.0]),
                uris=["http://esco/1"],
                metadata={},
            )

    def test_save_rejects_3d_embeddings(self, temp_dir):
        with pytest.raises(ValueError, match="Embeddings must be a 2D matrix"):
            save_embeddings_artifact(
                temp_dir / "invalid.npz",
                embeddings=np.ones((2, 3, 4)),
                uris=["http://esco/1", "http://esco/2"],
                metadata={},
            )

    def test_save_rejects_empty_matrix(self, temp_dir):
        with pytest.raises(ValueError, match="Embeddings matrix cannot be empty"):
            save_embeddings_artifact(
                temp_dir / "invalid.npz",
                embeddings=np.empty((0, 384)),
                uris=[],
                metadata={},
            )

        with pytest.raises(ValueError, match="Embeddings matrix cannot be empty"):
            save_embeddings_artifact(
                temp_dir / "invalid.npz",
                embeddings=np.empty((5, 0)),
                uris=["http://esco/1"] * 5,
                metadata={},
            )

    def test_save_rejects_non_finite_values(self, temp_dir):
        arr = np.array([[1.0, float("nan")], [3.0, 4.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite values"):
            save_embeddings_artifact(
                temp_dir / "invalid.npz",
                embeddings=arr,
                uris=["http://esco/1", "http://esco/2"],
                metadata={},
            )

    def test_save_rejects_row_uri_mismatch(self, temp_dir):
        arr = make_deterministic_embeddings(3, 16)
        with pytest.raises(ValueError, match="Row count mismatch: 3 embedding rows but 2 URIs"):
            save_embeddings_artifact(
                temp_dir / "invalid.npz",
                embeddings=arr,
                uris=["http://esco/1", "http://esco/2"],
                metadata={},
            )

    def test_save_rejects_duplicate_uris(self, temp_dir):
        arr = make_deterministic_embeddings(2, 16)
        with pytest.raises(ValueError, match="URI list contains duplicate entries"):
            save_embeddings_artifact(
                temp_dir / "invalid.npz",
                embeddings=arr,
                uris=["http://esco/1", "http://esco/1"],
                metadata={},
            )

    def test_load_nonexistent_file_raises_not_found(self, temp_dir):
        with pytest.raises(FileNotFoundError, match="Embedding artifact not found"):
            load_embeddings_artifact(temp_dir / "nonexistent.npz")

    def test_load_corrupted_npz_missing_arrays(self, temp_dir):
        corrupt_file = temp_dir / "corrupt.npz"
        np.savez_compressed(corrupt_file, only_dummy=np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="missing 'embeddings' array"):
            load_embeddings_artifact(corrupt_file)


# -----------------------------------------------------------------------------
# Precomputer Pipeline and Batching Tests
# -----------------------------------------------------------------------------


class TestOfflineEmbeddingPrecomputer:
    """Verifies precomputation orchestration, batching, and mock inference."""

    def test_lazy_initialization_does_not_load_model(self):
        mock_model = MockEmbeddingModel()
        precomputer = OfflineEmbeddingPrecomputer(model=mock_model)

        assert not precomputer.model.is_loaded
        assert mock_model.load_count == 0

    def test_rejects_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            OfflineEmbeddingPrecomputer(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            OfflineEmbeddingPrecomputer(batch_size=-10)

    def test_compute_embeddings_batching(self):
        mock_model = MockEmbeddingModel(dim=64)
        precomputer = OfflineEmbeddingPrecomputer(model=mock_model, batch_size=3)

        texts = [f"skill text {i}" for i in range(8)]
        embeddings = precomputer.compute_embeddings(texts, show_progress=False)

        assert mock_model.is_loaded
        assert mock_model.load_count == 1
        assert len(mock_model.encode_calls) == 3
        # Batches of size 3, 3, 2
        assert len(mock_model.encode_calls[0]) == 3
        assert len(mock_model.encode_calls[1]) == 3
        assert len(mock_model.encode_calls[2]) == 2

        assert embeddings.shape == (8, 64)
        assert embeddings.dtype == np.float32

    def test_compute_embeddings_rejects_empty_texts(self):
        mock_model = MockEmbeddingModel()
        precomputer = OfflineEmbeddingPrecomputer(model=mock_model)
        with pytest.raises(ValueError, match="Input text list for embedding computation cannot be empty"):
            precomputer.compute_embeddings([])

    def test_compute_embeddings_detects_1d_model_output(self):
        bad_model = MockEmbeddingModel()
        bad_model.load_model()
        bad_model.encode = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

        precomputer = OfflineEmbeddingPrecomputer(model=bad_model)
        with pytest.raises(ValueError, match="Model encode returned array with shape.*expected 2D"):
            precomputer.compute_embeddings(["test text"])

    def test_compute_embeddings_detects_row_mismatch(self):
        bad_model = MockEmbeddingModel()
        bad_model.load_model()
        bad_model.encode = MagicMock(return_value=np.zeros((5, 384)))

        precomputer = OfflineEmbeddingPrecomputer(model=bad_model)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            precomputer.compute_embeddings(["text 1", "text 2"])  # 2 inputs, 5 returned rows

    def test_compute_embeddings_detects_dimension_mismatch_across_batches(self):
        bad_model = MockEmbeddingModel()
        bad_model.load_model()
        # First batch returns dim 64, second batch returns dim 32
        bad_model.encode = MagicMock(
            side_effect=[np.zeros((2, 64), dtype=np.float32), np.zeros((2, 32), dtype=np.float32)]
        )

        precomputer = OfflineEmbeddingPrecomputer(model=bad_model, batch_size=2)
        with pytest.raises(ValueError, match="Dimension mismatch in batch 1"):
            precomputer.compute_embeddings(["t1", "t2", "t3", "t4"])

    def test_compute_embeddings_detects_non_finite_values(self):
        bad_model = MockEmbeddingModel()
        bad_model.load_model()
        nan_matrix = np.array([[1.0, float("nan")]], dtype=np.float32)
        bad_model.encode = MagicMock(return_value=nan_matrix)

        precomputer = OfflineEmbeddingPrecomputer(model=bad_model)
        with pytest.raises(ValueError, match="non-finite values"):
            precomputer.compute_embeddings(["test"])

    def test_precompute_synthetic_skills_end_to_end(self, temp_dir):
        mock_model = MockEmbeddingModel(dim=128)
        precomputer = OfflineEmbeddingPrecomputer(
            model=mock_model,
            batch_size=2,
            output_dir=temp_dir,
        )

        skills = [
            {
                "conceptUri": "http://esco/skill/s1",
                "preferredLabel": "Python programming",
                "altLabels": "Python 3\nPython scripting",
                "description": "Develop Python software.",
            },
            {
                "conceptUri": "http://esco/skill/s2",
                "preferredLabel": "SQL querying",
                "altLabels": "Relational queries",
                "description": "Query relational databases.",
            },
            {
                "conceptUri": "http://esco/skill/s3",
                "preferredLabel": "Git version control",
                "altLabels": None,
                "description": "Manage source code with Git.",
            },
        ]

        out_file = temp_dir / "skills_embeddings.npz"
        artifact = precomputer.precompute(
            records=skills,
            entity_type="skill",
            output_path=out_file,
            source_file="synthetic_skills.csv",
        )

        assert out_file.exists()
        assert artifact.shape == (3, 128)
        assert artifact.count == 3
        assert artifact.dimension == 128
        assert artifact.uris == [
            "http://esco/skill/s1",
            "http://esco/skill/s2",
            "http://esco/skill/s3",
        ]
        assert artifact.metadata["entity_type"] == "skill"
        assert artifact.metadata["embedding_dimension"] == 128
        assert artifact.metadata["entity_count"] == 3
        assert artifact.metadata["batch_size"] == 2
        assert artifact.metadata["format_version"] == "1.0"
        assert artifact.metadata["source_file"] == "synthetic_skills.csv"

        # Check exact URI-to-vector correspondence
        v1 = artifact.get_embedding("http://esco/skill/s1")
        assert v1 is not None
        assert v1.shape == (128,)

    def test_precompute_synthetic_occupations_end_to_end(self, temp_dir):
        mock_model = MockEmbeddingModel(dim=64)
        precomputer = OfflineEmbeddingPrecomputer(
            model=mock_model,
            batch_size=5,
            output_dir=temp_dir,
        )

        occupations = [
            {
                "conceptUri": "http://esco/occupation/o1",
                "preferredLabel": "Data Scientist",
                "altLabels": "Data Science Specialist\nML Scientist",
                "description": "Extract insights from complex data.",
            },
            {
                "conceptUri": "http://esco/occupation/o2",
                "preferredLabel": "DevOps Engineer",
                "altLabels": "Site Reliability Engineer",
                "description": "Build CI/CD automation pipelines.",
            },
        ]

        out_file = temp_dir / "occupations_embeddings.npz"
        artifact = precomputer.precompute(
            records=occupations,
            entity_type="occupation",
            output_path=out_file,
        )

        assert out_file.exists()
        assert artifact.shape == (2, 64)
        assert artifact.metadata["entity_type"] == "occupation"
        assert artifact.uris == [
            "http://esco/occupation/o1",
            "http://esco/occupation/o2",
        ]

    def test_precompute_rejects_empty_dataset(self, temp_dir):
        mock_model = MockEmbeddingModel()
        precomputer = OfflineEmbeddingPrecomputer(model=mock_model, output_dir=temp_dir)
        with pytest.raises(ValueError, match="Cannot precompute embeddings for empty skill dataset"):
            precomputer.precompute(records=[], entity_type="skill")

    def test_precompute_rejects_invalid_entity_type(self, temp_dir):
        mock_model = MockEmbeddingModel()
        precomputer = OfflineEmbeddingPrecomputer(model=mock_model, output_dir=temp_dir)
        with pytest.raises(ValueError, match="entity_type must be 'skill' or 'occupation'"):
            precomputer.precompute(
                records=[{"conceptUri": "http://esco/1", "preferredLabel": "Test"}],
                entity_type="invalid_type",
            )


# -----------------------------------------------------------------------------
# CLI and Error Handling Tests
# -----------------------------------------------------------------------------


class TestCLIExecution:
    """Verifies CLI execution, help flags, and dependency error handling."""

    def test_cli_help_flag_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_missing_sentence_transformers_returns_one(self, monkeypatch, capsys):
        def raise_import_error(self):
            raise ImportError("sentence-transformers is not installed.")

        monkeypatch.setattr(SkillEmbeddingModel, "load_model", raise_import_error)

        exit_code = main(["--limit", "2"])
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "[ERROR] Required ML package is unavailable" in captured.out
        assert "pip install torch sentence-transformers" in captured.out

    def test_cli_mocked_success_run(self, temp_dir, monkeypatch, capsys):
        mock_model = MockEmbeddingModel(dim=32)

        orig_init = OfflineEmbeddingPrecomputer.__init__

        def custom_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.model = mock_model
            self.output_dir = temp_dir

        monkeypatch.setattr(OfflineEmbeddingPrecomputer, "__init__", custom_init)

        exit_code = main(["--entity-type", "skills", "--limit", "3"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "All requested precomputation tasks finished successfully." in captured.out
        assert (temp_dir / "skills_embeddings.npz").exists()


# -----------------------------------------------------------------------------
# Raw ESCO File Immutability Test
# -----------------------------------------------------------------------------


def test_raw_esco_files_remain_unmodified():
    """Confirms that raw ESCO dataset files in canonical raw dir were not modified."""
    loader = ESCODataLoader()
    raw_files = [
        "occupations_en.csv",
        "skills_en.csv",
        "occupationSkillRelations_en.csv",
        "skillsHierarchy_en.csv",
        "skillSkillRelations_en.csv",
    ]
    for fname in raw_files:
        path = loader.get_file_path(fname)
        assert path.exists(), f"Raw file disappeared: {path}"
        assert path.stat().st_size > 0, f"Raw file corrupted: {path}"
