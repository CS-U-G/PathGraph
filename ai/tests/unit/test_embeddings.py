"""Comprehensive Automated Test Suite for PathGraph Embeddings Module.

Verifies:
1. Pure Python cosine similarity mathematical properties (identities, orthogonality,
   opposites, symmetry, scale-invariance, known analytical values).
2. Input validation for pure_cosine_similarity (empty vectors, dimension mismatch,
   non-finite values, type errors).
3. Clamping and zero-norm behavior for pure_cosine_similarity.
4. SkillEmbeddingModel lazy initialization and unloaded model behavior.
5. SkillEmbeddingModel input text validation for encode (None, empty strings, whitespace,
   empty lists, nested/invalid items, clear error messages).
6. SkillEmbeddingModel encode behavior with mocked SentenceTransformer.
7. SkillEmbeddingModel compute_similarity with numpy arrays, lists, mixed inputs,
   shape validation (1D, 2D single-vector squeeze, rejection of 2D matrices / 3D arrays),
   empty arrays, mismatched dimensions, non-finite values, and zero-norm vectors.
8. SkillEmbeddingModel find_top_k ranking order (strictly descending similarity).
9. SkillEmbeddingModel find_top_k edge cases (k > candidate count, k == candidate count,
   k < candidate count, empty candidates, non-positive k, invalid top_k type,
   invalid candidate collection type, mismatched dimensions, zero-norm candidates).
10. Consistency between vectorized numpy calculations and pure Python calculations.
"""

import math
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from pathgraph_ai.embeddings import model as embeddings
from pathgraph_ai.embeddings.model import (
    SkillEmbeddingModel,
    pure_cosine_similarity,
)

try:
    import numpy as np
except ImportError:
    np = None


# =============================================================================
# 1. Pure Cosine Similarity Mathematical Properties
# =============================================================================


class TestPureCosineSimilarityMath:
    """Verifies mathematical invariants and correctness of pure_cosine_similarity."""

    def test_identical_vectors_returns_one(self):
        """Identical non-zero vectors must yield a cosine similarity of exactly 1.0."""
        v1 = [1.0, 2.0, 3.0]
        assert pure_cosine_similarity(v1, v1) == pytest.approx(1.0, abs=1e-7)

        v2 = [-4.5, 0.2, -10.0, 7.3]
        assert pure_cosine_similarity(v2, v2) == pytest.approx(1.0, abs=1e-7)

    def test_orthogonal_vectors_returns_zero(self):
        """Orthogonal vectors must yield a cosine similarity of 0.0."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert pure_cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-7)

        v3 = [1.0, 0.0, 0.0]
        v4 = [0.0, 0.0, 5.0]
        assert pure_cosine_similarity(v3, v4) == pytest.approx(0.0, abs=1e-7)

    def test_opposite_vectors_returns_negative_one(self):
        """Vectors pointing in diametrically opposite directions must yield -1.0."""
        v1 = [1.0, 2.0, -3.0]
        v2 = [-1.0, -2.0, 3.0]
        assert pure_cosine_similarity(v1, v2) == pytest.approx(-1.0, abs=1e-7)

    def test_scale_invariance(self):
        """Cosine similarity must be invariant to positive scalar multiplication."""
        v1 = [1.0, -2.0, 4.0]
        v2 = [3.0, -6.0, 12.0]  # 3 * v1
        assert pure_cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-7)

        v3 = [-2.0, 4.0, -8.0]  # -2 * v1
        assert pure_cosine_similarity(v1, v3) == pytest.approx(-1.0, abs=1e-7)

    def test_symmetry(self):
        """sim(a, b) must strictly equal sim(b, a)."""
        v1 = [1.2, 3.4, -0.5, 6.7]
        v2 = [-0.8, 2.1, 4.0, -1.1]
        assert pure_cosine_similarity(v1, v2) == pytest.approx(
            pure_cosine_similarity(v2, v1), abs=1e-9
        )

    def test_known_analytical_value(self):
        """Verifies calculation against manual algebraic derivation."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        expected = 32.0 / math.sqrt(1078.0)
        assert pure_cosine_similarity(v1, v2) == pytest.approx(expected, abs=1e-7)

    def test_zero_norm_vectors_safe_handling(self):
        """Vectors with zero norm must return 0.0 safely without division by zero."""
        assert pure_cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert pure_cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0
        assert pure_cosine_similarity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == 0.0

    def test_clamping_boundary(self):
        """Results must never exceed [-1.0, 1.0] even under floating point inaccuracies."""
        v1 = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
        res = pure_cosine_similarity(v1, v1)
        assert -1.0 <= res <= 1.0


# =============================================================================
# 2. Pure Cosine Similarity Validation and Error Handling
# =============================================================================


class TestPureCosineSimilarityValidation:
    """Verifies strict input validation and descriptive error reporting."""

    def test_empty_vectors_raises_value_error(self):
        """Empty vectors must raise ValueError with clear message."""
        with pytest.raises(ValueError, match="Vectors must not be empty"):
            pure_cosine_similarity([], [])

        with pytest.raises(ValueError, match="Vectors must not be empty"):
            pure_cosine_similarity([1.0], [])

        with pytest.raises(ValueError, match="Vectors must not be empty"):
            pure_cosine_similarity([], [1.0])

    def test_mismatched_dimensions_raises_value_error(self):
        """Vectors of unequal dimension must raise ValueError."""
        with pytest.raises(ValueError, match="Vectors must have the same dimension: 2 != 3"):
            pure_cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Vectors must have the same dimension: 3 != 1"):
            pure_cosine_similarity([1.0, 2.0, 3.0], [1.0])

    def test_none_input_raises_value_error(self):
        """Passing None for either vector must raise ValueError."""
        with pytest.raises(ValueError, match="Vectors must not be None"):
            pure_cosine_similarity(None, [1.0, 2.0])  # type: ignore

        with pytest.raises(ValueError, match="Vectors must not be None"):
            pure_cosine_similarity([1.0, 2.0], None)  # type: ignore

    def test_non_sequence_input_raises_type_error(self):
        """Passing a non-sequence object must raise TypeError."""
        with pytest.raises(TypeError, match="Vectors must be sequences with a defined length"):
            pure_cosine_similarity(12345, [1.0, 2.0])  # type: ignore

    def test_non_numeric_elements_raises_type_error(self):
        """Passing non-numeric elements or booleans must raise TypeError."""
        with pytest.raises(TypeError, match="Vector A element at index 0 must be numeric"):
            pure_cosine_similarity(["invalid", 1.0], [1.0, 2.0])  # type: ignore

        with pytest.raises(TypeError, match="Vector B element at index 1 must be numeric"):
            pure_cosine_similarity([1.0, 2.0], [1.0, None])  # type: ignore

        with pytest.raises(TypeError, match="Vector A element at index 0 must be numeric, got bool"):
            pure_cosine_similarity([True, False], [1.0, 0.0])  # type: ignore

    def test_non_finite_elements_raises_value_error(self):
        """Vectors containing NaN or Inf must raise ValueError."""
        with pytest.raises(ValueError, match="Vector A contains non-finite value"):
            pure_cosine_similarity([float("nan"), 1.0], [1.0, 2.0])

        with pytest.raises(ValueError, match="Vector B contains non-finite value"):
            pure_cosine_similarity([1.0, 2.0], [float("inf"), 2.0])

        with pytest.raises(ValueError, match="Vector B contains non-finite value"):
            pure_cosine_similarity([1.0, 2.0], [1.0, float("-inf")])


# =============================================================================
# 3. SkillEmbeddingModel Initialization and Lazy Loading
# =============================================================================


class TestSkillEmbeddingModelInitialization:
    """Verifies that model loading is lazy and does not trigger downloads automatically."""

    def test_default_initialization_is_lazy(self):
        """Model instance must initialize without loading weights or downloading models."""
        model = SkillEmbeddingModel()
        assert model.model_name == SkillEmbeddingModel.DEFAULT_MODEL_NAME
        assert model.is_loaded is False
        assert model._model is None

    def test_custom_model_name_initialization(self):
        """Custom model name must be retained without loading weights."""
        custom_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        model = SkillEmbeddingModel(model_name=custom_name)
        assert model.model_name == custom_name
        assert model.is_loaded is False
        assert model._model is None

    def test_load_model_raises_import_error_when_package_unavailable(self):
        """Calling load_model when sentence-transformers is missing must raise ImportError."""
        with patch.object(embeddings, "SentenceTransformer", None), patch.dict(
            "sys.modules", {"sentence_transformers": None}
        ):
            model = SkillEmbeddingModel()
            with pytest.raises(
                ImportError,
                match="sentence-transformers is not installed. Install with 'pip install sentence-transformers'.",
            ):
                model.load_model()

    def test_load_model_succeeds_with_mock(self):
        """Calling load_model with a mocked SentenceTransformer loads the model successfully."""
        mock_st_class = MagicMock()
        mock_st_instance = MagicMock()
        mock_st_class.return_value = mock_st_instance

        with patch.object(embeddings, "SentenceTransformer", mock_st_class):
            model = SkillEmbeddingModel("custom-model")
            assert not model.is_loaded
            model.load_model()
            assert model.is_loaded
            assert model._model is mock_st_instance
            mock_st_class.assert_called_once_with("custom-model")


# =============================================================================
# 4. SkillEmbeddingModel Encode Input Validation
# =============================================================================


class TestSkillEmbeddingModelEncodeValidation:
    """Verifies input validation on encode() before model execution."""

    @pytest.fixture
    def unloaded_model(self) -> SkillEmbeddingModel:
        return SkillEmbeddingModel()

    def test_encode_unloaded_model_raises_runtime_error(
        self, unloaded_model: SkillEmbeddingModel
    ):
        """Valid text passed to unloaded model must raise RuntimeError."""
        with pytest.raises(
            RuntimeError, match="Model is not loaded. Call load_model\\(\\) before encoding."
        ):
            unloaded_model.encode("Python Programming")

        with pytest.raises(
            RuntimeError, match="Model is not loaded. Call load_model\\(\\) before encoding."
        ):
            unloaded_model.encode(["Python", "Machine Learning"])

    def test_encode_rejects_none(self, unloaded_model: SkillEmbeddingModel):
        """Passing None to encode must raise ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be None"):
            unloaded_model.encode(None)  # type: ignore

    def test_encode_rejects_empty_string(self, unloaded_model: SkillEmbeddingModel):
        """Passing an empty string must raise ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be empty or whitespace-only"):
            unloaded_model.encode("")

    def test_encode_rejects_whitespace_string(self, unloaded_model: SkillEmbeddingModel):
        """Passing whitespace-only strings must raise ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be empty or whitespace-only"):
            unloaded_model.encode("     ")

        with pytest.raises(ValueError, match="Input text cannot be empty or whitespace-only"):
            unloaded_model.encode("\t\n  \r\n")

    def test_encode_rejects_empty_list(self, unloaded_model: SkillEmbeddingModel):
        """Passing an empty list/collection must raise ValueError."""
        with pytest.raises(ValueError, match="Input text collection cannot be empty"):
            unloaded_model.encode([])

        with pytest.raises(ValueError, match="Input text collection cannot be empty"):
            unloaded_model.encode(())

    def test_encode_rejects_list_with_invalid_entries(
        self, unloaded_model: SkillEmbeddingModel
    ):
        """Collections with empty, whitespace, None, or non-string entries must be rejected."""
        with pytest.raises(
            ValueError,
            match="Invalid text entry at index 1: text cannot be empty or whitespace-only",
        ):
            unloaded_model.encode(["valid text", ""])

        with pytest.raises(
            ValueError,
            match="Invalid text entry at index 2: text cannot be empty or whitespace-only",
        ):
            unloaded_model.encode(["valid 1", "valid 2", "   "])

        with pytest.raises(
            ValueError, match="Invalid text entry at index 0: text cannot be None"
        ):
            unloaded_model.encode([None, "valid"])  # type: ignore

        with pytest.raises(
            TypeError, match="Invalid text entry at index 1: expected str, got int"
        ):
            unloaded_model.encode(["valid", 123])  # type: ignore

    def test_encode_rejects_invalid_root_type(self, unloaded_model: SkillEmbeddingModel):
        """Passing a non-str, non-sequence type must raise TypeError."""
        with pytest.raises(TypeError, match="Expected str or List\\[str\\], got int"):
            unloaded_model.encode(42)  # type: ignore

        with pytest.raises(TypeError, match="Expected str or List\\[str\\], got dict"):
            unloaded_model.encode({"key": "val"})  # type: ignore

    def test_encode_delegates_to_mocked_model(self):
        """Mocked model must receive valid inputs and return encoded results."""
        mock_backend = MagicMock()
        mock_backend.encode.return_value = (
            np.array([0.1, 0.2, 0.3]) if np is not None else [0.1, 0.2, 0.3]
        )

        model = SkillEmbeddingModel()
        model._model = mock_backend

        # Single string
        res = model.encode("Data Engineering")
        mock_backend.encode.assert_called_with("Data Engineering")
        assert res is not None

        # List of strings
        texts = ["Machine Learning", "Cloud Infrastructure"]
        model.encode(texts)
        mock_backend.encode.assert_called_with(texts)


# =============================================================================
# 5. SkillEmbeddingModel Compute Similarity Tests
# =============================================================================


class TestSkillEmbeddingModelComputeSimilarity:
    """Verifies compute_similarity dimension checks, vector shapes, and calculations."""

    @pytest.fixture
    def model(self) -> SkillEmbeddingModel:
        return SkillEmbeddingModel()

    def test_compute_similarity_pure_lists(self, model: SkillEmbeddingModel):
        """Pure list inputs must compute correct similarities."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert model.compute_similarity(v1, v2) == pytest.approx(0.0, abs=1e-7)

        v3 = [2.0, 3.0]
        assert model.compute_similarity(v3, v3) == pytest.approx(1.0, abs=1e-7)

    @pytest.mark.skipif(np is None, reason="NumPy not installed")
    def test_compute_similarity_numpy_arrays(self, model: SkillEmbeddingModel):
        """1D NumPy arrays must calculate correctly."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert model.compute_similarity(a, b) == pytest.approx(0.0, abs=1e-7)

        c = np.array([1.0, 2.0, 3.0])
        d = np.array([2.0, 4.0, 6.0])
        assert model.compute_similarity(c, d) == pytest.approx(1.0, abs=1e-7)

    @pytest.mark.skipif(np is None, reason="NumPy not installed")
    def test_compute_similarity_mixed_numpy_and_list(self, model: SkillEmbeddingModel):
        """Mixed numpy array and list inputs must be normalized and handled seamlessly."""
        a = np.array([1.0, 2.0, 3.0])
        b = [2.0, 4.0, 6.0]
        assert model.compute_similarity(a, b) == pytest.approx(1.0, abs=1e-7)
        assert model.compute_similarity(b, a) == pytest.approx(1.0, abs=1e-7)

    @pytest.mark.skipif(np is None, reason="NumPy not installed")
    def test_compute_similarity_2d_single_vector_shapes(self, model: SkillEmbeddingModel):
        """2D arrays representing a single vector (1, D) or (D, 1) must be safely flattened."""
        a_row = np.array([[1.0, 2.0, 3.0]])
        b_col = np.array([[1.0], [2.0], [3.0]])
        c_1d = np.array([1.0, 2.0, 3.0])

        assert model.compute_similarity(a_row, c_1d) == pytest.approx(1.0, abs=1e-7)
        assert model.compute_similarity(b_col, c_1d) == pytest.approx(1.0, abs=1e-7)
        assert model.compute_similarity(a_row, b_col) == pytest.approx(1.0, abs=1e-7)

    @pytest.mark.skipif(np is None, reason="NumPy not installed")
    def test_compute_similarity_rejects_multi_row_matrix(self, model: SkillEmbeddingModel):
        """2D matrices with shape (N, D) where N > 1 must be rejected with ValueError."""
        matrix_2d = np.ones((2, 3))
        vector_1d = np.ones((3,))

        with pytest.raises(ValueError, match="must be a 1D vector, got array with shape \\(2, 3\\)"):
            model.compute_similarity(matrix_2d, vector_1d)

        with pytest.raises(ValueError, match="must be a 1D vector, got array with shape \\(2, 3\\)"):
            model.compute_similarity(vector_1d, matrix_2d)

    @pytest.mark.skipif(np is None, reason="NumPy not installed")
    def test_compute_similarity_rejects_3d_array(self, model: SkillEmbeddingModel):
        """Higher dimensional arrays must be rejected."""
        arr_3d = np.ones((1, 2, 3))
        with pytest.raises(ValueError, match="must be a 1D vector, got array with shape"):
            model.compute_similarity(arr_3d, [1.0, 2.0, 3.0])

    def test_compute_similarity_empty_vectors_raises_value_error(
        self, model: SkillEmbeddingModel
    ):
        """Empty lists or empty arrays must raise ValueError."""
        with pytest.raises(ValueError, match="embedding_a must not be empty"):
            model.compute_similarity([], [1.0, 2.0])

        with pytest.raises(ValueError, match="embedding_b must not be empty"):
            model.compute_similarity([1.0, 2.0], [])

        if np is not None:
            with pytest.raises(ValueError, match="embedding_a must not be empty"):
                model.compute_similarity(np.array([]), np.array([1.0]))

    def test_compute_similarity_mismatched_dimensions_raises_value_error(
        self, model: SkillEmbeddingModel
    ):
        """Vectors with different dimensions must raise ValueError."""
        with pytest.raises(ValueError, match="Embedding dimensions do not match: 2 != 3"):
            model.compute_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

        if np is not None:
            with pytest.raises(ValueError, match="Embedding dimensions do not match: 2 != 4"):
                model.compute_similarity(np.zeros(2), np.zeros(4))

    def test_compute_similarity_zero_norm_vectors(self, model: SkillEmbeddingModel):
        """Zero norm inputs must return 0.0 safely."""
        assert model.compute_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert model.compute_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0
        assert model.compute_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

        if np is not None:
            assert model.compute_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0

    def test_compute_similarity_non_finite_values_raises_value_error(
        self, model: SkillEmbeddingModel
    ):
        """Vectors containing NaN or Inf must raise ValueError."""
        with pytest.raises(ValueError, match="contains non-finite value"):
            model.compute_similarity([float("nan"), 1.0], [1.0, 2.0])

        if np is not None:
            with pytest.raises(ValueError, match="contains non-finite values"):
                model.compute_similarity(np.array([np.nan, 1.0]), np.array([1.0, 2.0]))

            with pytest.raises(ValueError, match="contains non-finite values"):
                model.compute_similarity(np.array([1.0, 2.0]), np.array([np.inf, 2.0]))

    def test_compute_similarity_nested_sequence_raises_value_error(
        self, model: SkillEmbeddingModel
    ):
        """Passing nested lists (e.g. [[1, 2]]) must raise ValueError."""
        with pytest.raises(ValueError, match="must be a 1D vector, got nested sequence"):
            model.compute_similarity([[1.0, 2.0]], [1.0, 2.0])


# =============================================================================
# 6. SkillEmbeddingModel Find Top-K Ranking and Edge Cases
# =============================================================================


class TestSkillEmbeddingModelFindTopK:
    """Verifies find_top_k ranking accuracy, edge case handling, and vectorization."""

    @pytest.fixture
    def model(self) -> SkillEmbeddingModel:
        return SkillEmbeddingModel()

    def test_find_top_k_descending_order(self, model: SkillEmbeddingModel):
        """Results must be ordered by strictly descending similarity scores."""
        query = [1.0, 0.0, 0.0]
        candidates = {
            "c_high": [0.95, 0.05, 0.0],
            "c_ortho": [0.0, 1.0, 0.0],
            "c_med": [0.7, 0.3, 0.0],
            "c_opp": [-1.0, 0.0, 0.0],
        }

        results = model.find_top_k(query, candidates, top_k=4)
        assert len(results) == 4

        # Verify ordering: c_high > c_med > c_ortho > c_opp
        candidate_ids = [cid for cid, _ in results]
        assert candidate_ids == ["c_high", "c_med", "c_ortho", "c_opp"]

        # Verify scores are strictly decreasing
        scores = [score for _, score in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

        # Check approximate similarity values
        assert results[0][1] == pytest.approx(0.95 / math.sqrt(0.95**2 + 0.05**2), abs=1e-5)
        assert results[1][1] == pytest.approx(0.7 / math.sqrt(0.7**2 + 0.3**2), abs=1e-5)
        assert results[2][1] == pytest.approx(0.0, abs=1e-7)
        assert results[3][1] == pytest.approx(-1.0, abs=1e-7)

    def test_find_top_k_k_smaller_than_candidate_count(self, model: SkillEmbeddingModel):
        """When top_k < len(candidates), return exactly top_k items."""
        query = [1.0, 0.0]
        candidates = {
            "c1": [1.0, 0.0],
            "c2": [0.8, 0.2],
            "c3": [0.5, 0.5],
            "c4": [0.0, 1.0],
        }
        results = model.find_top_k(query, candidates, top_k=2)
        assert len(results) == 2
        assert [cid for cid, _ in results] == ["c1", "c2"]

    def test_find_top_k_k_equals_candidate_count(self, model: SkillEmbeddingModel):
        """When top_k == len(candidates), return all candidates sorted."""
        query = [1.0, 0.0]
        candidates = {
            "c1": [0.5, 0.5],
            "c2": [1.0, 0.0],
        }
        results = model.find_top_k(query, candidates, top_k=2)
        assert len(results) == 2
        assert [cid for cid, _ in results] == ["c2", "c1"]

    def test_find_top_k_k_greater_than_candidate_count(self, model: SkillEmbeddingModel):
        """When top_k > len(candidates), return all available candidates without padding."""
        query = [1.0, 0.0]
        candidates = {
            "c1": [0.9, 0.1],
            "c2": [0.1, 0.9],
        }
        results = model.find_top_k(query, candidates, top_k=10)
        assert len(results) == 2
        assert [cid for cid, _ in results] == ["c1", "c2"]

    def test_find_top_k_empty_candidate_collection_returns_empty_list(
        self, model: SkillEmbeddingModel
    ):
        """Empty candidate dictionary must safely return an empty list."""
        assert model.find_top_k([1.0, 0.0], {}, top_k=5) == []

    def test_find_top_k_non_positive_k_returns_empty_list(
        self, model: SkillEmbeddingModel
    ):
        """top_k <= 0 must safely return an empty list."""
        query = [1.0, 0.0]
        candidates = {"c1": [1.0, 0.0], "c2": [0.0, 1.0]}

        assert model.find_top_k(query, candidates, top_k=0) == []
        assert model.find_top_k(query, candidates, top_k=-1) == []
        assert model.find_top_k(query, candidates, top_k=-10) == []

    def test_find_top_k_invalid_top_k_type_raises_type_error(
        self, model: SkillEmbeddingModel
    ):
        """Non-integer top_k values or booleans must raise TypeError."""
        query = [1.0, 0.0]
        candidates = {"c1": [1.0, 0.0]}

        with pytest.raises(TypeError, match="top_k must be an integer, got str"):
            model.find_top_k(query, candidates, top_k="3")  # type: ignore

        with pytest.raises(TypeError, match="top_k must be an integer, got bool"):
            model.find_top_k(query, candidates, top_k=True)  # type: ignore

        with pytest.raises(TypeError, match="top_k must be an integer, got float"):
            model.find_top_k(query, candidates, top_k=2.5)  # type: ignore

    def test_find_top_k_invalid_candidates_type_raises_type_error(
        self, model: SkillEmbeddingModel
    ):
        """Passing non-dict for candidate_embeddings must raise TypeError."""
        query = [1.0, 0.0]
        with pytest.raises(TypeError, match="candidate_embeddings must be a dict"):
            model.find_top_k(query, None, top_k=5)  # type: ignore

        with pytest.raises(TypeError, match="candidate_embeddings must be a dict"):
            model.find_top_k(query, [("c1", [1.0, 0.0])], top_k=5)  # type: ignore

    def test_find_top_k_empty_query_raises_value_error(
        self, model: SkillEmbeddingModel
    ):
        """Passing an empty query vector must raise ValueError."""
        candidates = {"c1": [1.0, 0.0]}
        with pytest.raises(ValueError, match="query_embedding must not be empty"):
            model.find_top_k([], candidates, top_k=5)

    def test_find_top_k_empty_candidate_vector_raises_value_error(
        self, model: SkillEmbeddingModel
    ):
        """If any candidate vector is empty, raise ValueError."""
        query = [1.0, 0.0]
        candidates = {"c1": [1.0, 0.0], "c2": []}
        with pytest.raises(ValueError, match="candidate 'c2' must not be empty"):
            model.find_top_k(query, candidates, top_k=5)

    def test_find_top_k_mismatched_candidate_dimension_raises_value_error(
        self, model: SkillEmbeddingModel
    ):
        """If a candidate vector has mismatched dimension, raise ValueError."""
        query = [1.0, 0.0]
        candidates = {"c1": [1.0, 0.0], "c2": [1.0, 0.0, 0.0]}
        with pytest.raises(
            ValueError,
            match="Candidate 'c2' dimension \\(3\\) does not match query dimension \\(2\\)",
        ):
            model.find_top_k(query, candidates, top_k=5)

    def test_find_top_k_zero_norm_query_and_candidates(
        self, model: SkillEmbeddingModel
    ):
        """Zero norm query or candidates must evaluate to 0.0 similarity safely."""
        query_zero = [0.0, 0.0]
        candidates = {
            "c1": [1.0, 2.0],
            "c2": [0.0, 0.0],
        }
        res = model.find_top_k(query_zero, candidates, top_k=5)
        assert len(res) == 2
        for _, score in res:
            assert score == 0.0

        query_normal = [1.0, 0.0]
        res2 = model.find_top_k(query_normal, candidates, top_k=5)
        assert res2[0][0] == "c1"
        assert res2[1][0] == "c2"
        assert res2[1][1] == 0.0

    @pytest.mark.skipif(np is None, reason="NumPy not installed")
    def test_find_top_k_vectorized_matches_python_loop(
        self, model: SkillEmbeddingModel
    ):
        """Vectorized NumPy execution must match pure Python scores within FP tolerance."""
        query_list = [0.3, 0.5, -0.2, 0.8]
        candidates_dict_list: Dict[str, List[float]] = {
            "item_a": [0.1, 0.9, -0.1, 0.2],
            "item_b": [0.3, 0.5, -0.2, 0.8],
            "item_c": [-0.3, -0.5, 0.2, -0.8],
            "item_d": [0.0, 0.0, 0.0, 0.0],
            "item_e": [0.5, -0.2, 0.7, 0.1],
        }

        query_np = np.array(query_list)
        candidates_np = {k: np.array(v) for k, v in candidates_dict_list.items()}

        top_np = model.find_top_k(query_np, candidates_np, top_k=5)

        expected_scores = [
            (k, pure_cosine_similarity(query_list, v))
            for k, v in candidates_dict_list.items()
        ]
        expected_scores.sort(key=lambda x: x[1], reverse=True)

        assert len(top_np) == len(expected_scores)
        for (k_np, s_np), (k_exp, s_exp) in zip(top_np, expected_scores):
            assert k_np == k_exp
            assert s_np == pytest.approx(s_exp, abs=1e-6)
