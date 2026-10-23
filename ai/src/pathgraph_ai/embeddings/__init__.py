"""Embeddings subpackage for PathGraph AI."""

from pathgraph_ai.embeddings.artifact import (
    EmbeddingArtifact,
    load_embeddings_artifact,
    save_embeddings_artifact,
)
from pathgraph_ai.embeddings.model import (
    SkillEmbeddingModel,
    pure_cosine_similarity,
)
from pathgraph_ai.embeddings.text_serializer import (
    INVALID_TEXT_MARKERS,
    ESCOTextSerializer,
    clean_text,
    parse_alt_labels,
)

__all__ = [
    "EmbeddingArtifact",
    "load_embeddings_artifact",
    "save_embeddings_artifact",
    "SkillEmbeddingModel",
    "pure_cosine_similarity",
    "INVALID_TEXT_MARKERS",
    "ESCOTextSerializer",
    "clean_text",
    "parse_alt_labels",
]
