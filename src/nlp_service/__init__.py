import logging

from . import client
from .lib import (
    build_concat_embed_func,
    build_embed_func,
    build_sim_func,
    doc,
    docs,
    similarities,
    similarity,
    vector,
    vectors,
)
from .nlp_pb import EmbeddingModel, EmbeddingType, NlpConfig, Pooling, SimilarityMethod

__all__ = (
    "client",
    "docs",
    "doc",
    "vectors",
    "vector",
    "similarities",
    "similarity",
    "build_embed_func",
    "build_concat_embed_func",
    "build_sim_func",
    "NlpConfig",
    "EmbeddingModel",
    "SimilarityMethod",
    "EmbeddingType",
    "Pooling",
)

logging.getLogger(__name__).addHandler(logging.NullHandler())
