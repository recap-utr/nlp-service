import logging

from arg_services.nlp.v1.nlp_pb2 import (
    EmbeddingModel,
    EmbeddingType,
    NlpConfig,
    Pooling,
    SimilarityMethod,
)

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
