import logging

from . import client, sim_funcs, typing
from .lib import doc, docs, similarities, similarity, vector, vectors
from .nlp_pb import EmbeddingModel, EmbeddingType, NlpConfig, Pooling, SimilarityMethod

__all__ = (
    "client",
    "sim_funcs",
    "typing",
    "docs",
    "doc",
    "vectors",
    "vector",
    "similarities",
    "similarity",
    "NlpConfig",
    "EmbeddingModel",
    "SimilarityMethod",
    "EmbeddingType",
    "Pooling",
)

logging.getLogger(__name__).addHandler(logging.NullHandler())
