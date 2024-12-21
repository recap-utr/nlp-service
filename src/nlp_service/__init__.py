import logging

from arg_services.nlp.v1.nlp_pb2 import (
    EmbeddingModel,
    EmbeddingType,
    NlpConfig,
    SimilarityMethod,
)

from . import client
from .lib import Nlp, PipeSelection

__all__ = (
    "Nlp",
    "PipeSelection",
    "NlpConfig",
    "SimilarityMethod",
    "EmbeddingType",
    "EmbeddingModel",
    "client",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
