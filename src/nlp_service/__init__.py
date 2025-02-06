import logging

from arg_services.nlp.v1 import nlp_pb2 as model
from arg_services.nlp.v1 import nlp_pb2_grpc as rpc

from .lib import Nlp, PipeSelection

__all__ = [
    "Nlp",
    "PipeSelection",
    "model",
    "rpc",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
