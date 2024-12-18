import logging

from arg_services.nlp.v1 import nlp_pb2 as model
from torch.cuda import is_available as is_cuda_available

from . import apply, build, client

__all__ = (
    "apply",
    "build",
    "client",
    "model",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

torch_device = "cuda" if is_cuda_available() else "cpu"

logger.info(f"Using torch device '{torch_device}'.")
