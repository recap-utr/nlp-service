from __future__ import annotations

import contextlib
import multiprocessing
import socket
import typing as t
from abc import ABC, abstractmethod
from concurrent import futures

import arg_services_helper
import grpc
import numpy as np
import spacy
import tensorflow_hub as hub
import typer
from arg_services.base.v1 import base_pb2
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc, DocBin, Span, Token  # type: ignore

from recap_nlp import fuzzymax


def cosine(obj1, obj2) -> float:
    return 1 - distance.cosine(obj1.vector, obj2.vector)


def dynamax_jaccard(obj1, obj2) -> float:
    vecs1 = _token_vectors(obj1)
    vecs2 = _token_vectors(obj2)

    return fuzzymax.dynamax_jaccard(vecs1, vecs2)


def _token_vectors(obj) -> t.List[np.ndarray]:
    if isinstance(obj, Token):
        return [obj.vector]

    return [t.vector for t in obj]


mapping = {
    nlp_pb2.SIMILARITY_METHOD_COSINE: cosine,
    nlp_pb2.SIMILARITY_METHOD_DYNAMAX_JACCARD: dynamax_jaccard,
}
