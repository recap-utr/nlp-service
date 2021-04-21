from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import grpc
import numpy as np
import spacy
from recap_schema.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from scipy.spatial import distance
from spacy.tokens import Doc, DocBin  # type: ignore


@dataclass
class Client:
    host: str
    port: int
    protocol: str = "http"
    stub: nlp_pb2_grpc.NLPServiceStub = field(init=False)

    def __post_init__(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self.stub = nlp_pb2_grpc.NLPServiceStub(channel)


def docbin2doc(self, docbin_bytes: bytes) -> t.Tuple[Doc, ...]:
    nlp = spacy.blank("en")
    docbin = DocBin().from_bytes(docbin_bytes)

    return tuple(docbin.get_docs(nlp.vocab))


# https://github.com/babylonhealth/fuzzymax/blob/master/similarity/fuzzy.py
def fuzzify(s, u):
    """
    Sentence fuzzifier.
    Computes membership vector for the sentence S with respect to the
    universe U
    :param s: list of word embeddings for the sentence
    :param u: the universe matrix U with shape (K, d)
    :return: membership vectors for the sentence
    """
    f_s = np.dot(s, u.T)
    m_s = np.max(f_s, axis=0)
    m_s = np.maximum(m_s, 0, m_s)
    return m_s


# https://github.com/babylonhealth/fuzzymax/blob/master/similarity/fuzzy.py
def dynamax_jaccard(x, y):
    """
    DynaMax-Jaccard similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    u = np.vstack((x, y))
    m_x = fuzzify(x, u)
    m_y = fuzzify(y, u)

    m_inter = np.sum(np.minimum(m_x, m_y))
    m_union = np.sum(np.maximum(m_x, m_y))

    if m_union > 0:
        return m_inter / m_union

    return 0.0
