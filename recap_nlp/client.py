from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import grpc
import numpy as np
import spacy
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from scipy.spatial import distance
from spacy.tokens import Doc, DocBin, Span, Token  # type: ignore

Doc.set_extension("vector", default=None)
Span.set_extension("vector", default=None)
Token.set_extension("vector", default=None)


def docbin2doc(docbin_bytes: bytes) -> t.Tuple[Doc, ...]:
    nlp = spacy.blank("en")
    inject_pipes(nlp)
    docbin = DocBin().from_bytes(docbin_bytes)

    return tuple(docbin.get_docs(nlp.vocab))


def list2array(values: t.Iterable[float]) -> np.ndarray:
    return np.array(values)


def inject_vectors(
    doc: Doc,
    res: nlp_pb2.VectorResponse,
) -> None:
    if res.document:
        doc._.set("vector", list2array(res.document.vector))

    if res.sentences:
        for sent, sent_res in zip(doc.sents, res.sentences):
            sent._.set("vector", list2array(sent_res.vector))

    if res.tokens:
        for token, token_res in zip(doc, res.tokens):
            token._.set("vector", list2array(token_res.vector))


def inject_pipes(nlp: spacy.Language) -> None:
    nlp.add_pipe("vector", last=True)


@spacy.Language.component("vector")
def vector_component(doc):
    func = lambda x: x._.vector

    doc.user_hooks["vector"] = func
    doc.user_span_hooks["vector"] = func
    doc.user_token_hooks["vector"] = func

    return doc


# https://github.com/babylonhealth/fuzzymax/blob/master/similarity/fuzzy.py
def fuzzify(s, u) -> np.ndarray:
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
    return m_s  # type: ignore


# https://github.com/babylonhealth/fuzzymax/blob/master/similarity/fuzzy.py
def dynamax_jaccard(x, y) -> float:
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
