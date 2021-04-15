from __future__ import annotations

import typing as t
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import spacy
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from scipy.spatial import distance
from spacy.tokens import Doc, DocBin  # type: ignore

from recap_nlp import common
from recap_nlp.common import BaseQuery


def _check_response(r: httpx.Response) -> None:
    if r.status_code != httpx.codes.OK:
        raise RuntimeError(r.content)


_vector_cache = {}
Vector = t.Union[np.ndarray, t.Tuple[np.ndarray, ...]]


def _convert_vector(
    vector: t.Union[t.Tuple[t.Tuple[float, ...], ...], t.Tuple[float, ...]]
) -> Vector:
    if any(isinstance(i, float) for i in vector):
        return np.array(vector)
    else:
        return tuple(np.array(v) for v in vector)


@dataclass
class Client:
    host: str
    port: int
    base_query: common.BaseQuery

    # Select your transport with a defined url endpoint
    transport = AIOHTTPTransport(url="https://countries.trevorblades.com/")

    # Create a GraphQL client using the defined transport
    client = Client(transport=transport, fetch_schema_from_transport=True)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url(self, *parts: str) -> str:
        return "/".join([self.base_url, *parts])

    def vector(self, text: str) -> Vector:
        if text not in _vector_cache:
            response = session.post(
                self.url("vector"), json={"text": text, **self.base_query.dict()}
            )
            _check_response(response)

            _vector_cache[text] = _convert_vector(response.json())

        return _vector_cache[text]

    def vectors(self, texts: t.Iterable[str]) -> t.Tuple[Vector, ...]:
        unknown_texts = [text for text in texts if text not in _vector_cache]

        if unknown_texts:
            response = session.post(
                self.url("vectors"),
                json={"texts": unknown_texts, **self.base_query.dict()},
            )
            _check_response(response)

            for text, vector in zip(unknown_texts, response.json()):
                _vector_cache[text] = _convert_vector(vector)

        return tuple(_vector_cache[text] for text in texts)

    def model(self, text: str) -> Doc:
        with session.stream(
            "POST", self.url("model"), json={"text": text, **self.base_query.dict()}
        ) as response:
            _check_response(response)
            response_bytes = response.read()

        nlp = spacy.blank(self.base_query.language)
        docbin = DocBin().from_bytes(response_bytes)

        return tuple(docbin.get_docs(nlp.vocab))[0]

    def models(self, texts: t.Iterable[str]) -> t.Tuple[Doc, ...]:
        with session.stream(
            "POST", self.url("model"), json={"texts": texts, **self.base_query.dict()}
        ) as response:
            _check_response(response)
            response_bytes = response.read()

        nlp = spacy.blank(self.base_query.language)
        docbin = DocBin().from_bytes(response_bytes)

        return tuple(docbin.get_docs(nlp.vocab))

    def similarity(
        self,
        obj1: t.Union[str, Vector],
        obj2: t.Union[str, Vector],
    ) -> float:
        if isinstance(obj1, str):
            obj1 = self.vector(obj1)
        if isinstance(obj2, str):
            obj2 = self.vector(obj2)

        if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
            if np.any(obj1) and np.any(obj2):
                return float(1 - distance.cosine(obj1, obj2))
            return 0.0

        elif isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
            return dynamax_jaccard(obj1, obj2)

        else:
            raise ValueError(
                "Both vectors must have the same format (either 'np.ndarray' or 'List[np.ndarray]')."
            )


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
