from __future__ import annotations

import itertools
import typing as t

import nltk.metrics as nltk_dist
import numpy as np
from arg_services.nlp.v1 import nlp_pb2
from scipy.spatial import distance as scipy_dist
from spacy.tokens import Doc, Span, Token  # type: ignore

SpacyObj = t.Union[Doc, Token, Span]

# https://github.com/babylonhealth/fuzzymax/blob/master/similarity/fuzzy.py
def _fuzzify(s, u):
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


def cosine(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if np.any(vec1) and np.any(vec2):
        return 1 - scipy_dist.cosine(vec1, vec2)

    return 0.0


def _cosine(obj1: SpacyObj, obj2: SpacyObj) -> float:
    return 1 - scipy_dist.cosine(obj1.vector, obj2.vector)


def angular(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.any() and vec2.any():
        try:
            return (
                1.0
                - np.arccos(
                    np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                )
                / np.pi
            )
        except Exception:
            pass

    return 0.0


def _angular(obj1: SpacyObj, obj2: SpacyObj) -> float:
    return angular(obj1.vector, obj2.vector)


def dynamax_jaccard(x: t.Iterable[np.ndarray], y: t.Iterable[np.ndarray]) -> float:
    """
    DynaMax-Jaccard similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    u = np.vstack((x, y))
    m_x = _fuzzify(x, u)
    m_y = _fuzzify(y, u)

    m_inter = np.sum(np.minimum(m_x, m_y))
    m_union = np.sum(np.maximum(m_x, m_y))
    return m_inter / m_union


def _dynamax_jaccard(obj1: SpacyObj, obj2: SpacyObj) -> float:
    vecs1 = _token_vectors(obj1)
    vecs2 = _token_vectors(obj2)

    return dynamax_jaccard(vecs1, vecs2)


def maxpool_jaccard(x: t.Iterable[np.ndarray], y: t.Iterable[np.ndarray]) -> float:
    """
    MaxPool-Jaccard similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    m_x = np.max(x, axis=0)
    m_x = np.maximum(m_x, 0, m_x)
    m_y = np.max(y, axis=0)
    m_y = np.maximum(m_y, 0, m_y)
    m_inter = np.sum(np.minimum(m_x, m_y))
    m_union = np.sum(np.maximum(m_x, m_y))
    return m_inter / m_union


def _maxpool_jaccard(obj1: SpacyObj, obj2: SpacyObj) -> float:
    vecs1 = _token_vectors(obj1)
    vecs2 = _token_vectors(obj2)

    return maxpool_jaccard(vecs1, vecs2)


def dynamax_dice(x: t.Iterable[np.ndarray], y: t.Iterable[np.ndarray]) -> float:
    """
    DynaMax-Dice similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    u = np.vstack((x, y))
    m_x = _fuzzify(x, u)
    m_y = _fuzzify(y, u)

    f_inter = np.sum(np.minimum(m_x, m_y))
    m_x_card = np.sum(m_x)
    m_y_card = np.sum(m_y)
    return 2 * f_inter / (m_x_card + m_y_card)


def _dynamax_dice(obj1, obj2) -> float:
    vecs1 = _token_vectors(obj1)
    vecs2 = _token_vectors(obj2)

    return dynamax_dice(vecs1, vecs2)


def dynamax_otsuka(x: t.Iterable[np.ndarray], y: t.Iterable[np.ndarray]) -> float:
    """
    DynaMax-Otsuka similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    u = np.vstack((x, y))
    m_x = _fuzzify(x, u)
    m_y = _fuzzify(y, u)

    m_inter = np.sum(np.minimum(m_x, m_y))
    m_x_card = np.sum(m_x)
    m_y_card = np.sum(m_y)
    return m_inter / np.sqrt(m_x_card * m_y_card)


def _dynamax_otsuka(obj1: SpacyObj, obj2: SpacyObj) -> float:
    vecs1 = _token_vectors(obj1)
    vecs2 = _token_vectors(obj2)

    return dynamax_otsuka(vecs1, vecs2)


def fbow_jaccard_factory(u):
    """
    Factory for building FBoW-Jaccard similarity measures
    with the custom universe matrix U
    :param u: the universe matrix U
    :return: similarity function
    """

    def u_jaccard(x, y):
        m_x = _fuzzify(x, u)
        m_y = _fuzzify(y, u)

        m_inter = np.sum(np.minimum(m_x, m_y))
        m_union = np.sum(np.maximum(m_x, m_y))
        return m_inter / m_union

    return u_jaccard


def edit(text1: str, text2: str) -> float:
    return dist2sim(nltk_dist.edit_distance(text1, text2))


def _edit(obj1: SpacyObj, obj2: SpacyObj) -> float:
    return edit(obj1.text, obj2.text)


def jaccard(set1: t.AbstractSet[str], set2: t.AbstractSet[str]) -> float:
    return dist2sim(nltk_dist.jaccard_distance(set1, set2))


def _jaccard(obj1: SpacyObj, obj2: SpacyObj) -> float:
    set1 = _token_set(obj1)
    set2 = _token_set(obj2)

    return jaccard(set1, set2)


def _token_set(obj) -> t.Set[str]:
    if isinstance(obj, Token):
        return set() if obj.is_stop else {obj.text}

    return {t.text for t in obj if not t.is_stop}


def _token_vectors(obj) -> t.List[np.ndarray]:
    return [obj.vector] if isinstance(obj, Token) else [t.vector for t in obj]


def dist2sim(distance: float) -> float:
    return 1 / (1 + distance)


mapping = {
    "cosine": cosine,
    "angular": angular,
    "dynamax_jaccard": dynamax_jaccard,
    "maxpool_jaccard": maxpool_jaccard,
    "dynamax_dice": dynamax_dice,
    "dynamax_otsuka": dynamax_otsuka,
    "edit": edit,
    "jaccard": jaccard,
}

proto_mapping = {
    nlp_pb2.SIMILARITY_METHOD_COSINE: cosine,
    nlp_pb2.SIMILARITY_METHOD_ANGULAR: angular,
    nlp_pb2.SIMILARITY_METHOD_DYNAMAX_JACCARD: dynamax_jaccard,
    nlp_pb2.SIMILARITY_METHOD_MAXPOOL_JACCARD: maxpool_jaccard,
    nlp_pb2.SIMILARITY_METHOD_DYNAMAX_DICE: dynamax_dice,
    nlp_pb2.SIMILARITY_METHOD_DYNAMAX_OTSUKA: dynamax_otsuka,
    nlp_pb2.SIMILARITY_METHOD_EDIT: edit,
    nlp_pb2.SIMILARITY_METHOD_JACCARD: jaccard,
}

spacy_mapping = {
    nlp_pb2.SIMILARITY_METHOD_COSINE: _cosine,
    nlp_pb2.SIMILARITY_METHOD_ANGULAR: _angular,
    nlp_pb2.SIMILARITY_METHOD_DYNAMAX_JACCARD: _dynamax_jaccard,
    nlp_pb2.SIMILARITY_METHOD_MAXPOOL_JACCARD: _maxpool_jaccard,
    nlp_pb2.SIMILARITY_METHOD_DYNAMAX_DICE: _dynamax_dice,
    nlp_pb2.SIMILARITY_METHOD_DYNAMAX_OTSUKA: _dynamax_otsuka,
    nlp_pb2.SIMILARITY_METHOD_EDIT: _edit,
    nlp_pb2.SIMILARITY_METHOD_JACCARD: _jaccard,
}

try:
    from gensim.models import KeyedVectors

    # TODO: Implement generic version `wmd` for use without spacy

    def _wmd(obj1, obj2) -> float:
        words1, vecs1 = _wmd_model(obj1)
        words2, vecs2 = _wmd_model(obj2)

        dim = max(vec.shape[0] for vec in itertools.chain(vecs1, vecs2))
        gensim_model = KeyedVectors(dim)
        gensim_model.add_vectors(words1 + words2, vecs1 + vecs2)

        return dist2sim(gensim_model.wmdistance(words1, words2))

    def _wmd_model(obj) -> t.Tuple[t.List[str], t.List[np.ndarray]]:
        if isinstance(obj, Token):
            return ([], []) if obj.is_stop else ([obj.text], [obj.vector])

        return [t.text for t in obj if not t.is_stop], [
            t.vector for t in obj if not t.is_stop
        ]

    mapping[nlp_pb2.SIMILARITY_METHOD_WMD] = _wmd


except ImportError as e:
    pass
