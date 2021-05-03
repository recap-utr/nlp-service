from __future__ import annotations

import itertools
import typing as t

import nltk.metrics as nltk_dist
import numpy as np
from arg_services.nlp.v1 import nlp_pb2
from scipy.spatial import distance as scipy_dist
from spacy.tokens import Token  # type: ignore

from nlp_service import fuzzymax


def cosine(obj1, obj2) -> float:
    return 1 - scipy_dist.cosine(obj1.vector, obj2.vector)


def dynamax_jaccard(obj1, obj2) -> float:
    vecs1 = _token_vectors(obj1)
    vecs2 = _token_vectors(obj2)

    return fuzzymax.dynamax_jaccard(vecs1, vecs2)


def maxpool_jaccard(obj1, obj2) -> float:
    vecs1 = _token_vectors(obj1)
    vecs2 = _token_vectors(obj2)

    return fuzzymax.max_jaccard(vecs1, vecs2)


def dynamax_dice(obj1, obj2) -> float:
    vecs1 = _token_vectors(obj1)
    vecs2 = _token_vectors(obj2)

    return fuzzymax.dynamax_dice(vecs1, vecs2)


def dynamax_otsuka(obj1, obj2) -> float:
    vecs1 = _token_vectors(obj1)
    vecs2 = _token_vectors(obj2)

    return fuzzymax.dynamax_otsuka(vecs1, vecs2)


def edit_distance(obj1, obj2) -> float:
    return dist2sim(nltk_dist.edit_distance(obj1.text, obj2.text))


def jaccard_distance(obj1, obj2) -> float:
    set1 = _token_set(obj1)
    set2 = _token_set(obj2)

    return dist2sim(nltk_dist.jaccard_distance(set1, set2))


def _wmd_model(obj) -> t.Tuple[t.List[str], t.List[np.ndarray]]:
    if isinstance(obj, Token):
        if obj.is_stop:
            return [], []

        return [obj.text], [obj.vector]

    return [t.text for t in obj if not t.is_stop], [
        t.vector for t in obj if not t.is_stop
    ]


def _token_set(obj) -> t.AbstractSet[str]:
    if isinstance(obj, Token):
        if obj.is_stop:
            return set()

        return {obj.text}

    return {t.text for t in obj if not t.is_stop}


def _token_vectors(obj) -> t.List[np.ndarray]:
    if isinstance(obj, Token):
        return [obj.vector]

    return [t.vector for t in obj]


def dist2sim(distance: float) -> float:
    return 1 / (1 + distance)


mapping = {
    nlp_pb2.SIMILARITY_METHOD_COSINE: cosine,
    nlp_pb2.SIMILARITY_METHOD_DYNAMAX_JACCARD: dynamax_jaccard,
    nlp_pb2.SIMILARITY_METHOD_MAXPOOL_JACCARD: maxpool_jaccard,
    nlp_pb2.SIMILARITY_METHOD_DYNAMAX_DICE: dynamax_dice,
    nlp_pb2.SIMILARITY_METHOD_DYNAMAX_OTSUKA: dynamax_otsuka,
    nlp_pb2.SIMILARITY_METHOD_EDIT: edit_distance,
    nlp_pb2.SIMILARITY_METHOD_JACCARD: jaccard_distance,
}

try:
    from gensim.models import KeyedVectors

    def wmdistance(obj1, obj2) -> float:
        words1, vecs1 = _wmd_model(obj1)
        words2, vecs2 = _wmd_model(obj2)

        dim = max(vec.shape[0] for vec in itertools.chain(vecs1, vecs2))
        gensim_model = KeyedVectors(dim)
        gensim_model.add_vectors(words1 + words2, vecs1 + vecs2)

        return dist2sim(gensim_model.wmdistance(words1, words2))

    mapping[nlp_pb2.SIMILARITY_METHOD_WMD] = wmdistance


except ImportError as e:
    pass
