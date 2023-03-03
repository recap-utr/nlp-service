from __future__ import annotations

import typing as t

import numpy as np
import spacy
from arg_services.nlp.v1 import nlp_pb2
from spacy.language import Language as SpacyLanguage
from spacy.tokens import Doc, DocBin, Span, Token

from nlp_service.similarity import SimilarityFactory as SimilarityFactory
from nlp_service.typing import ArrayLike, NumpyVector

Doc.set_extension("vector", default=None)
Span.set_extension("vector", default=None)
Token.set_extension("vector", default=None)


def blank(
    language: str,
    similarity_method: nlp_pb2.SimilarityMethod.ValueType = nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_UNSPECIFIED,
) -> SpacyLanguage:
    spacy_lang = spacy.blank(language)
    inject_pipes(spacy_lang, similarity_method)

    return spacy_lang


def docbin2docs(
    docbin_bytes: bytes,
    language: t.Union[str, SpacyLanguage],
    similarity_method: nlp_pb2.SimilarityMethod.ValueType = nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_UNSPECIFIED,
) -> t.Tuple[Doc, ...]:
    if isinstance(language, str):
        language = blank(language, similarity_method)

    docbin = DocBin().from_bytes(docbin_bytes)

    return tuple(docbin.get_docs(language.vocab))


def list2array(values: ArrayLike) -> NumpyVector:
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


def inject_pipes(
    nlp: SpacyLanguage,
    similarity_method: nlp_pb2.SimilarityMethod.ValueType = nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_UNSPECIFIED,
) -> None:
    nlp.add_pipe("remote_vector", last=True)
    nlp.add_pipe("similarity_factory", last=True, config={"method": similarity_method})


@SpacyLanguage.component("remote_vector")
def remote_vector(doc):
    def func(x):
        return x._.vector

    doc.user_hooks["vector"] = func
    doc.user_span_hooks["vector"] = func
    doc.user_token_hooks["vector"] = func

    return doc
