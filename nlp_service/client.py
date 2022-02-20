from __future__ import annotations

import typing as t

import numpy as np
import spacy
from arg_services.base.v1 import base_pb2
from arg_services.nlp.v1 import nlp_pb2
from spacy.language import Language
from spacy.tokens import Doc, DocBin, Span, Token

from nlp_service import similarity

Doc.set_extension("vector", default=None)
Span.set_extension("vector", default=None)
Token.set_extension("vector", default=None)


def blank(language: str, similarity_method: int = 0) -> Language:
    spacy_lang = spacy.blank(language)
    inject_pipes(spacy_lang, similarity_method)

    return spacy_lang


def docbin2docs(
    docbin_bytes: bytes, language: t.Union[str, Language], similarity_method: int = 0
) -> t.Tuple[Doc, ...]:
    if isinstance(language, str):
        language = blank(language, similarity_method)

    docbin = DocBin().from_bytes(docbin_bytes)

    return tuple(docbin.get_docs(language.vocab))


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


def inject_pipes(nlp: Language, similarity_method: int = 0) -> None:
    nlp.add_pipe("user_vector", last=True)
    nlp.add_pipe("similarity_method", last=True, config={"method": similarity_method})


@Language.component("user_vector")
def _vector_component(doc):
    func = lambda x: x._.vector

    doc.user_hooks["vector"] = func
    doc.user_span_hooks["vector"] = func
    doc.user_token_hooks["vector"] = func

    return doc


@Language.factory("similarity_method")
class SimilarityFactory:
    def __init__(self, nlp, name, method):
        if method:
            self.func = similarity.mapping[method]

    def __call__(self, doc):
        if self.func:
            doc.user_hooks["similarity"] = self.func
            doc.user_span_hooks["similarity"] = self.func
            doc.user_token_hooks["similarity"] = self.func

        return doc
