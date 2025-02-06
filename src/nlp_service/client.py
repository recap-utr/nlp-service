import numpy as np
import spacy
from arg_services.nlp.v1 import nlp_pb2 as model
from arg_services.nlp.v1 import nlp_pb2_grpc as rpc
from spacy.language import Language as SpacyLanguage
from spacy.tokens import Doc, DocBin, Span, Token

Doc.set_extension("vector", default=None)
Span.set_extension("vector", default=None)
Token.set_extension("vector", default=None)

__all__ = [
    "model",
    "rpc",
    "blank",
    "docbin2docs",
    "inject_vectors",
    "inject_pipes",
    "remote_vector",
]


def blank(language: str) -> SpacyLanguage:
    spacy_lang = spacy.blank(language)
    inject_pipes(spacy_lang)

    return spacy_lang


def docbin2docs(docbin_bytes: bytes, language: str | SpacyLanguage) -> tuple[Doc, ...]:
    if isinstance(language, str):
        language = blank(language)

    docbin = DocBin().from_bytes(docbin_bytes)

    return tuple(docbin.get_docs(language.vocab))


def inject_vectors(doc: Doc, res: model.VectorResponse) -> None:
    if res.document:
        doc._.set("vector", np.array(res.document.vector))

    if res.sentences:
        for sent, sent_res in zip(doc.sents, res.sentences):
            sent._.set("vector", np.array(sent_res.vector))

    if res.tokens:
        for token, token_res in zip(doc, res.tokens):
            token._.set("vector", np.array(token_res.vector))


def inject_pipes(nlp: SpacyLanguage) -> None:
    nlp.add_pipe("remote_vector", last=True)


@SpacyLanguage.component("remote_vector")
def remote_vector(doc):
    def func(x):
        return x._.vector

    doc.user_hooks["vector"] = func
    doc.user_span_hooks["vector"] = func
    doc.user_token_hooks["vector"] = func

    return doc
