from collections.abc import Mapping, Sequence
from typing import TypedDict

import cbrkit
import spacy
from arg_services.nlp.v1 import nlp_pb2
from spacy.tokens import Doc

from . import build


class PipeSelection(TypedDict, total=False):
    enable: Sequence[str]
    disable: Sequence[str]


def vectors(
    batches: Sequence[str], config: nlp_pb2.NlpConfig
) -> Sequence[cbrkit.typing.NumpyArray]:
    func = build.embed(config)
    return func(batches)


def vector(text: str, config: nlp_pb2.NlpConfig) -> cbrkit.typing.NumpyArray:
    return vectors([text], config)[0]


def docs(
    batches: Sequence[str],
    config: nlp_pb2.NlpConfig,
    pipes_selection: PipeSelection | None = None,
    vectorize: bool = False,
) -> Sequence[Doc]:
    if not pipes_selection:
        pipes_selection = {"disable": []}

    if "enable" in pipes_selection:
        pipes_selection["enable"] = tuple(pipes_selection["enable"])

    nlp = spacy.load(config.spacy_model)

    with nlp.select_pipes(**pipes_selection):
        docs = list(nlp.pipe(batches))

    if vectorize:
        vecs = vectors([doc.text for doc in docs], config)

        for doc, vector in zip(docs, vecs, strict=True):
            doc._.set("vector", vector)

    return docs


def doc(
    text: str,
    config: nlp_pb2.NlpConfig,
    pipes_selection: PipeSelection | None = None,
) -> Doc:
    return docs([text], config, pipes_selection)[0]


def similarities(
    batches: Sequence[tuple[str, str]], config: nlp_pb2.NlpConfig
) -> Sequence[float]:
    func = build.similarity(config)
    return func(batches)


def similarity(text1: str, text2: str, config: nlp_pb2.NlpConfig) -> float:
    return similarities([(text1, text2)], config)[0]


def retrievals[K](
    batches: Sequence[tuple[Mapping[K, str], str]],
    config: nlp_pb2.NlpConfig,
):
    func = build.retrieval(config)
    return func(batches)


def retrieval[K](documents: Mapping[K, str], query, config: nlp_pb2.NlpConfig):
    return retrievals([(documents, query)], config)[0]
