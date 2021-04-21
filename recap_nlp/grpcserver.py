from __future__ import annotations

import asyncio
import io
import typing as t
from abc import ABC, abstractmethod
from base64 import b85decode
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

import graphene as g
import grpclib
import numpy as np
import spacy
import tensorflow_hub as hub
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from grpclib.server import Server
from grpclib.utils import graceful_exit
from recap_schema.nlp.v1 import (
    DocBinResponse,
    EmbeddingLevel,
    EmbeddingModel,
    EmbeddingType,
    Language,
    NlpServiceBase,
    Pooling,
    Vector,
    VectorResponse,
)
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc, DocBin, Span, Token  # type: ignore
from starlette.graphql import GraphQLApp

from recap_nlp import common

# https://spacy.io/usage/processing-pipelines#built-in
spacy_components = (
    "tagger",
    "parser",
    "ner",
    "entity_linker",
    "entity_ruler",
    "textcat",
    "textcat_multilabel",
    "lemmatizer",
    "morphologizer",
    "attribute_ruler",
    "senter",
    "sentencizer",
    # tok2vec, transformer
)


class EmbeddingBase(ABC):
    @abstractmethod
    def __init__(self, model: str, pooling: Pooling) -> None:
        pass

    @abstractmethod
    def vector(self, text: str) -> np.ndarray:
        pass


class TransformerModel(EmbeddingBase):
    def __init__(self, model: str, pooling: Pooling):
        self.model = SentenceTransformer(model)

    def vector(self, text: str):
        embeddings = self.model.encode([text])

        return embeddings[0]


class UseModel(EmbeddingBase):
    def __init__(self, model: str, pooling: Pooling):
        self.model = hub.load(model)

    def vector(self, text: str):
        embeddings = self.model([text])  # type: ignore

        return embeddings[0].numpy()


class SpacyModel(EmbeddingBase):
    def __init__(self, model: str, pooling: Pooling):
        self.model = spacy.load(model, exclude=spacy_components)
        self.pooling = pooling

    def vector(self, text: str):
        doc = self.model(text)

        if len(doc) > 1 and self.pooling != Pooling.POOLING_MEAN_UNSPECIFIED:
            return pool_map[self.pooling]([t.vector for t in doc])

        return doc.vector


@spacy.Language.factory("concat")
class ConcatModel:
    def __init__(self, nlp, name, models):
        self.models = models

    def __call__(self, doc):
        if len(self.models) > 0:
            doc.user_hooks["vector"] = self.vector
            doc.user_span_hooks["vector"] = self.vector
            doc.user_token_hooks["vector"] = self.vector

        return doc

    def vector(self, obj):
        vecs = [model.vector(obj.text) for model in self.models]
        return np.concatenate(vecs)


Doc.set_extension("vector", default=None)
Span.set_extension("vector", default=None)
Token.set_extension("vector", default=None)


def _load_spacy(
    language: "Language",
    spacy_model: str,
    embedding_models: t.Optional[t.List["EmbeddingModel"]],
    **kwargs,
) -> spacy.Language:
    models = []

    if embedding_models:
        for user_model in embedding_models:
            tmp_model = embedding_map[user_model.model_type](
                user_model.model_name, user_model.pooling
            )
            models.append(tmp_model)

    model = spacy.load(spacy_model, **kwargs)
    model.add_pipe(
        "concat",
        last=True,
        config={"models": models},
    )

    return model


pool_map = {
    Pooling.POOLING_MEAN_UNSPECIFIED: np.mean,
    Pooling.POOLING_FIRST: lambda vecs: vecs[0],
    Pooling.POOLING_LAST: lambda vecs: vecs[-1],
    Pooling.POOLING_MIN: np.min,
    Pooling.POOLING_MAX: np.max,
    Pooling.POOLING_SUM: np.sum,
}

embedding_map = {
    EmbeddingType.EMBEDDING_TYPE_SPACY: SpacyModel,
    EmbeddingType.EMBEDDING_TYPE_USE: UseModel,
    EmbeddingType.EMBEDDING_TYPE_SBERT: TransformerModel,
}


class NlpService(NlpServiceBase):
    async def doc_bin(
        self,
        language: "Language",
        texts: t.Optional[t.List[str]],
        spacy_model: str,
        attributes: t.Optional[t.List[str]],
        embedding_levels: t.Optional[t.List["EmbeddingLevel"]],
        embedding_models: t.Optional[t.List["EmbeddingModel"]],
    ) -> "DocBinResponse":
        if not all([texts, spacy_model]):
            raise grpclib.GRPCError(
                grpclib.const.Status.INVALID_ARGUMENT,
                "The arguments 'texts', 'spacy_model' are required.",
            )

        nlp = _load_spacy(language, spacy_model, embedding_models)
        docs = t.cast(t.List[Doc], list(nlp.pipe(texts)))

        if embedding_levels:
            for doc in docs:
                if EmbeddingLevel.EMBEDDING_LEVEL_DOCUMENT in embedding_levels:
                    doc._.set("vector", doc.vector)
                if EmbeddingLevel.EMBEDDING_LEVEL_TOKENS in embedding_levels:
                    for token in doc:
                        token._.set("vector", token.vector)
                if EmbeddingLevel.EMBEDDING_LEVEL_SENTENCES in embedding_levels:
                    for sent in doc.sents:
                        sent._.set("vector", sent.vector)

        res = DocBinResponse()

        if attributes:
            res.docbin = DocBin(attributes, docs=docs, store_user_data=True).to_bytes()
        else:
            res.docbin = DocBin(docs=docs, store_user_data=True).to_bytes()

        return res

    async def vector(
        self,
        language: "Language",
        texts: t.Optional[t.List[str]],
        spacy_model: str,
        embedding_levels: t.Optional[t.List["EmbeddingLevel"]],
        embedding_models: t.Optional[t.List["EmbeddingModel"]],
    ) -> "VectorResponse":
        if not all([texts, spacy_model, embedding_levels]):
            raise grpclib.GRPCError(
                grpclib.const.Status.INVALID_ARGUMENT,
                "The arguments 'texts', 'spacy_model' are required.",
            )

        nlp = _load_spacy(language, spacy_model, embedding_models)
        docs = t.cast(t.List[Doc], list(nlp.pipe(texts)))

        res = VectorResponse()

        for doc in docs:
            if EmbeddingLevel.EMBEDDING_LEVEL_DOCUMENT in embedding_levels:
                res.document = Vector(doc.vector.tolist())
            # if EmbeddingLevel.EMBEDDING_LEVEL_TOKENS in embedding_levels:
            #     for token in doc:
            #         token._.set("vector", token.vector)
            # if EmbeddingLevel.EMBEDDING_LEVEL_SENTENCES in embedding_levels:
            #     for token in doc:
            #         token._.set("vector", token.vector)

        return res


async def main(*, host="127.0.0.1", port=50051):
    server = Server([NlpService()])
    # Note: graceful_exit isn't supported in Windows
    with graceful_exit([server]):
        await server.start(host, port)
        print(f"Serving on {host}:{port}")
        await server.wait_closed()

    # await server.start(host, port)
    # await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
