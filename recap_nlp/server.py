from __future__ import annotations

import io
import typing as t
from abc import ABC, abstractmethod
from base64 import b85decode
from dataclasses import dataclass
from enum import Enum

import graphene as g
import numpy as np
import spacy
import tensorflow_hub as hub
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc, DocBin, Span, Token  # type: ignore
from starlette.graphql import GraphQLApp

from recap_nlp import common

Vector = t.Union[t.Tuple[float, ...], t.Tuple[t.Tuple[float, ...], ...]]
# TODO: Implement pooling strategies

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


class Pooling(g.Enum):
    MEAN = "mean"
    FIRST = "first"
    LAST = "last"
    MIN = "min"
    MAX = "max"
    SUM = "sum"


pool_map = {
    Pooling.MEAN: np.mean,
    Pooling.FIRST: lambda vecs: vecs[0],
    Pooling.LAST: lambda vecs: vecs[-1],
    Pooling.MIN: np.min,
    Pooling.MAX: np.max,
    Pooling.SUM: np.sum,
}


class EmbeddingModel(ABC):
    @abstractmethod
    def __init__(self, model: str, pooling: Pooling) -> None:
        pass

    @abstractmethod
    def vector(self, text: str) -> np.ndarray:
        pass


class TransformerModel(EmbeddingModel):
    def __init__(self, model: str, pooling: Pooling):
        self.model = SentenceTransformer(model)

    def vector(self, text: str):
        embeddings = self.model.encode([text])

        return embeddings[0]


class UseModel(EmbeddingModel):
    def __init__(self, model: str, pooling: Pooling):
        self.model = hub.load(model)

    def vector(self, text: str):
        embeddings = self.model([text])  # type: ignore

        return embeddings[0].numpy()


class SpacyModel(EmbeddingModel):
    def __init__(self, model: str, pooling: Pooling):
        self.model = spacy.load(model, exclude=spacy_components)
        self.pooling = pooling

    def vector(self, text: str):
        doc = self.model(text)

        if len(doc) > 1 and self.pooling != Pooling.MEAN:
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


def _load_spacy(data: t.Mapping[str, t.Any], **kwargs) -> spacy.Language:
    models = []

    for user_model in data["models"]:
        tmp_model = embedding_map[user_model.embedding_type](
            user_model.embedding_model, user_model.pooling
        )
        models.append(tmp_model)

    model = spacy.load(data["spacy_model"], **kwargs)
    model.add_pipe(
        "concat",
        last=True,
        config={"models": models},
    )

    return model


@dataclass
class _VectorType:
    doc: Doc

    @property
    def document(self):
        return self.doc.vector.tolist()

    # @property
    # def sentences(self):
    #     return [sent.vector.tolist() for sent in self.doc.sents]

    @property
    def tokens(self):
        return [token.vector.tolist() for token in self.doc]


class VectorType(g.ObjectType):
    document = g.List(g.Float)
    # sentences = g.List(g.List(g.Float))
    tokens = g.List(g.List(g.Float))


class Embedding(g.Enum):
    SPACY = "spacy"
    USE = "use"
    SENTENCE_TRF = "sentence-trf"


embedding_map = {
    Embedding.SPACY: SpacyModel,
    Embedding.USE: UseModel,
    Embedding.SENTENCE_TRF: TransformerModel,
}


class EmbedingType(g.InputObjectType):
    embedding_type = g.Field(Embedding, required=True)
    embedding_model = g.String(required=True)
    pooling = g.Field(Pooling, default_value=Pooling.MEAN)


base_args = {
    "texts": g.List(g.String, required=True),
    "language": g.String(required=True),
    "spacy_model": g.String(required=True),
    "models": g.List(EmbedingType, default_value=tuple()),
}


class Query(g.ObjectType):
    vectors = g.List(
        VectorType,
        args={**base_args},
    )
    docbin = g.Field(
        g.Base64,
        args={
            "attrs": g.List(g.String, default_value=None),
            "emb_levels": g.List(g.String, default_value=list()),
            **base_args,
        },
    )

    @staticmethod
    def resolve_vectors(parent, info, **kwargs):
        nlp = _load_spacy(kwargs, exclude=spacy_components)
        return [_VectorType(doc) for doc in nlp.pipe(kwargs["texts"])]

    @staticmethod
    def resolve_docbin(parent, info, **kwargs):
        nlp = _load_spacy(kwargs)
        docs = t.cast(t.List[Doc], list(nlp.pipe(kwargs["texts"])))
        emb_levels = kwargs["emb_levels"]
        attrs = kwargs["attrs"]

        for doc in docs:
            if "document" in emb_levels:
                doc._.set("vector", doc.vector)
            if "tokens" in emb_levels:
                for token in doc:
                    token._.set("vector", token.vector)

        if attrs is None:
            return DocBin(docs=docs, store_user_data=True).to_bytes()

        return DocBin(attrs, docs=docs, store_user_data=True).to_bytes()


app = FastAPI()
app.add_route("/graphql", GraphQLApp(schema=g.Schema(query=Query)))
