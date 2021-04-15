from __future__ import annotations

import io
import typing as t
from abc import ABC, abstractmethod
from base64 import b85decode
from enum import Enum

import graphene
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


class EmbeddingModel(ABC):
    @abstractmethod
    def __init__(self, model: str) -> None:
        pass

    @abstractmethod
    def vector(self, text: str) -> np.ndarray:
        pass


class TransformerModel(EmbeddingModel):
    def __init__(self, model: str):
        self._model = SentenceTransformer(model)

    def vector(self, text: str):
        embeddings = self._model.encode([text])

        return embeddings[0]


class UseModel(EmbeddingModel):
    def __init__(self, model: str):
        self._model = hub.load(model)

    def vector(self, text: str):
        embeddings = self._model([text])  # type: ignore

        return embeddings[0].numpy()


embedding_map = {
    "use": UseModel,
    "sentence-trf": TransformerModel,
}


def _load_spacy(data: t.Mapping[str, t.Any], **kwargs) -> spacy.Language:
    models = []

    for user_model in data["models"]:
        tmp_model = embedding_map[user_model.embedding_type.value](
            user_model.embedding_model
        )
        models.append(tmp_model)

    model = spacy.load(data["spacy_model"], **kwargs)
    model.add_pipe(
        "concat",
        last=True,
        config={"models": models, "token_vectors": data["token_vectors"]},
    )

    return model


@spacy.Language.factory(
    "concat", default_config={"models": tuple(), "token_vectors": False}
)
class ConcatModel:
    def __init__(self, nlp, name, models, token_vectors):
        self._models = models
        self._token_vectors = token_vectors

    def __call__(self, doc):
        if len(self._models) > 0 and self._token_vectors:
            doc.user_hooks["vector"] = self.concat_token_vectors
            doc.user_span_hooks["vector"] = self.concat_token_vectors
            doc.user_token_hooks["vector"] = self.concat
        elif len(self._models) > 0:
            doc.user_hooks["vector"] = self.concat
            doc.user_span_hooks["vector"] = self.concat
            doc.user_token_hooks["vector"] = self.concat
        elif self._token_vectors:
            doc.user_hooks["vector"] = self.token_vectors
            doc.user_span_hooks["vector"] = self.token_vectors

        return doc

    def token_vectors(self, obj):
        return [t.vector for t in obj]

    def concat_token_vectors(self, obj):
        vecs = [[model.vector(t.text) for t in obj] for model in self._models]
        return [np.concatenate(token_vecs) for token_vecs in vecs]

    def concat(self, obj):
        vecs = [model.vector(obj.text) for model in self._models]
        return np.concatenate(vecs)


def _convert_vector(
    vector: t.Union[np.ndarray, t.List[np.ndarray]], token_vectors: bool
) -> Vector:
    if isinstance(vector, (list, tuple)):  # Doc, Span
        return tuple(tuple(v.tolist()) for v in vector)
    elif token_vectors:  # Token
        return (tuple(vector.tolist()),)

    return tuple(vector.tolist())


class Embedding(graphene.Enum):
    USE = "use"
    SENTENCE_TRF = "sentence-trf"


class EmbedingType(graphene.InputObjectType):
    embedding_type = graphene.Field(Embedding, required=True)
    embedding_model = graphene.String(required=True)


base_args = {
    "language": graphene.String(required=True),
    "spacy_model": graphene.String(required=True),
    "models": graphene.List(EmbedingType, default_value=tuple()),
    "token_vectors": graphene.Boolean(default_value=False),
}


class VectorType(graphene.ObjectType):
    document = graphene.List(graphene.Float)
    sentences = graphene.List(graphene.List(graphene.Float))
    tokens = graphene.List(graphene.List(graphene.Float))

    @staticmethod
    def resolve_document(parent, info):
        return _convert_vector(parent["doc"].vector, parent["token_vectors"])

    @staticmethod
    def resolve_sentences(parent, info):
        return [
            _convert_vector(sentence.vector, parent["token_vectors"])
            for sentence in parent["doc"].sents
        ]

    @staticmethod
    def resolve_tokens(parent, info):
        return [
            _convert_vector(token.vector, parent["token_vectors"])
            for token in parent["doc"]
        ]


class ModelType(graphene.ObjectType):
    docbin = graphene.String(required=True)
    vectors = graphene.List(VectorType)

    @staticmethod
    def resolve_model(parent, info):
        docbin = DocBin(parent.get("attrs", tuple()))

        for doc in parent["docs"]:
            docbin.add(doc)

        return b85decode(docbin.to_bytes()).decode("utf-8")

    @staticmethod
    def resolve_vectors(parent, info):
        return [
            {
                "doc": doc,
                "token_vectors": parent["token_vectors"],
            }
            for doc in parent["docs"]
        ]


class Query(graphene.ObjectType):
    vectors = graphene.List(
        VectorType,
        args={"texts": graphene.List(graphene.String, required=True), **base_args},
    )
    models = graphene.Field(
        ModelType,
        args={
            "texts": graphene.List(graphene.String, required=True),
            "attrs": graphene.List(graphene.String),
            **base_args,
        },
    )

    @staticmethod
    def resolve_vectors(parent, info, **kwargs):
        nlp = _load_spacy(kwargs, exclude=spacy_components)
        return [
            {
                "doc": doc,
                "token_vectors": kwargs["token_vectors"],
            }
            for doc in nlp.pipe(kwargs["texts"])
        ]

    @staticmethod
    def resolve_models(parent, info, **kwargs):
        nlp = _load_spacy(kwargs)
        return {
            "docs": nlp.pipe(kwargs["texts"]),
            "token_vectors": kwargs["token_vectors"],
            "attrs": kwargs["attrs"],
        }


app = FastAPI()
app.add_route("/", GraphQLApp(schema=graphene.Schema(query=Query)))
