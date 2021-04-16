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


class SpacyModel(EmbeddingModel):
    def __init__(self, model: str):
        self._model = spacy.load(model, exclude=spacy_components)

    def vector(self, text: str):
        return self._model(text).vector


embedding_map = {
    "spacy": SpacyModel,
    "use": UseModel,
    "sentence-trf": TransformerModel,
}


class ConcatModel:
    def __init__(self, models: t.Tuple[EmbeddingModel]):
        self._models = models

    def vector(self, obj):
        if len(self._models) > 0:
            vecs = [model.vector(obj.text) for model in self._models]
            return np.concatenate(vecs)

        return obj.vector

    def vectors(self, obj):
        if isinstance(obj, Token):
            obj = [obj]

        if len(self._models) > 0:
            vecs = [[model.vector(t.text) for t in obj] for model in self._models]
            return [np.concatenate(token_vecs) for token_vecs in vecs]

        return [t.vector for t in obj]


@spacy.Language.factory("concat")
class ConcatComponent:
    def __init__(self, nlp, name, models):
        self._models = models

    def __call__(self, doc):
        doc._.set("concat_model", ConcatModel(self._models))

        return doc


Doc.set_extension("concat_model", default=None)  # TODO: Might cause exceptions
Doc.set_extension(
    "vectors",
    getter=lambda doc: doc._.concat_model.vectors(doc),
)
Doc.set_extension(
    "vector",
    getter=lambda doc: doc._.concat_model.vector(doc),
)
Span.set_extension(
    "vectors",
    getter=lambda span: span.doc._.concat_model.vectors(span),
)
Span.set_extension(
    "vector",
    getter=lambda span: span.doc._.concat_model.vector(span),
)
Token.set_extension(
    "vectors",
    getter=lambda token: token.doc._.concat_model.vectors(token),
)
Token.set_extension(
    "vector",
    getter=lambda token: token.doc._.concat_model.vector(token),
)


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
        config={"models": models},
    )

    return model


@dataclass
class _VectorsType:
    _doc: Doc

    @property
    def document(self):
        return [vec.tolist() for vec in self._doc._.vectors]

    @property
    def sentences(self):
        return [[vec.tolist() for vec in sent._.vectors] for sent in self._doc.sents]

    @property
    def tokens(self):
        return [[v.tolist() for v in token._.vectors] for token in self._doc]


@dataclass
class _VectorType:
    _doc: Doc

    @property
    def document(self):
        return self._doc._.vector.tolist()

    @property
    def sentences(self):
        return [sent._.vector.tolist() for sent in self._doc.sents]

    @property
    def tokens(self):
        return [token._.vector.tolist() for token in self._doc]


class VectorType(g.ObjectType):
    document = g.List(g.Float)
    sentences = g.List(g.List(g.Float))
    tokens = g.List(g.List(g.Float))


class VectorsType(g.ObjectType):
    document = g.List(g.List(g.Float))
    sentences = g.List(g.List(g.List(g.Float)))
    tokens = g.List(g.List(g.List(g.Float)))


@dataclass
class _ModelType:
    _docs: t.List[Doc]
    _attrs: t.Optional[t.List[str]]

    @property
    def docbin(self) -> bytes:
        models = []

        for doc in self._docs:
            models.append(doc._.get("concat_model"))
            doc._.set("concat_model", None)

        if self._attrs is None:
            out = DocBin(docs=self._docs).to_bytes()
        else:
            out = DocBin(self._attrs, docs=self._docs).to_bytes()

        for model, doc in zip(models, self._docs):
            doc._.set("concat_model", model)

        return out

    @property
    def vectors(self) -> t.List[_VectorsType]:
        return [_VectorsType(doc) for doc in self._docs]

    @property
    def vector(self) -> t.List[_VectorType]:
        return [_VectorType(doc) for doc in self._docs]


class ModelType(g.ObjectType):
    docbin = g.Base64(required=True)
    vectors = g.List(VectorsType)
    vector = g.List(VectorType)


class Embedding(g.Enum):
    SPACY = "spacy"
    USE = "use"
    SENTENCE_TRF = "sentence-trf"


class EmbedingType(g.InputObjectType):
    embedding_type = g.Field(Embedding, required=True)
    embedding_model = g.String(required=True)


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
    models = g.Field(
        ModelType,
        args={
            "attrs": g.List(g.String, default_value=None),
            **base_args,
        },
    )

    @staticmethod
    def resolve_vectors(parent, info, **kwargs):
        nlp = _load_spacy(kwargs, exclude=spacy_components)
        return [_VectorType(doc) for doc in nlp.pipe(kwargs["texts"])]

    @staticmethod
    def resolve_models(parent, info, **kwargs):
        nlp = _load_spacy(kwargs)
        return _ModelType(list(nlp.pipe(kwargs["texts"])), kwargs["attrs"])


app = FastAPI()
app.add_route("/graphql", GraphQLApp(schema=g.Schema(query=Query)))
