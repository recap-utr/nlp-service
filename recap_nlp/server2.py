from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from base64 import b85decode
from dataclasses import dataclass
from enum import Enum

import numpy as np
import spacy
import strawberry
import tensorflow_hub as hub
from sanic import Sanic
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc, DocBin, Span, Token  # type: ignore
from strawberry.sanic.views import GraphQLView

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


@strawberry.enum
class Embedding(Enum):
    USE = "use"
    SENTENCE_TRF = "sentence-trf"


@strawberry.input
class EmbedingType:
    embedding_type: Embedding
    embedding_model: str


@dataclass
class _VectorType:
    doc: Doc
    token_vectors: bool

    @property
    def document(self) -> Vector:
        return _convert_vector(self.doc.vector, self.token_vectors)

    @property
    def sentences(self) -> t.List[Vector]:
        return [
            _convert_vector(sent.vector, self.token_vectors) for sent in self.doc.sents
        ]

    @property
    def tokens(self) -> t.List[Vector]:
        return [
            _convert_vector(token.vector, self.token_vectors)
            for token in self.doc.sents
        ]


@strawberry.type
class VectorType:
    document: t.List[float]
    sentences: t.List[t.List[float]]
    tokens: t.List[t.List[float]]


@dataclass
class _ModelType:
    docs: t.List[Doc]
    attrs: t.List[str]
    token_vectors: bool


@strawberry.type
class ModelType:
    @strawberry.field
    def documents(self) -> str:
        store = DocBin(self.attrs)

        for doc in self.docs:
            store.add(doc)

        return b85decode(store.to_bytes()).decode("utf-8")

    @strawberry.field
    def vectors(self) -> t.Optional[t.List[VectorType]]:
        return [_VectorType(doc, self.token_vectors) for doc in self.docs]


@strawberry.type
class Query:
    @strawberry.field
    def nlp(
        self,
        texts: t.List[str],
        language: str,
        spacy_model: str,
        models: t.Optional[t.List[EmbedingType]],
        attrs: t.Optional[t.List[str]],
        token_vectors: bool = False,
    ) -> ModelType:
        spacy_models = []

        for user_model in models:
            tmp_model = embedding_map[user_model.embedding_type.value](
                user_model.embedding_model
            )
            spacy_models.append(tmp_model)

        nlp = spacy.load(spacy_model)
        nlp.add_pipe(
            "concat",
            last=True,
            config={"models": spacy_models, "token_vectors": token_vectors},
        )

        return _ModelType(list(nlp.pipe(texts)), attrs, token_vectors)


schema = strawberry.Schema(Query)
# app = Sanic(__name__)
# app.add_route(
#     GraphQLView.as_view(schema=schema, graphiql=True),
#     "/graphql",
# )

# if __name__ == "__main__":
#     app.run()
