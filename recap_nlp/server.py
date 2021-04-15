from __future__ import annotations

import io
import typing as t
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import spacy
import tensorflow_hub as hub
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc, DocBin, Span, Token  # type: ignore

from recap_nlp import common

Vector = t.Union[t.Tuple[float, ...], t.Tuple[t.Tuple[float, ...], ...]]
app = FastAPI()


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
    common.Embedding.USE: UseModel,
    common.Embedding.SENTENCE_TRF: TransformerModel,
}


# def _verify_query(q: common.BaseQuery) -> t.Optional[HTTPException]:
#     err = []

#     if len(q.models) == 0:
#         err.append("You did not specify any models. At least one is required.")

#     if len(q.models) > 1 and not q.concat_model:
#         err.append(
#             "You specified multiple embeddings, but no spacy model that is responsible for concatenation."
#         )

#     for i, model in enumerate(q.models):
#         if model.embedding_type and not model.embedding_model:
#             err.append(
#                 f"Model '{i}' specifies embedding_type '{model.embedding_type.value}', but no embedding_model."
#             )
#         elif model.embedding_model and not model.embedding_type:
#             err.append(
#                 f"Model '{i}' specifies embedding_model '{model.embedding_model}', but no embedding_type."
#             )

#     if err:
#         return HTTPException(status_code=422, detail=" ".join(err))

#     return None


def _load_spacy(q: common.BaseQuery, **kwargs) -> spacy.Language:
    models = []

    for user_model in q.models:
        tmp_model = embedding_map[user_model.embedding_type](user_model.embedding_model)
        models.append(tmp_model)

    model = spacy.load(q.spacy_model, **kwargs)
    model.add_pipe(
        "concat",
        last=True,
        config={"models": models, "token_vectors": q.token_vectors},
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


_vector_cache = {}


def _convert_vector(
    vector: t.Union[np.ndarray, t.List[np.ndarray]], token_vectors: bool
) -> Vector:
    if isinstance(vector, (list, tuple)):  # Doc, Span
        return tuple(tuple(v.tolist()) for v in vector)
    elif token_vectors:  # Token
        return (tuple(vector.tolist()),)

    return tuple(vector.tolist())


def _vector(text: str, q: common.BaseQuery, nlp: spacy.Language) -> Vector:
    if text not in _vector_cache:
        _vector_cache[text] = _convert_vector(nlp(text).vector, q.token_vectors)

    return _vector_cache[text]


def _vectors(
    texts: t.Iterable[str], q: common.BaseQuery, nlp: spacy.Language
) -> t.Tuple[Vector, ...]:
    unknown_texts = [text for text in texts if text not in _vector_cache]

    if unknown_texts:
        docs = nlp.pipe(unknown_texts)

        for text, doc in zip(unknown_texts, docs):
            _vector_cache[text] = _convert_vector(doc.vector)  # type: ignore

    return tuple(_vector_cache[text] for text in texts)


class SingleQuery(common.BaseQuery):
    text: str


@app.post("/vector")
def vector(q: SingleQuery):
    nlp = _load_spacy(q, exclude=spacy_components)
    return _vector(q.text, q, nlp)


class MultiQuery(common.BaseQuery):
    texts: t.Tuple[str, ...]


@app.post("/vectors")
def vectors(
    q: MultiQuery,
) -> t.Tuple[Vector, ...]:
    nlp = _load_spacy(q, exclude=spacy_components)
    return _vectors(q.texts, q, nlp)


@app.post("/model")
def model(q: SingleQuery):
    nlp = _load_spacy(q)
    docbin = DocBin()
    docbin.add(nlp(q.text))

    return StreamingResponse(io.BytesIO(docbin.to_bytes()))


@app.post("/models")
def models(q: MultiQuery):
    nlp = _load_spacy(q)
    docbin = DocBin()

    for doc in nlp.pipe(q.texts):
        docbin.add(doc)

    return StreamingResponse(io.BytesIO(docbin.to_bytes()))


@app.get("/")
def ready() -> bool:
    return True
