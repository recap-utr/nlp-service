from __future__ import annotations

import itertools
import typing as t
from abc import ABC, abstractmethod

import arg_services_helper
import grpc
import numpy as np
import scipy.stats
import spacy
import tensorflow_hub as hub
import typer
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc

# from arg_services.topic_modeling.v1 import topic_modeling_pb2, topic_modeling_pb2_grpc
# from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc, DocBin, Span, Token  # type: ignore

from nlp_service.similarity import spacy_mapping

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
custom_components = ("concat", "similarity")
# custom_components = []

# TODO: Extract spacy-specific code into its own file.
# Benefit: We could refactor the code s.t. the spacy `nlp` object can be
# imported directly in python applications (eliminating the need to use a server).


class EmbeddingBase(ABC):
    @abstractmethod
    def __init__(self, model: str, pooling: nlp_pb2.Pooling.V) -> None:
        pass

    @abstractmethod
    def vector(self, text: str) -> np.ndarray:
        pass


class TransformerModel(EmbeddingBase):
    def __init__(self, model: str, pooling: nlp_pb2.Pooling.V):
        self.model = SentenceTransformer(model)

    def vector(self, text: str):
        embeddings = self.model.encode([text])

        return embeddings[0]


class UseModel(EmbeddingBase):
    def __init__(self, model: str, pooling: nlp_pb2.Pooling.V):
        self.model = hub.load(model)

    def vector(self, text: str):
        embeddings = self.model([text])  # type: ignore

        return embeddings[0].numpy()


class SpacyModel(EmbeddingBase):
    def __init__(self, model: str, pooling: nlp_pb2.Pooling.V):
        self.model = spacy.load(model)
        self.pooling = pooling

    def vector(self, text: str):
        with self.model.select_pipes(enable=["senter"]):
            doc = self.model(text)

        if len(doc) > 1 and self.pooling != nlp_pb2.POOLING_MEAN:
            return pool_map[self.pooling]([t.vector for t in doc])

        return doc.vector


@spacy.Language.factory("concat")
class ConcatFactory:
    def __init__(self, nlp, name, models):
        self.models = []

        for model in models:
            if model not in embedding_cache:
                embedding_cache[model] = embedding_map[model[0]](*model[1:])

            self.models.append(embedding_cache[model])

    def __call__(self, doc):
        if len(self.models) > 0:
            doc.user_hooks["vector"] = self.vector
            doc.user_span_hooks["vector"] = self.vector
            doc.user_token_hooks["vector"] = self.vector

        return doc

    def vector(self, obj):
        vecs = [model.vector(obj.text) for model in self.models]
        return np.concatenate(vecs)


@spacy.Language.factory("similarity")
class SimilarityFactory:
    def __init__(self, nlp, name, method):
        if method:
            self.func = spacy_mapping[method]

    def __call__(self, doc):
        if self.func:
            doc.user_hooks["similarity"] = self.func
            doc.user_span_hooks["similarity"] = self.func
            doc.user_token_hooks["similarity"] = self.func

        return doc


# Doc.set_extension("vector", default=None)
# Span.set_extension("vector", default=None)
# Token.set_extension("vector", default=None)

spacy_cache = {}
embedding_cache = {}


def _hash_embedding_model(model: nlp_pb2.EmbeddingModel):
    return (model.model_type, model.model_name, model.pooling)


def _load_spacy(config: nlp_pb2.NlpConfig) -> spacy.Language:
    models = tuple(_hash_embedding_model(model) for model in config.embedding_models)
    key = (
        config.language,
        config.spacy_model,
        models,
    )

    if key not in spacy_cache:
        nlp = (
            spacy.load(config.spacy_model)
            if config.spacy_model
            else spacy.blank(config.language)
        )

        if models:
            nlp.add_pipe(
                "concat",
                last=True,
                config={"models": models},
            )

        if config.similarity_method not in [
            nlp_pb2.SIMILARITY_METHOD_COSINE,
            nlp_pb2.SIMILARITY_METHOD_UNSPECIFIED,
        ]:
            nlp.add_pipe(
                "similarity", last=True, config={"method": config.similarity_method}
            )

        spacy_cache[key] = nlp

    return spacy_cache[key]


pool_map = {
    nlp_pb2.POOLING_MEAN: np.mean,
    nlp_pb2.POOLING_FIRST: lambda vecs: vecs[0],
    nlp_pb2.POOLING_LAST: lambda vecs: vecs[-1],
    nlp_pb2.POOLING_MIN: np.min,
    nlp_pb2.POOLING_MAX: np.max,
    nlp_pb2.POOLING_SUM: np.sum,
    nlp_pb2.POOLING_MEDIAN: np.median,
    nlp_pb2.POOLING_GMEAN: scipy.stats.gmean,
    nlp_pb2.POOLING_HMEAN: scipy.stats.hmean,
}

embedding_map = {
    nlp_pb2.EMBEDDING_TYPE_SPACY: SpacyModel,
    nlp_pb2.EMBEDDING_TYPE_USE: UseModel,
    nlp_pb2.EMBEDDING_TYPE_SBERT: TransformerModel,
}


class NlpService(nlp_pb2_grpc.NlpServiceServicer):
    def DocBin(
        self,
        req: nlp_pb2.DocBinRequest,
        ctx: grpc.ServicerContext,
    ) -> nlp_pb2.DocBinResponse:
        arg_services_helper.require_all(["config.language"], req, ctx)
        arg_services_helper.forbid_all(["attributes", "no_attributes"], req, ctx)
        arg_services_helper.forbid_all(["pipes", "no_pipes"], req, ctx)

        for model in req.config.embedding_models:
            arg_services_helper.require_all(
                ["model_type", "model_name", "pooling"],
                model,
                ctx,
                parent="embedding_models",
            )

        res = nlp_pb2.DocBinResponse()

        nlp = _load_spacy(req.config)
        nlp_args = {"disable": []}  # if empty, spacy will raise an exception

        if req.no_attributes or req.no_pipes:
            nlp_args = {"enable": custom_components}
        elif req.pipes:
            nlp_args = {"enable": itertools.chain(custom_components, req.pipes)}

        with nlp.select_pipes(**nlp_args):
            docs = t.cast(t.List[Doc], list(nlp.pipe(req.texts)))

        if levels := req.embedding_levels:
            for doc in docs:
                if nlp_pb2.EMBEDDING_LEVEL_DOCUMENT in levels:
                    doc._.set("vector", doc.vector)
                if nlp_pb2.EMBEDDING_LEVEL_TOKENS in levels:
                    for token in doc:
                        token._.set("vector", token.vector)
                if nlp_pb2.EMBEDDING_LEVEL_SENTENCES in levels:
                    for sent in doc.sents:
                        sent._.set("vector", sent.vector)

        if attrs := req.attributes:
            res.docbin = DocBin(attrs, docs=docs, store_user_data=True).to_bytes()
        elif req.no_attributes:
            res.docbin = DocBin([], docs=docs, store_user_data=True).to_bytes()
        else:
            res.docbin = DocBin(docs=docs, store_user_data=True).to_bytes()

        return res

    def Vectors(
        self,
        req: nlp_pb2.VectorsRequest,
        ctx: grpc.ServicerContext,
    ) -> nlp_pb2.VectorsResponse:
        res = nlp_pb2.VectorsResponse()

        arg_services_helper.require_all(
            ["config.language", "embedding_levels"], req, ctx
        )
        arg_services_helper.require_all_repeated(
            "config.embedding_models",
            ["model_type", "model_name", "pooling"],
            req,
            ctx,
        )

        nlp = _load_spacy(req.config)

        with nlp.select_pipes(enable=custom_components):
            docs = t.cast(t.List[Doc], list(nlp.pipe(req.texts)))

        for doc in docs:
            vector_res = nlp_pb2.VectorResponse()

            for level in req.embedding_levels:
                if level == nlp_pb2.EMBEDDING_LEVEL_DOCUMENT:
                    vector_res.document.vector.extend(doc.vector.tolist())
                elif level == nlp_pb2.EMBEDDING_LEVEL_TOKENS:
                    for token in doc:
                        vector_res.tokens.append(
                            nlp_pb2.Vector(vector=token.vector.tolist())
                        )
                elif level == nlp_pb2.EMBEDDING_LEVEL_SENTENCES:
                    for sent in doc.sents:
                        vector_res.sentences.append(
                            nlp_pb2.Vector(vector=sent.vector.tolist())
                        )

            res.vectors.append(vector_res)

        return res

    def Similarities(
        self,
        req: nlp_pb2.SimilaritiesRequest,
        ctx: grpc.ServicerContext,
    ) -> nlp_pb2.SimilaritiesResponse:
        res = nlp_pb2.SimilaritiesResponse()

        arg_services_helper.require_all(
            ["config.language", "config.similarity_method"], req, ctx
        )
        arg_services_helper.require_all_repeated(
            "config.embedding_models",
            ["model_type", "model_name", "pooling"],
            req,
            ctx,
        )
        arg_services_helper.require_all_repeated(
            "text_tuples",
            ["text1", "text2"],
            req,
            ctx,
        )

        nlp = _load_spacy(req.config)
        text_tuples = [(x.text1, x.text2) for x in req.text_tuples]
        texts1, texts2 = zip(*text_tuples)

        with nlp.select_pipes(enable=custom_components):
            docs1 = t.cast(t.List[Doc], list(nlp.pipe(texts1)))
            docs2 = t.cast(t.List[Doc], list(nlp.pipe(texts2)))

        res.similarities.extend(
            doc1.similarity(doc2) for doc1, doc2 in zip(docs1, docs2)
        )

        return res


# [
#     "",
#     "IS_ALPHA",
#     "IS_ASCII",
#     "IS_DIGIT",
#     "IS_LOWER",
#     "IS_PUNCT",
#     "IS_SPACE",
#     "IS_TITLE",
#     "IS_UPPER",
#     "LIKE_URL",
#     "LIKE_NUM",
#     "LIKE_EMAIL",
#     "IS_STOP",
#     "IS_OOV_DEPRECATED",
#     "IS_BRACKET",
#     "IS_QUOTE",
#     "IS_LEFT_PUNCT",
#     "IS_RIGHT_PUNCT",
#     "IS_CURRENCY",
#     "ID",
#     "ORTH",
#     "LOWER",
#     "NORM",
#     "SHAPE",
#     "PREFIX",
#     "SUFFIX",
#     "LENGTH",
#     "CLUSTER",
#     "LEMMA",
#     "POS",
#     "TAG",
#     "DEP",
#     "ENT_IOB",
#     "ENT_TYPE",
#     "ENT_ID",
#     "ENT_KB_ID",
#     "HEAD",
#     "SENT_START",
#     "SENT_END",
#     "SPACY",
#     "PROB",
#     "LANG",
#     "MORPH",
#     "IDX",
# ]
