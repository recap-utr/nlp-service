from __future__ import annotations

import itertools
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import arg_services_helper
import grpc
import numpy as np
import scipy.stats
import spacy
import tensorflow_hub as hub
import typer
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from sentence_transformers import SentenceTransformer
from spacy.language import Language
from spacy.tokens import Doc, DocBin

from nlp_service.similarity import spacy_mapping as spacy_similarity_map

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
custom_components = ("embedding_models", "similarity_method")
# custom_components = []

# TODO: Extract spacy-specific code into its own file.
# Benefit: We could refactor the code s.t. the spacy `nlp` object can be
# imported directly in python applications (eliminating the need to use a server).


@dataclass(frozen=True, eq=True)
class EmbeddingModel:
    model_type: int
    model_name: str
    pooling_type: t.Optional[int]
    pmean: t.Optional[float]

    @classmethod
    def from_protobuf(cls, pb: nlp_pb2.EmbeddingModel) -> EmbeddingModel:
        return cls(pb.model_type, pb.model_name, pb.pooling_type, pb.pmean)


class ModelBase(ABC):
    @abstractmethod
    def __init__(self, model: EmbeddingModel) -> None:
        pass

    @abstractmethod
    def vector(self, text: str) -> np.ndarray:
        pass


class SentenceTransformersModel(ModelBase):
    def __init__(self, model: EmbeddingModel):
        self.model = SentenceTransformer(model.model_name)

    def vector(self, text: str):
        embeddings = self.model.encode([text])

        return embeddings[0]


class TensorflowHubModel(ModelBase):
    def __init__(self, model: EmbeddingModel):
        self.model = hub.load(model.model_name)

    def vector(self, text: str):
        embeddings = self.model([text])  # type: ignore

        return embeddings[0].numpy()


def pmean(vectors: t.Any, p: float) -> np.ndarray:
    # vectors: t.Iterable[np.ndarray]
    return np.power(
        np.mean(np.power(np.array(vectors, dtype=complex), p), axis=0), 1 / p
    ).real


class SpacyModel(ModelBase):
    def __init__(self, model: EmbeddingModel):
        self.model = spacy.load(model.model_name)
        self.pooling_type = t.cast(nlp_pb2.Pooling.V, model.pooling_type)
        self.pmean = model.pmean

    def vector(self, text: str):
        with self.model.select_pipes(enable=["senter"]):
            doc = self.model(text)

        if len(doc) > 1:
            if self.pooling_type and self.pooling_type != nlp_pb2.Pooling.POOLING_MEAN:
                return pool_map[self.pooling_type](t.vector for t in doc)
            elif self.pmean:
                return pmean((t.vector for t in doc), self.pmean)

        return doc.vector


@Language.factory("embedding_models")
class EmbeddingsFactory:
    def __init__(self, nlp, name, models):
        self.models = []

        for model in models:
            if model not in embedding_cache:
                embedding_cache[model] = embedding_map[model.model_name](model)

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


@Language.factory("similarity_method")
class SimilarityFactory:
    def __init__(self, nlp, name, method):
        if method:
            self.func = spacy_similarity_map[method]

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


def _load_spacy(config: nlp_pb2.NlpConfig) -> Language:
    models = tuple(
        EmbeddingModel.from_protobuf(model) for model in config.embedding_models
    )
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
                "embedding_models",
                last=True,
                config={"models": models},
            )

        if config.similarity_method not in [
            nlp_pb2.SIMILARITY_METHOD_COSINE,
            nlp_pb2.SIMILARITY_METHOD_UNSPECIFIED,
        ]:
            nlp.add_pipe(
                "similarity_method",
                last=True,
                config={"method": config.similarity_method},
            )

        spacy_cache[key] = nlp

    return spacy_cache[key]


pool_map = {
    nlp_pb2.Pooling.POOLING_MEAN: np.mean,
    nlp_pb2.Pooling.POOLING_FIRST: lambda vecs: vecs[0],
    nlp_pb2.Pooling.POOLING_LAST: lambda vecs: vecs[-1],
    nlp_pb2.Pooling.POOLING_MIN: np.min,
    nlp_pb2.Pooling.POOLING_MAX: np.max,
    nlp_pb2.Pooling.POOLING_SUM: np.sum,
    nlp_pb2.Pooling.POOLING_MEDIAN: np.median,
    nlp_pb2.Pooling.POOLING_GMEAN: scipy.stats.gmean,
    nlp_pb2.Pooling.POOLING_HMEAN: scipy.stats.hmean,
}

embedding_map = {
    nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SPACY: SpacyModel,
    nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS: SentenceTransformersModel,
    nlp_pb2.EmbeddingType.EMBEDDING_TYPE_TENSORFLOW_HUB: TensorflowHubModel,
    # nlp_pb2.EmbeddingType.EMBEDDING_TYPE_TRANSFORMERS: TODO,
}


class NlpService(nlp_pb2_grpc.NlpServiceServicer):
    def DocBin(
        self,
        req: nlp_pb2.DocBinRequest,
        ctx: grpc.ServicerContext,
    ) -> nlp_pb2.DocBinResponse:
        arg_services_helper.require_all(["config.language"], req, ctx)

        for model in req.config.embedding_models:
            arg_services_helper.require_all(
                ["model_type", "model_name", "pooling"],
                model,
                ctx,
                parent="embedding_models",
            )

        res = nlp_pb2.DocBinResponse()

        nlp = _load_spacy(req.config)
        pipes_selection = {"disable": []}  # if empty, spacy will raise an exception

        if not req.attributes:
            pipes_selection = {"enable": custom_components}
        elif req.WhichOneof("pipes") == nlp_pb2:
            pipes_selection = {"enable": list(req.enabled_pipes.values)}
        elif req.WhichOneof("pipes") == "disabled_pipes":
            pipes_selection = {"disable": list(req.disabled_pipes.values)}

        with nlp.select_pipes(**pipes_selection):
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

        if not req.attributes:
            res.docbin = DocBin(docs=docs, store_user_data=True).to_bytes()
        else:
            res.docbin = DocBin(
                req.attributes.values, docs=docs, store_user_data=True
            ).to_bytes()

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


app = typer.Typer()


def add_services(server: grpc.Server):
    """Add the services to the grpc server."""

    nlp_pb2_grpc.add_NlpServiceServicer_to_server(NlpService(), server)
    # topic_modeling_pb2_grpc.add_TopicModelingServiceServicer_to_server(
    #     TopicModelingService(), server
    # )


@app.command()
def main(address: str):
    """Main entry point for the server."""

    arg_services_helper.serve(
        address,
        add_services,
        [arg_services_helper.full_service_name(nlp_pb2, "NlpService")],
    )


if __name__ == "__main__":
    app()
