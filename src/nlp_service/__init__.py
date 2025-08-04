import logging
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, override

import cbrkit
from arg_services.nlp.v1 import nlp_pb2
from arg_services.nlp.v1 import nlp_pb2 as model
from arg_services.nlp.v1 import nlp_pb2_grpc as rpc
from spacy.tokens import Doc

from . import client

__all__ = [
    "Nlp",
    "PipeSelection",
    "EmbedFunc",
    "ScoreFunc",
    "model",
    "rpc",
    "client",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


type EmbedFunc = cbrkit.typing.BatchConversionFunc[str, cbrkit.typing.NumpyArray]
type ScoreFunc = cbrkit.typing.AnySimFunc[cbrkit.typing.NumpyArray, float]


class PipeSelection(TypedDict, total=False):
    enable: Sequence[str]
    disable: Sequence[str]


embed_funcs: Mapping[nlp_pb2.EmbeddingType, Callable[[str], EmbedFunc]] = {
    nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SPACY: cbrkit.sim.embed.spacy,
    nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS: cbrkit.sim.embed.sentence_transformers,
    nlp_pb2.EmbeddingType.EMBEDDING_TYPE_OPENAI: cbrkit.sim.embed.openai,
    nlp_pb2.EmbeddingType.EMBEDDING_TYPE_COHERE: cbrkit.sim.embed.cohere,
    nlp_pb2.EmbeddingType.EMBEDDING_TYPE_VOYAGEAI: cbrkit.sim.embed.voyageai,
    nlp_pb2.EmbeddingType.EMBEDDING_TYPE_OLLAMA: cbrkit.sim.embed.ollama,
}
score_funcs: Mapping[nlp_pb2.SimilarityMethod, Callable[[], ScoreFunc]] = {
    nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_UNSPECIFIED: cbrkit.sim.embed.cosine,
    nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE: cbrkit.sim.embed.cosine,
    nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_ANGULAR: cbrkit.sim.embed.angular,
    nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_EUCLIDEAN: cbrkit.sim.embed.euclidean,
    nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_MANHATTAN: cbrkit.sim.embed.manhattan,
    nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_DOT: cbrkit.sim.embed.dot,
}


with cbrkit.helpers.optional_dependencies():
    import tensorflow as tf
    import tensorflow_hub as hub

    @dataclass(slots=True)
    class tf_hub_embedder(
        cbrkit.typing.BatchConversionFunc[str, cbrkit.typing.NumpyArray]
    ):
        model: Callable[[Sequence[str]], tf.Tensor]

        def __init__(self, model: str):
            self.model = hub.load(model)  # pyright: ignore

        @override
        def __call__(self, texts: Sequence[str]) -> Sequence[cbrkit.typing.NumpyArray]:
            if not texts:
                return []

            return self.model(texts).numpy().tolist()  # pyright: ignore

    embed_funcs[nlp_pb2.EmbeddingType.EMBEDDING_TYPE_TENSORFLOW_HUB] = tf_hub_embedder  # pyright: ignore


@dataclass(slots=True)
class Nlp:
    cache_path: Path | None = None
    provider_cache: bool = True
    provider_init: Mapping[nlp_pb2.EmbeddingType, Callable[[str], EmbedFunc]] | None = (
        field(default_factory=dict)
    )
    retrieval_init: Callable[
        [cbrkit.typing.AnySimFunc[str, float]],
        cbrkit.typing.RetrieverFunc[Any, str, float],
    ] = cbrkit.retrieval.build
    provider_store: MutableMapping[tuple[nlp_pb2.EmbeddingType, str], EmbedFunc] = (
        field(default_factory=dict, init=False, repr=False)
    )

    def embed_provider(
        self, model_type: nlp_pb2.EmbeddingType, model_name: str
    ) -> EmbedFunc:
        key = (model_type, model_name)

        if key in self.provider_store:
            return self.provider_store[key]

        if self.provider_init is None:
            embed_func = None
        else:
            init_func = self.provider_init.get(
                model_type,
                embed_funcs[model_type],
            )
            embed_func = init_func(model_name)

        table_prefix = (
            nlp_pb2.EmbeddingType.Name(model_type)
            .removeprefix("EMBEDDING_TYPE_")
            .lower()
        )

        func = cbrkit.sim.embed.cache(
            embed_func,
            path=self.cache_path,
            table=f"{table_prefix}_{model_name}",
        )

        if self.provider_cache:
            self.provider_store[key] = func

        return func

    def embed_providers(self, config: nlp_pb2.NlpConfig) -> list[EmbedFunc]:
        if not config.embedding_models:
            raise ValueError("No embedding models provided.")

        return [
            self.embed_provider(x.model_type, x.model_name)
            for x in config.embedding_models
        ]

    def embed_func(self, config: nlp_pb2.NlpConfig) -> EmbedFunc:
        embed_funcs = self.embed_providers(config)

        if len(embed_funcs) == 1:
            return embed_funcs[0]

        return cbrkit.sim.embed.concat(embed_funcs)

    def score_func(
        self, config: nlp_pb2.NlpConfig
    ) -> cbrkit.typing.AnySimFunc[cbrkit.typing.NumpyArray, float]:
        return score_funcs[config.similarity_method]()

    def sim_func(
        self, config: nlp_pb2.NlpConfig
    ) -> cbrkit.typing.BatchSimFunc[str, float]:
        return cbrkit.sim.embed.build(self.embed_func(config), self.score_func(config))

    def retrieval_func(
        self, config: nlp_pb2.NlpConfig
    ) -> cbrkit.typing.RetrieverFunc[Any, str, float]:
        return self.retrieval_init(self.sim_func(config))

    def pipe_docs(
        self,
        config: nlp_pb2.NlpConfig,
        batches: Sequence[str],
        pipes_selection: PipeSelection | None = None,
        vectorize: bool = False,
    ) -> Sequence[Doc]:
        if not pipes_selection:
            pipes_selection = {"disable": []}

        if "enable" in pipes_selection:
            pipes_selection["enable"] = tuple(pipes_selection["enable"])

        nlp = cbrkit.sim.embed.load_spacy(config.spacy_model)

        with nlp.select_pipes(**pipes_selection):
            docs = list(nlp.pipe(batches))

        if vectorize:
            embed_func = self.embed_func(config)
            vecs = embed_func([doc.text for doc in docs])

            for doc, vector in zip(docs, vecs, strict=True):
                doc._.set("vector", vector)

        return docs
