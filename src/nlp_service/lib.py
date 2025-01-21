import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import cbrkit
import spacy
from arg_services.nlp.v1 import nlp_pb2
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc

logger = logging.getLogger(__name__)

type EmbedFunc = cbrkit.typing.BatchConversionFunc[str, cbrkit.typing.NumpyArray]


class PipeSelection(TypedDict, total=False):
    enable: Sequence[str]
    disable: Sequence[str]


@functools.cache
def embed_provider(
    model_type: nlp_pb2.EmbeddingType,
    model_name: str,
    cache_dir: Path | None,
    init_func: Callable[[str], EmbedFunc] | None,
) -> EmbedFunc:
    if init_func is not None:
        embed_func = init_func(model_name)
    else:
        match model_type:
            case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SPACY:
                embed_func = cbrkit.sim.embed.spacy(model_name)
            case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS:
                embed_func = cbrkit.sim.embed.sentence_transformers(model_name)
            case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_OPENAI:
                embed_func = cbrkit.sim.embed.openai(model_name)
            case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_COHERE:
                embed_func = cbrkit.sim.embed.cohere(model_name)
            case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_VOYAGEAI:
                embed_func = cbrkit.sim.embed.voyageai(model_name)
            case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_OLLAMA:
                embed_func = cbrkit.sim.embed.ollama(model_name)
            case _:
                raise ValueError("Unknown embedding model type.")

    if cache_dir:
        model_type_label = (
            nlp_pb2.EmbeddingType.Name(model_type)
            .removeprefix("EMBEDDING_TYPE_")
            .lower()
        )
        cache_filename = f"{model_type_label}_{model_name}.npz"

        return cbrkit.sim.embed.cache(embed_func, cache_dir / cache_filename)

    return cbrkit.sim.embed.cache(embed_func)


@dataclass(slots=True, frozen=True)
class Nlp:
    config: nlp_pb2.NlpConfig
    cache_dir: Path | None = None
    provider_init: Mapping[nlp_pb2.EmbeddingType, Callable[[str], EmbedFunc]] = field(
        default_factory=dict
    )
    retrieval_init: Callable[
        [cbrkit.typing.AnySimFunc[str, float]],
        cbrkit.typing.RetrieverFunc[Any, str, float],
    ] = cbrkit.retrieval.build

    def dump(
        self,
    ) -> None:
        for func in self.embed_providers:
            if isinstance(func, cbrkit.sim.embed.cache):
                func.dump()

    @property
    def embed_providers(self) -> list[EmbedFunc]:
        if not self.config.embedding_models:
            raise ValueError("No embedding models provided.")

        return [
            embed_provider(
                x.model_type,
                x.model_name,
                self.cache_dir,
                self.provider_init.get(x.model_type),
            )
            for x in self.config.embedding_models
        ]

    @property
    def embed(self) -> EmbedFunc:
        embed_funcs = self.embed_providers

        if len(embed_funcs) == 1:
            return embed_funcs[0]

        return cbrkit.sim.embed.concat(embed_funcs)

    @property
    def score(self) -> cbrkit.typing.AnySimFunc[cbrkit.typing.NumpyArray, float]:
        match self.config.similarity_method:
            case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_UNSPECIFIED:
                return cbrkit.sim.embed.cosine()
            case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE:
                return cbrkit.sim.embed.cosine()
            case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_ANGULAR:
                return cbrkit.sim.embed.angular()
            # case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_DOT:
            #     return cbrkit.sim.embed.dot()
            case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_EUCLIDEAN:
                return cbrkit.sim.embed.euclidean()
            case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_MANHATTAN:
                return cbrkit.sim.embed.manhattan()

        raise ValueError("Unknown similarity method.")

    @property
    def similarity(self) -> cbrkit.typing.BatchSimFunc[str, float]:
        return cbrkit.sim.embed.build(self.embed, self.score)

    @property
    def retrieval(self) -> cbrkit.typing.RetrieverFunc[Any, str, float]:
        if (
            self.config.similarity_method
            in (
                nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
                nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_UNSPECIFIED,
            )
            and len(self.config.embedding_models) != 1
        ):
            embedding_model = self.config.embedding_models[0]

            match embedding_model.model_type:
                case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS:
                    # We do NOT use torch_device (cpu/cuda) and instead let sbert auto-detect it
                    return cbrkit.retrieval.sentence_transformers(
                        SentenceTransformer(embedding_model.model_name)
                    )
                case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_COHERE:
                    return cbrkit.retrieval.cohere(embedding_model.model_name)
                case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_VOYAGEAI:
                    return cbrkit.retrieval.voyageai(embedding_model.model_name)

        return self.retrieval_init(self.similarity)

    def doc(
        self,
        batches: Sequence[str],
        pipes_selection: PipeSelection | None = None,
        vectorize: bool = False,
    ) -> Sequence[Doc]:
        if not pipes_selection:
            pipes_selection = {"disable": []}

        if "enable" in pipes_selection:
            pipes_selection["enable"] = tuple(pipes_selection["enable"])

        nlp = spacy.load(self.config.spacy_model)

        with nlp.select_pipes(**pipes_selection):
            docs = list(nlp.pipe(batches))

        if vectorize:
            vecs = self.embed([doc.text for doc in docs])

            for doc, vector in zip(docs, vecs, strict=True):
                doc._.set("vector", vector)

        return docs
