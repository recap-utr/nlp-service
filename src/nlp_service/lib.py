import logging
import typing as t
from collections import abc
from dataclasses import dataclass

import cbrkit
import numpy.typing as npt
import spacy
from arg_services.nlp.v1 import nlp_pb2
from spacy.tokens import Doc
from torch.cuda import is_available as is_cuda_available

logger = logging.getLogger(__name__)

torch_device = "cuda" if is_cuda_available() else "cpu"
logger.info(f"Using torch device '{torch_device}'.")


@dataclass(frozen=True, slots=True, eq=True)
class EmbedKey:
    model_type: nlp_pb2.EmbeddingType
    model_name: str


type EmbedFunc = cbrkit.typing.BatchConversionFunc[str, npt.NDArray]

embed_cache: dict[EmbedKey, EmbedFunc] = {}


def build_embed_func(
    config: nlp_pb2.EmbeddingModel,
) -> EmbedFunc:
    key = EmbedKey(config.model_type, config.model_name)

    if key not in embed_cache:
        embed_func: EmbedFunc

        match config.model_type:
            case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SPACY:
                embed_func = cbrkit.sim.embed.spacy(config.model_name)
            case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS:
                embed_func = cbrkit.sim.embed.sentence_transformers(config.model_name)
            case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_OPENAI:
                embed_func = cbrkit.sim.embed.openai(config.model_name)
            case _:
                raise ValueError("Unknown embedding model type.")

        embed_cache[key] = cbrkit.sim.embed.cache(embed_func)

    return embed_cache[key]


def build_concat_embed_func(
    config: nlp_pb2.NlpConfig,
) -> EmbedFunc:
    if not config.embedding_models:
        raise ValueError("No embedding models provided.")

    if len(config.embedding_models) == 1:
        return build_embed_func(config.embedding_models[0])

    return cbrkit.sim.embed.concat(
        [build_embed_func(x) for x in config.embedding_models]
    )


def build_sim_func(config: nlp_pb2.NlpConfig) -> cbrkit.typing.BatchSimFunc[str, float]:
    score_func: cbrkit.typing.AnySimFunc[npt.NDArray, float]

    match config.similarity_method:
        case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_UNSPECIFIED:
            score_func = cbrkit.sim.embed.cosine()
        case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE:
            score_func = cbrkit.sim.embed.cosine()
        case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_ANGULAR:
            score_func = cbrkit.sim.embed.angular()
        case _:
            raise ValueError("Unknown similarity method.")

    return cbrkit.sim.embed.build(build_concat_embed_func(config), score_func)


class PipeSelection(t.TypedDict, total=False):
    enable: abc.Sequence[str]
    disable: abc.Sequence[str]


def vectors(
    texts: abc.Sequence[str], config: nlp_pb2.NlpConfig
) -> abc.Sequence[npt.NDArray]:
    func = build_concat_embed_func(config)
    return func(texts)


def vector(text: str, config: nlp_pb2.NlpConfig) -> npt.NDArray:
    return vectors([text], config)[0]


def docs(
    texts: abc.Sequence[str],
    config: nlp_pb2.NlpConfig,
    pipes_selection: PipeSelection | None = None,
    vectorize: bool = False,
) -> abc.Sequence[Doc]:
    if not pipes_selection:
        pipes_selection = {"disable": []}

    if "enable" in pipes_selection:
        pipes_selection["enable"] = tuple(pipes_selection["enable"])

    nlp = spacy.load(config.spacy_model)

    with nlp.select_pipes(**pipes_selection):
        docs = list(nlp.pipe(texts))

    if vectorize:
        vecs = vectors([doc.text for doc in docs], config)

        for doc, vector in zip(docs, vecs, strict=True):
            doc._.set("vector", vector)

    return docs


def doc(
    text: str,
    config: nlp_pb2.NlpConfig,
    pipes_selection: PipeSelection | None = None,
) -> Doc:
    return docs([text], config, pipes_selection)[0]


def similarities(
    text_tuples: abc.Sequence[tuple[str, str]], config: nlp_pb2.NlpConfig
) -> abc.Sequence[float]:
    func = build_sim_func(config)
    return func(text_tuples)


def similarity(text1: str, text2: str, config: nlp_pb2.NlpConfig) -> float:
    return similarities([(text1, text2)], config)[0]
