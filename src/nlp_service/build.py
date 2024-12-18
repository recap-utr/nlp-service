import functools
from typing import Any

import cbrkit
from arg_services.nlp.v1 import nlp_pb2

type EmbedFunc = cbrkit.typing.BatchConversionFunc[str, cbrkit.typing.NumpyArray]


@functools.cache
def embed_provider(model_type: nlp_pb2.EmbeddingType, model_name: str) -> EmbedFunc:
    match model_type:
        case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SPACY:
            embed_func = cbrkit.sim.embed.spacy(model_name)
        case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS:
            embed_func = cbrkit.sim.embed.sentence_transformers(model_name)
        case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_OPENAI:
            embed_func = cbrkit.sim.embed.openai(model_name)
        # case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_COHERE:
        #     embed_func = cbrkit.sim.embed.cohere(model_name)
        # case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_VOYAGEAI:
        #     embed_func = cbrkit.sim.embed.voyageai(model_name)
        # case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_OLLAMA:
        #     embed_func = cbrkit.sim.embed.ollama(model_name)
        case _:
            raise ValueError("Unknown embedding model type.")

    return cbrkit.sim.embed.cache(embed_func)


def embed(config: nlp_pb2.NlpConfig) -> EmbedFunc:
    if not config.embedding_models:
        raise ValueError("No embedding models provided.")

    if len(config.embedding_models) == 1:
        provider = config.embedding_models[0]
        return embed_provider(provider.model_type, provider.model_name)

    return cbrkit.sim.embed.concat(
        [embed_provider(x.model_type, x.model_name) for x in config.embedding_models]
    )


def similarity(config: nlp_pb2.NlpConfig) -> cbrkit.typing.BatchSimFunc[str, float]:
    score_func: cbrkit.typing.AnySimFunc[cbrkit.typing.NumpyArray, float]

    match config.similarity_method:
        case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_UNSPECIFIED:
            score_func = cbrkit.sim.embed.cosine()
        case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE:
            score_func = cbrkit.sim.embed.cosine()
        case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_ANGULAR:
            score_func = cbrkit.sim.embed.angular()
        # case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_DOT:
        #     score_func = cbrkit.sim.embed.dot()
        # case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_EUCLIDEAN:
        #     score_func = cbrkit.sim.embed.euclidean()
        # case nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_MANHATTAN:
        #     score_func = cbrkit.sim.embed.manhattan()
        case _:
            raise ValueError("Unknown similarity method.")

    return cbrkit.sim.embed.build(embed(config), score_func)


def retrieval(
    config: nlp_pb2.NlpConfig,
) -> cbrkit.typing.RetrieverFunc[Any, str, float]:
    if (
        config.similarity_method
        not in (
            nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
            nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_UNSPECIFIED,
        )
        or len(config.embedding_models) != 1
    ):
        return cbrkit.retrieval.build(similarity(config))

    embedding_model = config.embedding_models[0]

    match embedding_model.model_type:
        case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS:
            return cbrkit.retrieval.sentence_transformers(embedding_model.model_name)
        # case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_COHERE:
        #     return cbrkit.retrieval.cohere(embedding_model.model_name)
        # case nlp_pb2.EmbeddingType.EMBEDDING_TYPE_VOYAGEAI:
        #     return cbrkit.retrieval.voyageai(embedding_model.model_name)

    raise ValueError("Invalid embedding model for retriever.")
