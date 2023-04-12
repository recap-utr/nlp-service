from __future__ import annotations

import itertools
import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import arg_services
import grpc
import numpy as np
import scipy.stats
import spacy
import typer
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from mashumaro.mixins.dict import DataClassDictMixin
from spacy.cli.download import download as spacy_download
from spacy.language import Language as SpacyLanguage
from spacy.tokens import Doc, DocBin
from thinc.types import Floats1d as SpacyVector

from nlp_service.similarity import SimilarityFactory as SimilarityFactory
from nlp_service.typing import ArrayLike, NumpyMatrix, NumpyVector

log = logging.getLogger(__name__)

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
custom_components = ("embeddings_factory", "similarity_factory")

# TODO: Extract spacy-specific code into its own file.
# Benefit: We could refactor the code s.t. the spacy `nlp` object can be
# imported directly in python applications (eliminating the need to use a server).


@dataclass(frozen=True, eq=True)
class EmbeddingModel(DataClassDictMixin):
    model_type: nlp_pb2.EmbeddingType.ValueType
    model_name: str
    pooling_type: nlp_pb2.Pooling.ValueType
    pmean: float

    @classmethod
    def from_protobuf(cls, pb: nlp_pb2.EmbeddingModel) -> EmbeddingModel:
        return cls(pb.model_type, pb.model_name, pb.pooling_type, pb.pmean)


class ModelBase(ABC):
    @abstractmethod
    def __init__(self, model: EmbeddingModel) -> None:
        pass

    @abstractmethod
    def vector(self, text: str) -> SpacyVector:
        pass


def pmean(vectors: ArrayLike, p: float) -> SpacyVector:
    return np.power(
        np.mean(np.power(np.array(vectors, dtype=complex), p), axis=0), 1 / p
    ).real


class SpacyModel(ModelBase):
    def __init__(self, model: EmbeddingModel):
        self.model = spacy.load(model.model_name)
        self.pooling_type = model.pooling_type
        self.pmean = model.pmean

    def vector(self, text: str) -> SpacyVector:
        with self.model.select_pipes(enable=["senter"]):
            doc = self.model(text)

        if len(doc) > 1:
            if self.pooling_type and self.pooling_type != nlp_pb2.Pooling.POOLING_MEAN:
                return t.cast(
                    SpacyVector,
                    pool_map[self.pooling_type](
                        np.array([token.vector for token in doc])
                    ),
                )
            elif self.pmean:
                return pmean(
                    [t.cast(NumpyVector, token.vector) for token in doc], self.pmean
                )

        return doc.vector


embedding_map: t.Dict[
    nlp_pb2.EmbeddingType.ValueType, t.Callable[[EmbeddingModel], ModelBase]
] = {
    nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SPACY: SpacyModel,
}


try:
    import torch
    from torch.cuda import is_available as is_cuda_available
    from transformers import AutoModel, AutoTokenizer

    torch_device = "cuda" if is_cuda_available() else "cpu"

    class TransformersModel(ModelBase):
        def __init__(self, model: EmbeddingModel):
            # Load model from HuggingFace Hub
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model.model_name, use_fast=True
                )
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(model.model_name)

            self.model = AutoModel.from_pretrained(model.model_name).to(torch_device)

        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
            # First element of model_output contains all token embeddings
            token_embeddings = model_output[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        def vector(self, text: str) -> SpacyVector:
            # Tokenize sentences
            encoded_input = self.tokenizer(
                text, padding=True, truncation=True, return_tensors="pt"
            )
            encoded_input = encoded_input.to(torch_device)
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            # Perform pooling
            sentence_embeddings = self.mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            # Normalize embeddings; is this needed? Normilize? Logit?
            # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings[0].cpu().numpy()

    embedding_map[nlp_pb2.EmbeddingType.EMBEDDING_TYPE_TRANSFORMERS] = TransformersModel

except ModuleNotFoundError:
    log.info("'transformers' not installed.")


try:
    from sentence_transformers import SentenceTransformer
    from torch.cuda import is_available as is_cuda_available

    class SentenceTransformersModel(ModelBase):
        def __init__(self, model: EmbeddingModel):
            self.model = SentenceTransformer(model.model_name, device=torch_device)

        def vector(self, text: str) -> SpacyVector:
            embeddings = t.cast(
                NumpyVector, self.model.encode([text], convert_to_numpy=True)
            )

            return embeddings[0]

    embedding_map[
        nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS
    ] = SentenceTransformersModel

except ModuleNotFoundError:
    log.info("'sentence-transformers' not installed.")


try:
    import tensorflow_hub as hub

    class TensorflowHubModel(ModelBase):
        def __init__(self, model: EmbeddingModel):
            self.model: t.Any = hub.load(model.model_name)

        def vector(self, text: str) -> SpacyVector:
            embeddings: t.Sequence[t.Any] = self.model([text])

            return embeddings[0].numpy()

    embedding_map[
        nlp_pb2.EmbeddingType.EMBEDDING_TYPE_TENSORFLOW_HUB
    ] = TensorflowHubModel

except ModuleNotFoundError:
    log.info("'tensorflow-hub' not installed.")


try:
    import openai

    openai.api_key_path = "./openai_api_key.txt"

    class OpenaiModel(ModelBase):
        def __init__(self, model: EmbeddingModel):
            self.model_name: str = model.model_name

        def vector(self, text: str) -> SpacyVector:
            res: t.Any = openai.Embedding.create(input=[text], model=self.model_name)

            return t.cast(SpacyVector, np.array(res["data"][0]["embedding"]))

    embedding_map[nlp_pb2.EmbeddingType.EMBEDDING_TYPE_OPENAI] = OpenaiModel

except ModuleNotFoundError:
    log.info("'openai' not installed.")


@SpacyLanguage.factory("embeddings_factory")
class EmbeddingsFactory:
    def __init__(self, nlp, name, models):
        self.models: list[ModelBase] = []

        for model_dict in models:
            model = EmbeddingModel.from_dict(model_dict)
            model_class = embedding_map.get(model.model_type)

            if model_class is None:
                raise ValueError(
                    "The packages required for"
                    f" '{nlp_pb2.EmbeddingType.Name(model.model_type)}' are not"
                    " installed"
                )

            if model not in model_cache:
                model_cache[model] = model_class(model)

            self.models.append(model_cache[model])

    def __call__(self, doc):
        if len(self.models) > 0:
            doc.user_hooks["vector"] = self.vector
            doc.user_span_hooks["vector"] = self.vector
            doc.user_token_hooks["vector"] = self.vector

        return doc

    def vector(self, obj) -> SpacyVector:
        vecs = np.array([model.vector(obj.text) for model in self.models])
        return t.cast(SpacyVector, np.concatenate(vecs))


SpacyKey = tuple[str, str, tuple[EmbeddingModel, ...]]
SpacyCache = t.Tuple[SpacyLanguage, dict[str, Doc]]
spacy_cache: dict[SpacyKey, SpacyCache] = {}
model_cache: dict[EmbeddingModel, ModelBase] = {}


def _load_spacy_model(name: t.Optional[str]) -> SpacyLanguage:
    if not name:
        return spacy.blank("en")

    try:
        return spacy.load(name)
    except OSError:
        spacy_download(name)
        return spacy.load(name)


def _load_spacy(config: nlp_pb2.NlpConfig) -> SpacyCache:
    models = tuple(
        EmbeddingModel.from_protobuf(model) for model in config.embedding_models
    )
    key: SpacyKey = (
        config.language,
        config.spacy_model,
        models,
    )

    if key not in spacy_cache:
        nlp = _load_spacy_model(config.spacy_model)

        if models:
            nlp.add_pipe(
                "embeddings_factory",
                last=True,
                config={"models": [model.to_dict() for model in models]},
            )

        if config.similarity_method not in [
            nlp_pb2.SIMILARITY_METHOD_COSINE,
            nlp_pb2.SIMILARITY_METHOD_UNSPECIFIED,
        ]:
            nlp.add_pipe(
                "similarity_factory",
                last=True,
                config={"method": config.similarity_method},
            )

        spacy_cache[key] = (nlp, {})

    return spacy_cache[key]


pool_map: t.Dict[int, t.Callable[[NumpyMatrix], NumpyVector]] = {
    nlp_pb2.Pooling.POOLING_MEAN: lambda x: np.mean(x, axis=0),
    nlp_pb2.Pooling.POOLING_FIRST: lambda x: x[0],
    nlp_pb2.Pooling.POOLING_LAST: lambda x: x[-1],
    nlp_pb2.Pooling.POOLING_MIN: lambda x: np.min(x, axis=0),
    nlp_pb2.Pooling.POOLING_MAX: lambda x: np.max(x, axis=0),
    nlp_pb2.Pooling.POOLING_SUM: lambda x: np.sum(x, axis=0),
    nlp_pb2.Pooling.POOLING_MEDIAN: lambda x: np.median(x, axis=0),
    nlp_pb2.Pooling.POOLING_GMEAN: lambda x: scipy.stats.gmean(x, axis=0),
    nlp_pb2.Pooling.POOLING_HMEAN: lambda x: scipy.stats.hmean(x, axis=0),
}


class NlpService(nlp_pb2_grpc.NlpServiceServicer):
    def DocBin(
        self,
        req: nlp_pb2.DocBinRequest,
        ctx: grpc.ServicerContext,
    ) -> nlp_pb2.DocBinResponse:
        res = nlp_pb2.DocBinResponse()

        arg_services.require_all(["config.language"], req, ctx)

        for model in req.config.embedding_models:
            arg_services.require_all(
                ["model_type", "model_name"],
                model,
                ctx,
                parent="embeddings_factory",
            )

        # TODO: Cache not used due to the ability to enable/disable certain pipes
        nlp, _ = _load_spacy(req.config)

        pipes_selection = {"disable": []}  # if empty, spacy will raise an exception

        if req.HasField("attributes") and not req.attributes.values:
            pipes_selection = {"enable": custom_components}
        elif req.WhichOneof("pipes") == "enabled_pipes":
            pipes_selection = {"enable": list(req.enabled_pipes.values)}
        elif req.WhichOneof("pipes") == "disabled_pipes":
            pipes_selection = {"disable": list(req.disabled_pipes.values)}

        with nlp.select_pipes(**pipes_selection):
            docs = list(nlp.pipe(req.texts))

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

        if req.HasField("attributes"):
            res.docbin = DocBin(
                req.attributes.values, docs=docs, store_user_data=True
            ).to_bytes()
        else:
            res.docbin = DocBin(docs=docs, store_user_data=True).to_bytes()

        return res

    def Vectors(
        self,
        req: nlp_pb2.VectorsRequest,
        ctx: grpc.ServicerContext,
    ) -> nlp_pb2.VectorsResponse:
        res = nlp_pb2.VectorsResponse()

        arg_services.require_all(["config.language", "embedding_levels"], req, ctx)
        arg_services.require_all_repeated(
            "config.embedding_models",
            ["model_type", "model_name"],
            req,
            ctx,
        )

        nlp, doc_cache = _load_spacy(req.config)

        if new_texts := [text for text in req.texts if text not in doc_cache]:
            with nlp.select_pipes(enable=custom_components):
                doc_cache.update(zip(new_texts, nlp.pipe(new_texts)))

        for text in req.texts:
            doc = doc_cache[text]
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

        arg_services.require_all(
            ["config.language", "config.similarity_method"], req, ctx
        )
        arg_services.require_all_repeated(
            "config.embedding_models",
            ["model_type", "model_name"],
            req,
            ctx,
        )
        arg_services.require_all_repeated(
            "text_tuples",
            ["text1", "text2"],
            req,
            ctx,
        )

        nlp, doc_cache = _load_spacy(req.config)
        texts = itertools.chain.from_iterable(
            ((x.text1, x.text2) for x in req.text_tuples)
        )

        if new_texts := [text for text in texts if text not in doc_cache]:
            with nlp.select_pipes(enable=custom_components):
                doc_cache.update(zip(new_texts, nlp.pipe(new_texts)))

        res.similarities.extend(
            doc_cache[x.text1].similarity(doc_cache[x.text2]) for x in req.text_tuples
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
def main(address: str = typer.Argument("127.0.0.1:50100")):
    """Main entry point for the server."""

    arg_services.serve(
        address,
        add_services,
        [arg_services.full_service_name(nlp_pb2, "NlpService")],
    )


if __name__ == "__main__":
    app()
