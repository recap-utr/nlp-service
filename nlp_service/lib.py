import itertools
import logging
import tarfile
import typing as t
import urllib.request
import warnings
from abc import ABC, abstractmethod
from collections import abc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.stats
import spacy
from mashumaro.mixins.dict import DataClassDictMixin
from rich.progress import Progress
from spacy import about as spacy_about
from spacy.cli.download import (
    get_latest_version as spacy_get_latest_version,
)
from spacy.cli.download import (
    get_model_filename as spacy_get_model_filename,
)
from spacy.language import Language as SpacyLanguage
from spacy.tokens import Doc
from thinc.types import Floats1d as SpacyVector

from nlp_service import nlp_pb
from nlp_service.sim_funcs import SimilarityFactory as SimilarityFactory
from nlp_service.typing import ArrayLike, NumpyMatrix, NumpyVector

log = logging.getLogger(__name__)

__all__ = ("docs", "doc", "vectors", "vector", "similarities", "similarity")

CACHE_FOLDER = Path.home() / ".cache" / "nlp-service"

torch_device = "cpu"

try:
    from torch.cuda import is_available as is_cuda_available

    torch_device = "cuda" if is_cuda_available() else "cpu"
    log.info(f"Using torch device '{torch_device}'.")
except ModuleNotFoundError:
    log.info("'torch' not installed.")

# https://spacy.io/usage/processing-pipelines#built-in
spacy_components: tuple[str, ...] = (
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
custom_components: tuple[str, ...] = ("embeddings_factory", "similarity_factory")


@dataclass(frozen=True, eq=True)
class SerializableEmbeddingModel(DataClassDictMixin):
    model_type: nlp_pb.EmbeddingType
    model_name: str
    pooling_type: nlp_pb.Pooling
    pmean: float

    @classmethod
    def from_protobuf(cls, pb: nlp_pb.EmbeddingModel) -> "SerializableEmbeddingModel":
        return cls(pb.model_type, pb.model_name, pb.pooling_type, pb.pmean)


class ModelBase(ABC):
    @abstractmethod
    def __init__(self, model: SerializableEmbeddingModel) -> None:
        pass

    @abstractmethod
    def vector(self, text: str) -> SpacyVector:
        pass


def pmean(vectors: ArrayLike, p: float) -> SpacyVector:
    return np.power(
        np.mean(np.power(np.array(vectors, dtype=complex), p), axis=0), 1 / p
    ).real


class SpacyModel(ModelBase):
    def __init__(self, model: SerializableEmbeddingModel):
        self.model = spacy.load(model.model_name)
        self.pooling_type = model.pooling_type
        self.pmean = model.pmean

    def vector(self, text: str) -> SpacyVector:
        with self.model.select_pipes(enable=["senter"]):
            doc = self.model(text)

        if len(doc) > 1:
            if self.pooling_type and self.pooling_type != nlp_pb.Pooling.MEAN:
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


embedding_map: dict[
    nlp_pb.EmbeddingType, t.Callable[[SerializableEmbeddingModel], ModelBase]
] = {
    nlp_pb.EmbeddingType.SPACY: SpacyModel,
}


try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    class TransformersModel(ModelBase):
        def __init__(self, model: SerializableEmbeddingModel):
            # Load model from HuggingFace Hub
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model.model_name, use_fast=True
                )
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(model.model_name)

            self.model = AutoModel.from_pretrained(model.model_name).to(torch_device)

        # Mean nlp_pb.Pooling - Take attention mask into account for correct averaging
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

    embedding_map[nlp_pb.EmbeddingType.TRANSFORMERS] = TransformersModel

except ModuleNotFoundError:
    log.info("'transformers' not installed.")


try:
    from sentence_transformers import SentenceTransformer

    class SentenceTransformersModel(ModelBase):
        def __init__(self, model: SerializableEmbeddingModel):
            self.model = SentenceTransformer(model.model_name, device=torch_device)

        def vector(self, text: str) -> SpacyVector:
            embeddings = t.cast(
                NumpyVector, self.model.encode([text], convert_to_numpy=True)
            )

            return embeddings[0]

    embedding_map[
        nlp_pb.EmbeddingType.SENTENCE_TRANSFORMERS
    ] = SentenceTransformersModel

except ModuleNotFoundError:
    log.info("'sentence-transformers' not installed.")


try:
    import openai

    class OpenaiModel(ModelBase):
        def __init__(self, model: SerializableEmbeddingModel):
            self.model_name: str = model.model_name
            self.client = openai.Client()

        def vector(self, text: str) -> SpacyVector:
            res = self.client.embeddings.create(input=[text], model=self.model_name)

            return t.cast(SpacyVector, np.array(res.data[0].embedding))

    embedding_map[nlp_pb.EmbeddingType.OPENAI] = OpenaiModel

except ModuleNotFoundError:
    log.info("'openai' not installed.")


@SpacyLanguage.factory("embeddings_factory")
class EmbeddingsFactory:
    def __init__(self, nlp, name, models):
        self.models: list[ModelBase] = []

        for model_dict in models:
            model = SerializableEmbeddingModel.from_dict(model_dict)
            model_class = embedding_map.get(model.model_type)

            if model_class is None:
                raise ValueError(
                    "The packages required for"
                    f" '{model.model_type.name}' are not"
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


SpacyKey = tuple[str, str, tuple[SerializableEmbeddingModel, ...]]
SpacyCache = t.Tuple[SpacyLanguage, dict[str, Doc]]
spacy_cache: dict[SpacyKey, SpacyCache] = {}
model_cache: dict[SerializableEmbeddingModel, ModelBase] = {}


class UrlReportHook:
    def __init__(self, progress: Progress, name: str):
        self.task = None
        self.progress = progress
        self.name = name

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.task is None:
            self.task = self.progress.add_task(
                f"Downloading {self.name}...", total=total_size
            )

        downloaded = block_num * block_size

        if downloaded < total_size:
            self.progress.update(self.task, completed=downloaded)

        if self.progress.finished:
            self.task = None


def get_tarfile_members(
    tf: tarfile.TarFile, prefix: str
) -> t.Generator[tarfile.TarInfo, None, None]:
    prefix_len = len(prefix)

    for member in tf.getmembers():
        if member.path.startswith(prefix):
            member.path = member.path[prefix_len:]
            yield member


def _load_spacy_model(name: str | None) -> SpacyLanguage:
    if not name:
        return spacy.blank("en")

    version = spacy_get_latest_version(name)
    filename = spacy_get_model_filename(name, version, sdist=True)
    versioned_name = f"{name}-{version}"
    path = CACHE_FOLDER / "spacy" / versioned_name
    tmpfile = path.with_suffix(".tar.gz")

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        download_url = f"{spacy_about.__download_url__}/{filename}"
        with Progress() as progress:
            urllib.request.urlretrieve(
                download_url, tmpfile, UrlReportHook(progress, versioned_name)
            )

        with tarfile.open(tmpfile, mode="r:gz") as tf:
            member_prefix = f"{versioned_name}/{name}/{versioned_name}/"
            members = get_tarfile_members(tf, member_prefix)
            tf.extractall(path=path, members=members)

        tmpfile.unlink()

    return spacy.load(path)


def load_spacy(config: nlp_pb.NlpConfig) -> SpacyCache:
    models = tuple(
        SerializableEmbeddingModel.from_protobuf(model)
        for model in config.embedding_models
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
            nlp_pb.SimilarityMethod.COSINE,
            nlp_pb.SimilarityMethod.UNSPECIFIED,
        ]:
            nlp.add_pipe(
                "similarity_factory",
                last=True,
                config={"method": config.similarity_method},
            )

        spacy_cache[key] = (nlp, {})

    return spacy_cache[key]


pool_map: dict[nlp_pb.Pooling, t.Callable[[NumpyMatrix], NumpyVector]] = {
    nlp_pb.Pooling.MEAN: lambda x: np.mean(x, axis=0),
    nlp_pb.Pooling.FIRST: lambda x: x[0],
    nlp_pb.Pooling.LAST: lambda x: x[-1],
    nlp_pb.Pooling.MIN: lambda x: np.min(x, axis=0),
    nlp_pb.Pooling.MAX: lambda x: np.max(x, axis=0),
    nlp_pb.Pooling.SUM: lambda x: np.sum(x, axis=0),
    nlp_pb.Pooling.MEDIAN: lambda x: np.median(x, axis=0),
    nlp_pb.Pooling.GMEAN: lambda x: scipy.stats.gmean(x, axis=0),
    nlp_pb.Pooling.HMEAN: lambda x: scipy.stats.hmean(x, axis=0),
}


class PipeSelection(t.TypedDict, total=False):
    enable: list[str] | tuple[str, ...]
    disable: list[str] | tuple[str, ...]


def docs(
    texts: abc.Iterable[str],
    config: nlp_pb.NlpConfig,
    pipes_selection: PipeSelection | None = None,
) -> list[Doc]:
    if not pipes_selection:
        pipes_selection = {"disable": []}  # if empty, spacy will raise an exception

    if "enable" in pipes_selection:
        pipes_selection["enable"] = tuple(pipes_selection["enable"]) + custom_components

    # arg_services.require_all(["config.language"], req, ctx)

    # for model in req.config.embedding_models:
    #     arg_services.require_all(
    #         ["model_type", "model_name"],
    #         model,
    #         ctx,
    #         parent="embeddings_factory",
    #     )

    # TODO: Cache not used due to the ability to enable/disable certain pipes
    nlp, _ = load_spacy(config)

    with nlp.select_pipes(**pipes_selection):
        return list(nlp.pipe(texts))


def doc(
    text: str,
    config: nlp_pb.NlpConfig,
    pipes_selection: PipeSelection | None = None,
) -> Doc:
    return docs([text], config, pipes_selection)[0]


EmbeddingLevelSingle = t.Literal[
    nlp_pb.EmbeddingLevel.DOCUMENT, nlp_pb.EmbeddingLevel.UNSPECIFIED
]
EmbeddingLevelMulti = t.Literal[
    nlp_pb.EmbeddingLevel.SENTENCES, nlp_pb.EmbeddingLevel.TOKENS
]


@t.overload
def vectors(
    texts: abc.Iterable[str],
    config: nlp_pb.NlpConfig,
    embedding_level: EmbeddingLevelSingle,
) -> list[SpacyVector]:
    ...


@t.overload
def vectors(
    texts: abc.Iterable[str],
    config: nlp_pb.NlpConfig,
    embedding_level: EmbeddingLevelMulti,
) -> list[list[SpacyVector]]:
    ...


@t.overload
def vectors(
    texts: abc.Iterable[str],
    config: nlp_pb.NlpConfig,
    embedding_level: nlp_pb.EmbeddingLevel,
) -> list[SpacyVector] | list[list[SpacyVector]]:
    ...


def vectors(
    texts: abc.Iterable[str],
    config: nlp_pb.NlpConfig,
    embedding_level: nlp_pb.EmbeddingLevel = nlp_pb.EmbeddingLevel.UNSPECIFIED,
) -> list[SpacyVector] | list[list[SpacyVector]]:
    # arg_services.require_all(["config.language", "embedding_levels"], req, ctx)
    # arg_services.require_all_repeated(
    #     "config.embedding_models", ["model_type", "model_name"], req, ctx
    # )

    nlp, doc_cache = load_spacy(config)

    if new_texts := [text for text in texts if text not in doc_cache]:
        with nlp.select_pipes(enable=custom_components):
            doc_cache.update(zip(new_texts, nlp.pipe(new_texts)))

    if embedding_level in {
        nlp_pb.EmbeddingLevel.DOCUMENT,
        nlp_pb.EmbeddingLevel.UNSPECIFIED,
    }:
        doc_vecs: list[SpacyVector] = []

        for text in texts:
            doc = doc_cache[text]
            doc_vecs.append(doc.vector)

        return doc_vecs
    else:
        span_vecs: list[list[SpacyVector]] = []

        for text in texts:
            doc = doc_cache[text]

            if embedding_level == nlp_pb.EmbeddingLevel.TOKENS:
                span_vecs.append([token.vector for token in doc])
            elif embedding_level == nlp_pb.EmbeddingLevel.SENTENCES:
                span_vecs.append([sent.vector for sent in doc.sents])

        return span_vecs


@t.overload
def vector(
    text: str, config: nlp_pb.NlpConfig, embedding_level: EmbeddingLevelSingle
) -> SpacyVector:
    ...


@t.overload
def vector(
    text: str, config: nlp_pb.NlpConfig, embedding_level: EmbeddingLevelMulti
) -> list[SpacyVector]:
    ...


@t.overload
def vector(
    text: str, config: nlp_pb.NlpConfig, embedding_level: nlp_pb.EmbeddingLevel
) -> SpacyVector | list[SpacyVector]:
    ...


def vector(
    text: str,
    config: nlp_pb.NlpConfig,
    embedding_level: nlp_pb.EmbeddingLevel = nlp_pb.EmbeddingLevel.UNSPECIFIED,
) -> SpacyVector | list[SpacyVector]:
    return vectors([text], config, embedding_level)[0]


def similarities(
    text_tuples: abc.Iterable[tuple[str, str]], config: nlp_pb.NlpConfig
) -> list[float]:
    # arg_services.require_all(["config.language", "config.similarity_method"], req, ctx)
    # arg_services.require_all_repeated(
    #     "config.embedding_models", ["model_type", "model_name"], req, ctx
    # )
    # arg_services.require_all_repeated("text_tuples", ["text1", "text2"], req, ctx)

    nlp, doc_cache = load_spacy(config)
    texts = itertools.chain.from_iterable((x[0], x[1]) for x in text_tuples)

    if new_texts := [text for text in texts if text not in doc_cache]:
        with nlp.select_pipes(enable=custom_components):
            doc_cache.update(zip(new_texts, nlp.pipe(new_texts)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        return [doc_cache[x[0]].similarity(doc_cache[x[1]]) for x in text_tuples]


def similarity(text1: str, text2: str, config: nlp_pb.NlpConfig) -> float:
    return similarities([(text1, text2)], config)[0]
