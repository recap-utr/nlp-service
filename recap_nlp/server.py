from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from concurrent import futures

import grpc
import numpy as np
import recap_schema
import spacy
import tensorflow_hub as hub
import typer
from recap_schema.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc, DocBin, Span, Token  # type: ignore

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
        self.model = spacy.load(model, exclude=spacy_components)
        self.pooling = pooling

    def vector(self, text: str):
        doc = self.model(text)

        if len(doc) > 1 and self.pooling != nlp_pb2.Pooling.POOLING_MEAN_UNSPECIFIED:
            return pool_map[self.pooling]([t.vector for t in doc])

        return doc.vector


@spacy.Language.factory("concat")
class ConcatModel:
    def __init__(self, nlp, name, models):
        self.models = models

    def __call__(self, doc):
        if len(self.models) > 0:
            doc.user_hooks["vector"] = self.vector
            doc.user_span_hooks["vector"] = self.vector
            doc.user_token_hooks["vector"] = self.vector

        return doc

    def vector(self, obj):
        vecs = [model.vector(obj.text) for model in self.models]
        return np.concatenate(vecs)


Doc.set_extension("vector", default=None)
Span.set_extension("vector", default=None)
Token.set_extension("vector", default=None)


def _load_spacy(
    language: nlp_pb2.Language.V,
    spacy_model: str,
    embedding_models: t.Iterable[nlp_pb2.EmbeddingModel],
    **kwargs,
) -> spacy.Language:
    models = []

    if embedding_models:
        for user_model in embedding_models:
            tmp_model = embedding_map[user_model.model_type](
                user_model.model_name, user_model.pooling
            )
            models.append(tmp_model)

    model = spacy.load(spacy_model, **kwargs)
    model.add_pipe(
        "concat",
        last=True,
        config={"models": models},
    )

    return model


pool_map = {
    nlp_pb2.POOLING_MEAN_UNSPECIFIED: np.mean,
    nlp_pb2.POOLING_FIRST: lambda vecs: vecs[0],
    nlp_pb2.POOLING_LAST: lambda vecs: vecs[-1],
    nlp_pb2.POOLING_MIN: np.min,
    nlp_pb2.POOLING_MAX: np.max,
    nlp_pb2.POOLING_SUM: np.sum,
}

embedding_map = {
    nlp_pb2.EMBEDDING_TYPE_SPACY: SpacyModel,
    nlp_pb2.EMBEDDING_TYPE_USE: UseModel,
    nlp_pb2.EMBEDDING_TYPE_SBERT: TransformerModel,
}


def check_embedding_models(
    models: t.Iterable[nlp_pb2.EmbeddingModel], context: grpc.ServicerContext
) -> bool:
    for model in models:
        if model.model_type == nlp_pb2.EMBEDDING_TYPE_UNSPECIFIED:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "You have to specify an embedding type.",
            )
            return False

    return True


class NlpService(nlp_pb2_grpc.NLPServiceServicer):
    def DocBin(
        self,
        req: nlp_pb2.DocBinRequest,
        ctx: grpc.ServicerContext,
    ) -> nlp_pb2.DocBinResponse:
        recap_schema.check_required(["texts", "spacy_model"], req, ctx)
        check_embedding_models(req.embedding_models, ctx)

        res = nlp_pb2.DocBinResponse()

        try:
            nlp = _load_spacy(req.language, req.spacy_model, req.embedding_models)
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
            else:
                res.docbin = DocBin(docs=docs, store_user_data=True).to_bytes()

        except Exception as e:
            recap_schema.grpc_traceback(e, ctx)

        return res

    def Vector(
        self,
        req: nlp_pb2.VectorRequest,
        ctx: grpc.ServicerContext,
    ) -> nlp_pb2.VectorResponse:
        res = nlp_pb2.VectorResponse()

        recap_schema.check_required(
            ["texts", "spacy_model", "embedding_levels"], req, ctx
        )
        check_embedding_models(req.embedding_models, ctx)

        try:
            nlp = _load_spacy(req.language, req.spacy_model, req.embedding_models)
            docs = t.cast(t.List[Doc], list(nlp.pipe(req.texts)))

            for level in req.embedding_levels:
                for doc in docs:
                    if level == nlp_pb2.EMBEDDING_LEVEL_DOCUMENT:
                        res.document.vector.extend(doc.vector.tolist())
                    elif level == nlp_pb2.EMBEDDING_LEVEL_TOKENS:
                        for token in doc:
                            res.tokens.append(
                                nlp_pb2.Vector(vector=token.vector.tolist())
                            )
                    elif level == nlp_pb2.EMBEDDING_LEVEL_SENTENCES:
                        for sent in doc.sents:
                            res.sentences.append(
                                nlp_pb2.Vector(vector=sent.vector.tolist())
                            )

        except Exception as e:
            recap_schema.grpc_traceback(e, ctx)

        return res


# def _get_open_port() -> int:
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.bind(("", 0))
#     s.listen(1)
#     port = s.getsockname()[1]
#     s.close()
#     return port

app = typer.Typer()


@app.command()
def run_server(host: str, port: int, workers: int = 1):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    nlp_pb2_grpc.add_NLPServiceServicer_to_server(NlpService(), server)

    actual_port = server.add_insecure_port(f"{host}:{port}")
    server.start()

    print(f"Serving on {host}:{actual_port}")
    server.wait_for_termination()


if __name__ == "__main__":
    app()
