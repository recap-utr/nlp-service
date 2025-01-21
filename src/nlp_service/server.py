from dataclasses import dataclass
from pathlib import Path

import arg_services
import grpc
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from spacy.tokens import DocBin
from typer import Typer

from .lib import Nlp, PipeSelection


@dataclass(frozen=True, slots=True)
class NlpService(nlp_pb2_grpc.NlpServiceServicer):
    cache_dir: Path | None
    autodump: bool

    def DocBin(
        self, request: nlp_pb2.DocBinRequest, context: grpc.ServicerContext
    ) -> nlp_pb2.DocBinResponse:
        pipes_selection: PipeSelection | None = None

        if request.WhichOneof("pipes") == "enabled_pipes":
            pipes_selection = {"enable": tuple(request.enabled_pipes.values)}
        elif request.WhichOneof("pipes") == "disabled_pipes":
            pipes_selection = {"disable": tuple(request.disabled_pipes.values)}

        vectorize = (
            nlp_pb2.EmbeddingLevel.EMBEDDING_LEVEL_DOCUMENT in request.embedding_levels
            or nlp_pb2.EmbeddingLevel.EMBEDDING_LEVEL_UNSPECIFIED
            in request.embedding_levels
        )

        nlp = Nlp(request.config, self.cache_dir, self.autodump)
        docs = nlp.doc(
            request.texts,
            pipes_selection,
            vectorize,
        )

        if request.attributes.values:
            return nlp_pb2.DocBinResponse(
                docbin=DocBin(
                    request.attributes.values, docs=docs, store_user_data=True
                ).to_bytes()
            )

        return nlp_pb2.DocBinResponse(
            docbin=DocBin(docs=docs, store_user_data=True).to_bytes()
        )

    async def Vectors(
        self, request: nlp_pb2.VectorsRequest, context: grpc.ServicerContext
    ) -> nlp_pb2.VectorsResponse:
        nlp = Nlp(request.config, self.cache_dir, self.autodump)
        return nlp_pb2.VectorsResponse(
            vectors=[
                nlp_pb2.VectorResponse(document=nlp_pb2.Vector(vector=vector.tolist()))
                for vector in nlp.embed(request.texts)
            ]
        )

    async def Similarities(
        self, request: nlp_pb2.SimilaritiesRequest, context: grpc.ServicerContext
    ) -> nlp_pb2.SimilaritiesResponse:
        nlp = Nlp(request.config, self.cache_dir, self.autodump)
        return nlp_pb2.SimilaritiesResponse(
            similarities=nlp.similarity(
                [(x.text1, x.text2) for x in request.text_tuples]
            )
        )


app = Typer()


@dataclass(frozen=True, slots=True)
class ServiceAdder:
    cache_dir: Path | None
    autodump: bool

    def __call__(self, server: grpc.Server):
        """Add the services to the grpc server."""

        nlp_pb2_grpc.add_NlpServiceServicer_to_server(
            NlpService(self.cache_dir, self.autodump), server
        )


@app.command()
def main(
    host: str = "127.0.0.1",
    port: int = 50100,
    cache_dir: Path | None = None,
    autodump: bool = False,
):
    arg_services.serve(
        f"{host}:{port}",
        ServiceAdder(cache_dir, autodump),
        [arg_services.full_service_name(nlp_pb2, "NlpService")],
    )
