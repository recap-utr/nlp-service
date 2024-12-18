from typing import cast

import arg_services
import grpc
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from spacy.tokens import DocBin
from typer import Typer

from . import apply


class NlpService(nlp_pb2_grpc.NlpServiceServicer):
    def DocBin(
        self, request: nlp_pb2.DocBinRequest, context: grpc.ServicerContext
    ) -> nlp_pb2.DocBinResponse:
        pipes_selection: apply.PipeSelection | None = None

        if request.WhichOneof("pipes") == "enabled_pipes":
            pipes_selection = {"enable": tuple(request.enabled_pipes.values)}
        elif request.WhichOneof("pipes") == "disabled_pipes":
            pipes_selection = {"disable": tuple(request.disabled_pipes.values)}

        vectorize = (
            nlp_pb2.EmbeddingLevel.EMBEDDING_LEVEL_DOCUMENT in request.embedding_levels
            or nlp_pb2.EmbeddingLevel.EMBEDDING_LEVEL_UNSPECIFIED
            in request.embedding_levels
        )

        docs = apply.docs(
            request.texts,
            request.config,
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
        return nlp_pb2.VectorsResponse(
            vectors=[
                nlp_pb2.VectorResponse(document=nlp_pb2.Vector(vector=vector.tolist()))
                for vector in apply.vectors(request.texts, request.config)
            ]
        )

    async def Similarities(
        self, request: nlp_pb2.SimilaritiesRequest, context: grpc.ServicerContext
    ) -> nlp_pb2.SimilaritiesResponse:
        return nlp_pb2.SimilaritiesResponse(
            similarities=cast(
                list[float],
                apply.similarities(
                    [(x.text1, x.text2) for x in request.text_tuples],
                    request.config,
                ),
            )
        )


app = Typer()


def add_services(server: grpc.Server):
    """Add the services to the grpc server."""

    nlp_pb2_grpc.add_NlpServiceServicer_to_server(NlpService(), server)


@app.command()
def main(host: str = "127.0.0.1", port: int = 50100):
    arg_services.serve(
        f"{host}:{port}",
        add_services,
        [arg_services.full_service_name(nlp_pb2, "NlpService")],
    )
