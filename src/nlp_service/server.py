import logging
from dataclasses import dataclass
from pathlib import Path

import arg_services
import grpc
from spacy.tokens import DocBin
from typer import Typer

from . import Nlp, PipeSelection, model, rpc

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass(slots=True)
class NlpService(rpc.NlpServiceServicer):
    nlp: Nlp

    def DocBin(
        self, request: model.DocBinRequest, context: grpc.ServicerContext
    ) -> model.DocBinResponse:
        pipes_selection: PipeSelection | None = None

        if request.WhichOneof("pipes") == "enabled_pipes":
            pipes_selection = {"enable": tuple(request.enabled_pipes.values)}
        elif request.WhichOneof("pipes") == "disabled_pipes":
            pipes_selection = {"disable": tuple(request.disabled_pipes.values)}

        vectorize = (
            model.EmbeddingLevel.EMBEDDING_LEVEL_DOCUMENT in request.embedding_levels
            or model.EmbeddingLevel.EMBEDDING_LEVEL_UNSPECIFIED
            in request.embedding_levels
        )

        docs = self.nlp.pipe_docs(
            request.config,
            request.texts,
            pipes_selection,
            vectorize,
        )

        if request.attributes.values:
            return model.DocBinResponse(
                docbin=DocBin(
                    request.attributes.values, docs=docs, store_user_data=True
                ).to_bytes()
            )

        return model.DocBinResponse(
            docbin=DocBin(docs=docs, store_user_data=True).to_bytes()
        )

    async def Vectors(
        self, request: model.VectorsRequest, context: grpc.ServicerContext
    ) -> model.VectorsResponse:
        embed_func = self.nlp.embed_func(request.config)
        return model.VectorsResponse(
            vectors=[
                model.VectorResponse(document=model.Vector(vector=vector.tolist()))
                for vector in embed_func(request.texts)
            ]
        )

    async def Similarities(
        self, request: model.SimilaritiesRequest, context: grpc.ServicerContext
    ) -> model.SimilaritiesResponse:
        sim_func = self.nlp.sim_func(request.config)
        return model.SimilaritiesResponse(
            similarities=sim_func([(x.text1, x.text2) for x in request.text_tuples])
        )


app = Typer()


@dataclass(slots=True)
class ServiceAdder:
    nlp: Nlp

    def __call__(self, server: grpc.Server):
        """Add the services to the grpc server."""

        rpc.add_NlpServiceServicer_to_server(NlpService(self.nlp), server)


@app.command()
def main(
    host: str = "127.0.0.1",
    port: int = 50100,
    cache: Path | None = None,
):
    arg_services.serve(
        f"{host}:{port}",
        ServiceAdder(Nlp(cache_path=cache)),
        [arg_services.full_service_name(model, "NlpService")],
    )
