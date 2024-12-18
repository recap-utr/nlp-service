import asyncio
from typing import cast

import betterproto
from grpclib.server import Server
from grpclib.utils import graceful_exit
from spacy.tokens import DocBin
from typer import Typer

from . import lib, nlp_pb


class NlpService(nlp_pb.NlpServiceBase):
    async def doc_bin(
        self, doc_bin_request: nlp_pb.DocBinRequest
    ) -> nlp_pb.DocBinResponse:
        pipes_selection: lib.PipeSelection | None = None

        if betterproto.which_one_of(doc_bin_request, "pipes") == "enabled_pipes":
            pipes_selection = {"enable": tuple(doc_bin_request.enabled_pipes.values)}
        elif betterproto.which_one_of(doc_bin_request, "pipes") == "disabled_pipes":
            pipes_selection = {"disable": tuple(doc_bin_request.disabled_pipes.values)}

        vectorize = (
            nlp_pb.EmbeddingLevel.DOCUMENT in doc_bin_request.embedding_levels
            or nlp_pb.EmbeddingLevel.UNSPECIFIED in doc_bin_request.embedding_levels
        )

        docs = lib.docs(
            doc_bin_request.texts,
            doc_bin_request.config,
            pipes_selection,
            vectorize,
        )

        if doc_bin_request.attributes is not None:
            return nlp_pb.DocBinResponse(
                docbin=DocBin(
                    doc_bin_request.attributes.values, docs=docs, store_user_data=True
                ).to_bytes()
            )

        return nlp_pb.DocBinResponse(
            docbin=DocBin(docs=docs, store_user_data=True).to_bytes()
        )

    async def vectors(
        self, vectors_request: nlp_pb.VectorsRequest
    ) -> nlp_pb.VectorsResponse:
        return nlp_pb.VectorsResponse(
            vectors=[
                nlp_pb.VectorResponse(document=nlp_pb.Vector(vector=vector.tolist()))
                for vector in lib.vectors(vectors_request.texts, vectors_request.config)
            ]
        )

    async def similarities(
        self, similarities_request: nlp_pb.SimilaritiesRequest
    ) -> nlp_pb.SimilaritiesResponse:
        return nlp_pb.SimilaritiesResponse(
            similarities=cast(
                list[float],
                lib.similarities(
                    [(x.text1, x.text2) for x in similarities_request.text_tuples],
                    similarities_request.config,
                ),
            )
        )


async def serve(host: str, port: int):
    server = Server([NlpService()])

    with graceful_exit([server]):
        await server.start(host, port)
        print(f"Serving on {host}:{port}")
        await server.wait_closed()


app = Typer()


@app.command()
def run(host: str = "127.0.0.1", port: int = 50100):
    """Main entry point for the server."""
    asyncio.run(serve(host, port))
