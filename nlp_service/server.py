import logging

import betterproto
from grpclib.server import Server
from grpclib.utils import graceful_exit
from spacy.tokens import DocBin

from nlp_service import lib, nlp_pb
from nlp_service.sim_funcs import SimilarityFactory as SimilarityFactory

log = logging.getLogger(__name__)


class NlpService(nlp_pb.NlpServiceBase):
    async def doc_bin(self, req: nlp_pb.DocBinRequest) -> nlp_pb.DocBinResponse:
        res = nlp_pb.DocBinResponse()

        pipes_selection: lib.PipeSelection | None = None

        if betterproto.which_one_of(req, "pipes") == "enabled_pipes":
            pipes_selection = {"enable": tuple(req.enabled_pipes.values)}
        elif betterproto.which_one_of(req, "pipes") == "disabled_pipes":
            pipes_selection = {"disable": tuple(req.disabled_pipes.values)}

        docs = lib.docs(req.texts, req.config, pipes_selection)

        if levels := req.embedding_levels:
            for doc in docs:
                if nlp_pb.EmbeddingLevel.DOCUMENT in levels:
                    doc._.set("vector", doc.vector)
                if nlp_pb.EmbeddingLevel.TOKENS in levels:
                    for token in doc:
                        token._.set("vector", token.vector)
                if nlp_pb.EmbeddingLevel.SENTENCES in levels:
                    for sent in doc.sents:
                        sent._.set("vector", sent.vector)

        if req.attributes is not None:
            res.docbin = DocBin(
                req.attributes.values, docs=docs, store_user_data=True
            ).to_bytes()
        else:
            res.docbin = DocBin(docs=docs, store_user_data=True).to_bytes()

        return res

    async def vectors(self, req: nlp_pb.VectorsRequest) -> nlp_pb.VectorsResponse:
        res = nlp_pb.VectorsResponse(
            vectors=[nlp_pb.VectorResponse() for _ in req.texts]
        )

        for level in req.embedding_levels:
            if (
                level == nlp_pb.EmbeddingLevel.DOCUMENT
                or level == nlp_pb.EmbeddingLevel.UNSPECIFIED
            ):
                text_vectors = lib.vectors(req.texts, req.config, level)

                for i, vector in enumerate(text_vectors):
                    res.vectors[i].document.vector.extend(vector.tolist())

            elif level == nlp_pb.EmbeddingLevel.TOKENS:
                text_vectors = lib.vectors(req.texts, req.config, level)

                for i, vectors in enumerate(text_vectors):
                    res.vectors[i].tokens.extend(
                        nlp_pb.Vector(vector=vector.tolist()) for vector in vectors
                    )

            elif level == nlp_pb.EmbeddingLevel.SENTENCES:
                text_vectors = lib.vectors(req.texts, req.config, level)

                for i, vectors in enumerate(text_vectors):
                    res.vectors[i].tokens.extend(
                        nlp_pb.Vector(vector=vector.tolist()) for vector in vectors
                    )

        return res

    async def similarities(
        self, req: nlp_pb.SimilaritiesRequest
    ) -> nlp_pb.SimilaritiesResponse:
        text_tuples = [(x.text1, x.text2) for x in req.text_tuples]

        return nlp_pb.SimilaritiesResponse(
            similarities=lib.similarities(text_tuples, req.config)
        )


async def main(host: str, port: int):
    server = Server([NlpService()])
    with graceful_exit([server]):
        await server.start(host, port)
        print(f"Serving on {host}:{port}")
        await server.wait_closed()
