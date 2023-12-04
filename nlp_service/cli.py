import asyncio
import logging
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich import print

from nlp_service import lib, nlp_pb, server
from nlp_service.sim_funcs import SimilarityFactory as SimilarityFactory

log = logging.getLogger(__name__)

app = typer.Typer()


class NlpConfig(str, Enum):
    default = "default"
    strf = "strf"
    openai = "openai"


NLP_CONFIG: dict[NlpConfig, nlp_pb.NlpConfig] = {
    NlpConfig.default: nlp_pb.NlpConfig(
        language="en",
        spacy_model="en_core_web_lg",
        similarity_method=nlp_pb.SimilarityMethod.COSINE,
    ),
    NlpConfig.strf: nlp_pb.NlpConfig(
        language="en",
        similarity_method=nlp_pb.SimilarityMethod.COSINE,
        embedding_models=[
            nlp_pb.EmbeddingModel(
                model_type=nlp_pb.EmbeddingType.TRANSFORMERS,
                model_name="multi-qa-MiniLM-L6-cos-v1",
                pooling_type=nlp_pb.Pooling.MEAN,
            )
        ],
    ),
    NlpConfig.openai: nlp_pb.NlpConfig(
        language="en",
        similarity_method=nlp_pb.SimilarityMethod.COSINE,
        embedding_models=[
            nlp_pb.EmbeddingModel(
                model_type=nlp_pb.EmbeddingType.OPENAI,
                model_name="text-embedding-ada-002",
                pooling_type=nlp_pb.Pooling.MEAN,
            )
        ],
    ),
}


@app.command()
def clear_cache():
    shutil.rmtree(lib.CACHE_FOLDER)


@app.command()
def retrieve(
    cases_folder: Path,
    query: Optional[str] = None,
    query_file: Optional[Path] = None,
    include_pattern: str = "**/*.txt",
    exclude_pattern: str = "",
    show_contents: bool = False,
    limit: int = sys.maxsize,
    config: NlpConfig = NlpConfig.default,
):
    """Find the most similar cases to a given query."""

    assert (
        query is not None or query_file is not None
    ), "Either query or query_file must be provided"
    assert not (
        query is not None and query_file is not None
    ), "Only one of query or query_file must be provided"

    query_name = "stdin"
    query_text = query or ""

    if query_file is not None:
        query_name = query_file.name
        query_text = query_file.read_text()

    print("Retrieving ranking...")
    print(f"Query '{query_name}'")

    if show_contents:
        print(f"{query_text}")
        print()

    cases = {
        str(case_file.relative_to(cases_folder)): case_file.read_text()
        for case_file in cases_folder.glob(include_pattern)
        if not case_file.match(exclude_pattern)
    }

    sim_values = lib.similarities(
        [(query_name, case_name) for case_name in cases.keys()], NLP_CONFIG[config]
    )
    sim_tuples = zip(cases, sim_values)
    ranking = sorted(sim_tuples, key=lambda x: x[1], reverse=True)

    for i, (case_name, sim) in enumerate(ranking[:limit], start=1):
        print(f"{i}. '{case_name}' ({sim:.3f})")

        if show_contents:
            print(f"{cases[case_name]}")
            print()


@app.command()
def serve(host: str = "127.0.0.1", port: int = 50100):
    """Main entry point for the server."""
    asyncio.run(server.main(host, port))


if __name__ == "__main__":
    app()
