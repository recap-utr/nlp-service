import arg_services_helper
import grpc
import typer
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc

from nlp_service.server import NlpService

app = typer.Typer()


def add_services(server: grpc.Server):
    """Add the services to the grpc server."""

    nlp_pb2_grpc.add_NlpServiceServicer_to_server(NlpService(), server)
    # topic_modeling_pb2_grpc.add_TopicModelingServiceServicer_to_server(
    #     TopicModelingService(), server
    # )


@app.command()
def main(host: str, port: int, processes: int = 1):
    """Main entry point for the server."""

    arg_services_helper.serve(
        host,
        port,
        add_services,
        processes=processes,
        reflection_services=[
            arg_services_helper.full_service_name(nlp_pb2, "NlpService"),
        ],
    )


if __name__ == "__main__":
    app()
