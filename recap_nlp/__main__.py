import socket
import typing as t

import typer
import uvicorn

app = typer.Typer()


def _get_open_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


@app.command()
def spacy(host: str, port: t.Optional[int] = None, workers: int = 1):
    if not port:
        port = _get_open_port()

    uvicorn.run(
        "recap_nlp.server:app",
        **{
            "host": host,
            "port": port,
            "workers": workers,
            "log_level": "info",
            "access_log": False,
        },
    )


if __name__ == "__main__":
    app()
