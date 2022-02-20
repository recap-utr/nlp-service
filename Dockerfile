# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
# https://github.com/microsoft/vscode-dev-containers/blob/master/containers/python-3/.devcontainer/Dockerfile
# https://github.com/nautobot/nautobot/blob/develop/docker/Dockerfile

ARG POETRY_VERSION=1.1.13
ARG PYTHON_VERSION=3.9

FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt update && \
    apt install -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"
RUN curl -sS https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python - && \
    poetry config virtualenvs.create false

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root --extras server

RUN python -m spacy download en_core_web_lg \
    && python -m spacy download en_core_web_sm
