ARG PYTHON_VERSION=3.9

FROM python:${PYTHON_VERSION}-slim

ARG POETRY_VERSION=1.3.1
# ARG SPACY_MODEL=en_core_web_lg

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /dependencies

RUN apt update \
    && apt install -y curl build-essential \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"
RUN curl -sSL https://install.python-poetry.org | python - \
    && poetry config virtualenvs.in-project true

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --extras server

ENV VIRTUAL_ENV="/dependencies/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
WORKDIR /app

# RUN python -m spacy download ${SPACY_MODEL}

COPY . .

CMD ["python", "-m", "nlp_service", "0.0.0.0:50051"]
