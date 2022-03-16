# https://towardsdatascience.com/a-complete-guide-to-building-a-docker-image-serving-a-machine-learning-system-in-production-d8b5b0533bde
ARG POETRY_VERSION=1.1.12
ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=11.5.1
ARG UBUNTU_VERSION=20.04

# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PY=python${PYTHON_VERSION}

WORKDIR /app

RUN apt update \
    && apt install -y curl build-essential software-properties-common python3-pip \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt install -y ${PY}-dev \
    && apt clean && rm -rf /var/lib/apt/lists*

ENV PATH="/root/.local/bin:${PATH}"
RUN curl -sS https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | ${PY} - && \
    poetry config virtualenvs.create false

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi

RUN python -m spacy download en_core_web_lg \
    && python -m spacy download en_core_web_sm
