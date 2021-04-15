import typing as t
from enum import Enum

from pydantic import BaseModel

# class Embedding(Enum):
#     USE = "use"
#     SENTENCE_TRF = "sentence-trf"


# class QueryModel(BaseModel):
#     embedding_type: Embedding
#     embedding_model: str


# class BaseQuery(BaseModel):
#     language: str
#     spacy_model: str
#     models: t.Tuple[QueryModel, ...] = tuple()
#     token_vectors: bool = False
