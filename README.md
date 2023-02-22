# NLP Microservice

The goal of this project is to provide a [gRPC](https://grpc.io) server for resource-heavy NLP tasks&mdash;for instance, computing vectors/embeddings for words or sentences.
By using [protobuf](https://developers.google.com/protocol-buffers) internally, our NLP server provides native and strongly typed interfaces for many programming languages.
There are multiple advantages that arise from outsourcing such computations to such a server:

- If multiple apps rely on NLP, the underlying models (which are usually quite large) only need to be loaded once into the main memory.
- All programming languages supported by gRPC get easy access to state-of-the-art NLP architectures (e.g., transformers).
- The logic is consolidated at a central place, drastically decreasing the maintenance effort required.

In addition to the server, we also provide a client containing convenience functions.
This makes it easier for python applications to interact with the gRPC server.
We will discuss the client at the end of this README.

## Installation and Setup

We are using [poetry](https://python-poetry.org) to manage the dependencies.
For easier setup, we also provide a `Dockerfile` and a `docker-compose` specification.

### Docker-Compose (recommended)

You first need to pull this repository.
Then execute the following in the project directory:

```sh
docker-compose build cpu
# OR, if you need extras:
docker-compose build --build-arg EXTRAS="levenshtein transformers" cpu
# Start the CPU-only container
docker-compose up cpu
```

In case you have a **CUDA-enabled GPU**, you can replace `cpu` with `cuda` in the above commands and make full use of your card for advanced models like BERT.

### Poetry (advanced)

```sh
# The server dependencies are optional, thus they have to be installed explicitly.
poetry install --extras server
# To get startet, we recommend to use the default spacy model.
# In case you are dealing with English texts, you can run.
poetry run python -m spacy download core_en_web_lg
# To run the server, you need to specify the address it should listen on.
# In this example, it should liston on port 5678 on localhost.
poetry run python -m nlp_service "127.0.0.1:50100"
```

## General Usage

Once the server is running, you are free to call any of the functions defined in the underlying [protobuf file](https://github.com/recap-utr/arg-services/blob/main/arg_services/nlp/v1/nlp.proto).
The corresponding documentation is located at the [Buf Schema Registry](https://buf.build/recap/arg-services/docs/main:arg_services.nlp.v1).
_Please note:_ The examples here use the Python programming language, but are also directly applicable to any other language supported by gRPC.

```python
import grpc
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc

# First of all, we are creating a channel (i.e., establish a connection to our server)
channel = grpc.insecure_channel("127.0.0.1:5678")

# The channel can now be used to create the actual client (allowing us to call all available functions)
client = nlp_pb2_grpc.NlpServiceStub(channel)

# Now the time has come to prepare our actual function call.
# We will start by creating a very simple NlpConfig with the default spacy model.
# FOr details about the parameters, please have a look at the next section.
config = nlp_pb2.NlpConfig(
  language="en",
  spacy_model="en_core_web_lg",
)

# Next, we will build a request to query vectors from our server.
request = nlp_pb2.VectorsRequest(
  # The first parameter is a list of strings that shall be embedded by our server.
  texts=["What a great tutorial!", "I will definitely recommend this to my friends."],
  # Now we need to specify which embeddings have to be computed. In this example, we create one vector for each text
  embedding_levels=[nlp_pb2.EmbeddingLevel.EMBEDDING_LEVEL_DOCUMENT],
  # The only thing missing now is the spacy configuration we created in the previous step.
  config=config
)

# Having created the request, we can now send it to the server and retrieve the corresponding response.
response = client.Vectors(request)

# Due to technical constraints, we cannot directly transfer numpy arrays, thus we convert our response.
vectors = [np.array(entry.document.vector) for entry in response.vectors]
```

<!-- TODO: Prefer Vectors instead of Similarities for Python to increase performacne. -->

## Advanced Usage

A central piece for all available function is the `NlpConfig` message, allowing you to create even complex embedding models easily.
In addition to [its documentation](https://buf.build/recap/arg-services/docs/main:arg_services.nlp.v1), we will in the following present some examples to demonstrate the possibilities you have.

```python
from arg_services.nlp.v1 import nlp_pb2

# In the example above, we already introduced a quite basic config:
config = nlp_pb2.NlpConfig(
  # You have to provide a language for every config: https://spacy.io/usage/models#languages
  language="en",
  # Also, you need to specify the model that spacy should load: https://spacy.io/models/en
  spacy_model="en_core_web_lg",
)

# A central feature of our library is the possibility to combine multiple embedding models, potentially capturing more contextual information.
config = nlp_pb2.NlpConfig(
  language="en",
  # This parameter expects a list of models. If you pass more than one, the respective vectors are **concatenated** to each other
  # (e.g., two 300-dimensional embeddings will result in a 600-dimensional one).
  # This approach is based on https://arxiv.org/abs/1803.01400
  embedding_models=[
    nlp_pb2.EmbeddingModel(
      # First select the type of model you would like to use (e.g., SBERT/Sentence Transformers).
      model_type=nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS,
      # Then select the actual model.
      # Any of those specified on the website (https://www.sbert.net/docs/pretrained_models.html) are allowed.
      model_name="all-mpnet-base-v2"
    ),
    nlp_pb2.EmbeddingModel(
      # It is also possible to use a standard spacy model
      model_type=nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SPACY,
      model_name="en_core_web_lg",
      # Since we have selected a word embedding (i.e., it cannot directly encode sentences), the token vectors need to be aggregated somehow.
      # The default strategy is to use the arithmetic mean, but you are free to use other strategies (e.g., the geometric mean).
      pooling_type=nlp_pb2.Pooling.POOLING_GMEAN
    ),
    nlp_pb2.EmbeddingModel(
      model_type=nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SPACY,
      model_name="en_core_web_lg",
      # Alternatively, it is also possible to use the generalized mean / power mean.
      # In this example, the selected pmean corresponds to the geometic mean (thus this embedding is identical to the previous one).
      # This approach is based on https://arxiv.org/abs/1803.01400
      pmean=0
    )
  ]
  # This setting is now optional and only needed if you need spacy features (e.g., POS tagging) besides embeddings.
  # spacy_model="en_core_web_lg",
)

# If computing the similarity between strings, you get one additional parameter.
config = nlp_pb2.NlpConfig(
  language="en",
  # To keep the example simple, we will now only use a single spacy model instead of the more powerful embedding models.
  # However, it is of course possible to use them here as well.
  spacy_model="en_core_web_lg",
  # If not specified, we will always use the cosine similarity when comparing two strings.
  # As indicated in a recent paper (https://arxiv.org/abs/1904.13264), you may achieve better results with alternative approaches like DynaMax Jaccard.
  # Please note that this particular method ignores your selected pooling method due to the fact that even plain word embeddings are not pooled at all.
  similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_DYNAMAX_JACCARD
)

# It is also possible to determine a similarity score without the use of embeddings.
config = nlp_pb2.NlpConfig(
  language="en",
  spacy_model="en_core_web_lg",
  # Traditional metric (Jaccard similarity and Levenshtein edit distance) are also available
  similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_EDIT
)
```
