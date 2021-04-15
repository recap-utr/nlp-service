import pickle
import sys

import numpy as np
import spacy
from spacy.tokens import Doc, DocBin  # type: ignore

nlp = spacy.load("en_core_web_lg")
doc: Doc = nlp("i am running here.")
print(doc[2].lemma_)

# pickle_bytes = pickle.dumps(doc)
# print("Pickle:", sys.getsizeof(pickle_bytes))
# pickle_doc = pickle.loads(pickle_bytes)
# print(pickle_doc[2].lemma_)


docbin = DocBin()
docbin.add(doc)
docbin_bytes = docbin.to_bytes()
print("Doc to_bytes:", sys.getsizeof(doc.to_bytes()))
# print("Vocab to_bytes:", sys.getsizeof(doc.vocab.to_bytes()))
print("DocBin:", sys.getsizeof(docbin_bytes))
docbin_nlp = spacy.blank("en")
docbin_load = DocBin().from_bytes(docbin_bytes)
docbin_doc = list(docbin_load.get_docs(docbin_nlp.vocab))[0]
print(docbin_doc[2].lemma_)
print("DocBin pickled:", sys.getsizeof(pickle.dumps(docbin_doc)))
print("DocBin to_bytes:", sys.getsizeof(docbin_doc.to_bytes()))

# [
#     "",
#     "IS_ALPHA",
#     "IS_ASCII",
#     "IS_DIGIT",
#     "IS_LOWER",
#     "IS_PUNCT",
#     "IS_SPACE",
#     "IS_TITLE",
#     "IS_UPPER",
#     "LIKE_URL",
#     "LIKE_NUM",
#     "LIKE_EMAIL",
#     "IS_STOP",
#     "IS_OOV_DEPRECATED",
#     "IS_BRACKET",
#     "IS_QUOTE",
#     "IS_LEFT_PUNCT",
#     "IS_RIGHT_PUNCT",
#     "IS_CURRENCY",
#     "ID",
#     "ORTH",
#     "LOWER",
#     "NORM",
#     "SHAPE",
#     "PREFIX",
#     "SUFFIX",
#     "LENGTH",
#     "CLUSTER",
#     "LEMMA",
#     "POS",
#     "TAG",
#     "DEP",
#     "ENT_IOB",
#     "ENT_TYPE",
#     "ENT_ID",
#     "ENT_KB_ID",
#     "HEAD",
#     "SENT_START",
#     "SENT_END",
#     "SPACY",
#     "PROB",
#     "LANG",
#     "MORPH",
#     "IDX",
# ]
# attrs = ["LEMMA"]
# array_nlp = spacy.blank("en")
# np_array = doc.to_array(attrs)
# array_doc = Doc(array_nlp.vocab, words=[t.text for t in doc])
# array_doc.from_array(attrs, np_array)
# print(array_doc[2].lemma_)
