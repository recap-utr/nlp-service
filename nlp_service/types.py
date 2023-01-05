import typing as t

import nptyping as npt
from numpy.typing import ArrayLike as ArrayLike
from spacy.tokens import Doc, Span, Token
from thinc.types import Floats1d

SpacyObj = t.Union[Doc, Token, Span]
NumpyVector = npt.NDArray[npt.Shape["*"], npt.Float]
NumpyMatrix = npt.NDArray[npt.Shape["*, ..."], npt.Float]
SpacyVector = Floats1d
