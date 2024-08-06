"""
Functions used by annotators

"""

from typing import List

from sparv.api import Annotation, Text


def prepare_inputs(text : Text, sentence : Annotation = Annotation("<sentence>"), append_str : str = ""):
    txt = text.read()
    inputs : List[str] = []
    for start, end in sentence.read_spans():
        input_string = f"{txt[start:end]}{append_str}"
        inputs.append(input_string)
    return inputs