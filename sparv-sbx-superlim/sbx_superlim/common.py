"""
Functions used by annotators

"""

from typing import List

from sparv.api import AllSourceFilenames, Annotation, Text


def prepare_inputs(text : Text, sentence : Annotation = Annotation("<sentence>"), append_str : str = ""):
    txt = text.read()
    inputs : List[str] = []
    for start, end in sentence.read_spans():
        input_string = f"{txt[start:end]}{append_str}"
        inputs.append(input_string)
    return inputs

def pair_files(filenames : AllSourceFilenames) -> List[tuple]:
    pairs : dict = {}
    for fn in filenames:
        stem, _, = fn.split('.')
        pairs.setdefault(stem, []).append(fn)
    assert [l1.split(".")[0] == l2.split(".")[1] for l1, l2 in pairs.values()]
    return [tuple(values) for values in pairs.values()]