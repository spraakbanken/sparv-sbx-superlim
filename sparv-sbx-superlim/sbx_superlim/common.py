"""
Functions used by annotators

"""

from typing import List, Literal

from sparv.api import AllSourceFilenames, Annotation, Text


def prepare_inputs(text : Text,
                   sentence : Annotation = Annotation("<sentence>"),
                   input_type : Literal['sentence', 'sentence_pair'] = 'sentence',
                   append_str : str = ""
):
    txt = text.read()
    inputs : List[str] = []
    for start, end in sentence.read_spans():
        input_string = f"{txt[start:end]}{append_str}"
        inputs.append(input_string)
    if input_type == 'sentence_pair':
        input_pairs : List[str]
        input_pairs = [inputs[i-1] + inputs[i] for i in range(len(inputs)) if i != 0]
        assert len(input_pairs) == len(inputs) - 1
        return input_pairs
    return inputs


def pair_files(filenames : AllSourceFilenames) -> List[tuple]:
    pairs : dict = {}
    for fn in filenames:
        stem, _, = fn.split('.')
        pairs.setdefault(stem, []).append(fn)
    assert [l1.split(".")[0] == l2.split(".")[1] for l1, l2 in pairs.values()]
    return [tuple(values) for values in pairs.values()]