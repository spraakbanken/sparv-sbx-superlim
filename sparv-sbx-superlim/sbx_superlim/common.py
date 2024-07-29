"""
Functions used by annotators

"""

from typing import List

from sparv.api import Annotation


def prepare_inputs(sentence: Annotation = Annotation("<sentence>"), word: Annotation = Annotation("<token:word>"), append_str : str = ""):
    sentences, _orphans = sentence.get_children(word)
    token_word = list(word.read())
    inputs : List[str] = []
    for s in sentences:
        s_words = []
        for w_idx in s:
            s_words.append(token_word[w_idx])
        sentence_string = " ".join(s_words)
        # TODO: Temporary string formatting hack until training scripts and models are fixed
        input_string = f"{sentence_string}{append_str}"
        inputs.append(input_string)
    return inputs