"""Example for a custom annotator."""

from typing import List, Literal

from sparv.api import Annotation, Config, Output, annotator
from transformers import pipeline


@annotator("Determine if a block of text is for, against or unrelated to a topic", language="swe")
def argumentation(
    sentence: Annotation = Annotation("<sentence>"),
    out_stance: Output = Output("<sentence>:sbx_superlim.argumentation.stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.argumentation.certainty"),
    hf_model_path: str = Config("sbx_superlim.hf_model_path")
):
    # TODO: Add as argument and automatically fetch allowed topics from from HF
    topic : List[Literal['abort', 'minimilön', 'marijuanalegalisering', 'dödsstraff', 'kärnkraft', 'kloning']] = 'abort'
    # TODO: Run in batches
    pipe = pipeline("text-classification", model=hf_model_path)
    # TODO: Temporary string formatting hack until training scripts and models are fixed
    output = pipe([f"{s} $ {topic}" for s in sentence.read()])
    out_stance.write([f"{topic}.{o['label']}" for o in output])
    out_stance_certainty.write([str(o['score']) for o in output])


@annotator("Label the sentiment towards immigration on a continuous 1--5 scale", language="swe")
def absabank_imm(
    sentence: Annotation = Annotation("<sentence>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Determine whether a sentence is correct Swedish or not")
def dalaj_ged(
    sentence: Annotation = Annotation("<sentence>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Determine how related two words are on a continuous scale from 0 to 10")
def supersim_relatedness(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Determine how similar two words are on a continuous scale from 0 to 10")
def supersim_similarity(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Given a word pair A:B and a word C, find a word D such that A:B = C:D")
def sweanalogy(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Given a newspaper article, provide its summary")
def swedn(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Given the question, find the suitable answer within the same category")
def swefaq(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Select the correct synonym")
def swesat_synonyms(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Determine if the target word in two contexts expresses the same sense.")
def swewic(
    tokens: Annotation = Annotation("<token>")
):
    raise NotImplementedError("Sparv use case not yet defined.")


@annotator("Determine the logical relation between two sentences", language="swe")
def swenli(
    sentence: Annotation = Annotation("<sentence>")
):
    raise NotImplementedError("Sparv use case not yet defined.")
