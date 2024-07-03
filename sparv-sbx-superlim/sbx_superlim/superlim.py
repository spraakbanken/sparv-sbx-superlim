"""Example for a custom annotator."""

from typing import List, Literal


from sparv.api import Annotation, Config, Output, annotator
from transformers import pipeline

from .helpers import get_label_mapper


@annotator(
    "Identify the stance towards a given topic", 
    language="swe"
)
def argumentation(
    sentence: Annotation = Annotation("<sentence>"),
    out_stance: Output = Output("<sentence>:sbx_superlim.argumentation.stance"),
    out_stance_certainty: Output = Output("<sentence>:sbx_superlim.argumentation.certainty"),
    # TODO: figure out how to pass this as an argument
    hf_model_path: str = "sbx/bert-base-swedish-cased-argumentation_sent"
):
    # TODO: Add as argument and automatically fetch allowed topics from from HF
    topic : List[Literal['abort', 'minimilön', 'marijuanalegalisering', 'dödsstraff', 'kärnkraft', 'kloning']] = 'abort'
    # TODO: Run in batches
    pipe = pipeline("text-classification", model=hf_model_path)
    # TODO: Temporary string formatting hack until training scripts and models are fixed
    output = pipe([f"{s} $ {topic}" for s in sentence.read()])
    label_mapper = get_label_mapper("argumentation_sent", pipe.model.config)
    labels = [label_mapper[o['label']] for o in output]
    out_stance.write([f"{topic}.{l}" for l in labels])
    out_stance_certainty.write([str(o['score']) for o in output])


@annotator("Label the sentiment towards immigration on a continuous 1--5 scale", language="swe")
def absabank_imm(
    sentence: Annotation = Annotation("<sentence>"),
    out_sentiment: Output = Output("<sentence>:sbx_superlim.absabank_imm.sentiment"),
    out_sentiment_certainty: Output = Output("<sentence>:sbx_superlim.absabank_imm.certainty"),
    hf_model_path: str = "sbx/bert-base-swedish-cased-absabank_imm"
):
    raise NotImplementedError
    pipe = pipeline("text-classification", model=hf_model_path)
    output = pipe([s for s in sentence.read()])
    out_sentiment.write([l for l in output['label']])
    out_sentiment_certainty.write([str(o['score']) for o in output])
    


@annotator("Determine whether a sentence is correct Swedish or not")
def dalaj_ged(
    sentence: Annotation = Annotation("<sentence>"),
    out_label : Output = Output("<sentence>:sbx_superlim.dalaj-ged.label"),
    out_certainty: Output = Output("<sentence>:sbx_superlim.dalaj-ged.certainty"),
    hf_model_path: str = "sbx/bert-base-swedish-cased_dalaj-ged"
):  
    pipe = pipeline("text-classification", model=hf_model_path)
    output = pipe([s for s in sentence.read()])
    label_mapper = get_label_mapper("dalaj-ged", pipe.model.config)
    labels = [label_mapper[o['label']] for o in output]
    out_label.write(labels)
    out_certainty.write([str(o['score']) for o in output])    


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
